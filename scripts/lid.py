"""Identify the language of each Phase 2 clip, from the audio.

Coverage is the dominant unknown in every plan on the table: the estimated
Phase 2 language mix came from character n-grams over our own ASR output, which
inherits the model's biases and is blind to Ethiopic entirely. This measures the
mix from the audio instead.

Language ID is a much easier problem than transcription -- languages separate on
phonotactics and prosody long before words are recoverable -- so a frozen
w2v-BERT encoder with a logistic regression on mean-pooled embeddings is enough.
No fine-tuning, and the whole thing runs in well under an hour.

The classifier is trained on the corpus itself, so it is in-domain in a way an
off-the-shelf LID is not: same collection, same task, same recording conditions
as Phase 2. The corresponding weakness is that it can only answer with one of
the 19 languages it was shown -- a clip in something else gets confidently
mislabelled. `--min-conf` and the reported confidence distribution are there to
expose that rather than hide it; cross-check against facebook/mms-lid-4017,
which is independent, before trusting a surprising result.

    python scripts/lid.py --per-lang 150 --predict-dir data/phase_2/audio
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import torch                                                # noqa: E402

from infer_phase2 import read_wav                           # noqa: E402
from waxal import data as wdata                             # noqa: E402
from waxal import hw                                        # noqa: E402

BASE = "facebook/w2v-bert-2.0"
# Purpose-built LID. Measured 2026-07-28, a linear probe on mean-pooled
# w2v-BERT features reached only 52.9% on held-out speakers across 19
# languages, confusing exactly the related pairs that matter here (sog->lug,
# ewe->kpo, dga->ewe). MMS-LID is trained for this task and emits ISO 639-3
# codes, which is already how the corpus names its languages.
MMS = "facebook/mms-lid-4017"

# Language ID needs a few seconds of speech, not the whole clip. Truncating
# keeps the embedding pass cheap and makes every clip the same cost.
CLIP_S = 10.0


def load_encoder(device):
    import transformers
    fe = transformers.AutoFeatureExtractor.from_pretrained(BASE)
    model = transformers.Wav2Vec2BertModel.from_pretrained(BASE).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return fe, model


@torch.no_grad()
def embed(arrays, fe, model, device, dtype, batch_size=8):
    """Mean-pooled encoder states, masked so padding contributes nothing."""
    out = []
    for i in range(0, len(arrays), batch_size):
        chunk = [a[:int(CLIP_S * wdata.SR)] for a in arrays[i:i + batch_size]]
        inp = fe(chunk, sampling_rate=wdata.SR, return_tensors="pt",
                 padding=True).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=device.type == "cuda"):
            h = model(**inp).last_hidden_state.float()
        mask = inp.get("attention_mask")
        if mask is not None:
            # The encoder subsamples in time, so the input mask does not line up
            # with the output frames; interpolate rather than assume.
            m = torch.nn.functional.interpolate(
                mask[:, None, :].float(), size=h.shape[1], mode="nearest")[:, 0]
            h = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(1)
        out.append(h.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  embedded {min(i + batch_size, len(arrays)):,}/{len(arrays):,}",
                  flush=True)
    return np.concatenate(out)


def load_mms(model_id, device):
    import transformers
    fe = transformers.AutoFeatureExtractor.from_pretrained(model_id)
    model = transformers.Wav2Vec2ForSequenceClassification.from_pretrained(
        model_id).eval().to(device)
    return fe, model


@torch.no_grad()
def mms_predict(arrays, fe, model, device, langs, batch_size=4):
    """Per-clip language, both unrestricted and restricted to the corpus's 19.

    Unrestricted says what MMS actually hears, including languages the corpus
    does not contain -- which is the check the in-domain probe structurally
    cannot perform. Restricted is the answer to use for training decisions,
    since those are the only languages we can get labeled audio for.
    """
    id2label = model.config.id2label
    keep = [i for i, l in id2label.items() if l in set(langs)]
    if not keep:
        raise SystemExit(
            f"{model.config._name_or_path} labels none of {sorted(set(langs))}. "
            f"Try --mms-model facebook/mms-lid-4017")
    missing = sorted(set(langs) - {id2label[i] for i in keep})
    if missing:
        print(f"  not covered by this LID model: {missing}")

    free, restr, conf = [], [], []
    for i in range(0, len(arrays), batch_size):
        chunk = [a[:int(CLIP_S * wdata.SR)] for a in arrays[i:i + batch_size]]
        inp = fe(chunk, sampling_rate=wdata.SR, return_tensors="pt",
                 padding=True).to(device)
        logits = model(**inp).logits.float()
        p = logits.softmax(-1)
        for row in p:
            free.append(id2label[int(row.argmax())])
            sub = row[keep]
            j = int(sub.argmax())
            restr.append(id2label[keep[j]])
            conf.append(float(sub[j] / sub.sum()))
        if (i // batch_size) % 25 == 0:
            print(f"  {min(i + batch_size, len(arrays)):,}/{len(arrays):,}",
                  flush=True)
    return free, restr, conf


def collect_training(langs, per_lang, shards, seed):
    """Clips and labels, holding out whole speakers for an honest accuracy."""
    # Train split only. The corpus validation split would double the download
    # (~15 GB across 19 languages) and buy nothing: held-out speakers are carved
    # out of whatever is collected here, so a second split is not needed.
    ds = wdata.load_labeled(tuple(langs), splits=("train",), shards=shards)
    taken = collections.Counter()
    arrays, labels, speakers = [], [], []
    for i in range(len(ds)):
        r = ds[i]
        lang = r["language"]
        if taken[lang] >= per_lang:
            continue
        if all(taken[l] >= per_lang for l in langs):
            break
        arr, sr = wdata.audio_array(r["audio"])
        if len(arr) / sr < 3.0:
            continue
        arrays.append(arr)
        labels.append(lang)
        speakers.append(str(r.get("speaker_id", "")))
        taken[lang] += 1
    print(f"collected {len(arrays):,} clips: {dict(taken)}")
    return arrays, labels, speakers


def resolve_clips(predict_dir: Path, test_csv: Path | None):
    """Ids and wav paths to label, with a readable failure instead of a crash."""
    if "{" in str(predict_dir):
        raise SystemExit(
            f"--predict-dir is the literal string {str(predict_dir)!r}. That is an "
            f"un-expanded notebook variable: run the section 6 data-prep cell "
            f"first, or pass a real path.")
    if not predict_dir.is_dir():
        raise SystemExit(f"--predict-dir {predict_dir} is not a directory")

    if test_csv and test_csv.exists():
        ids = pd.read_csv(test_csv, escapechar="\\").ID.astype(str).tolist()
        paths = [predict_dir / f"{i}.wav" for i in ids]
        absent = [p for p in paths if not p.exists()]
        if absent:
            raise SystemExit(
                f"{len(absent)} of {len(ids)} ids have no wav in {predict_dir}, "
                f"e.g. {[p.name for p in absent[:3]]}")
    else:
        paths = sorted(predict_dir.glob("*.wav"))
        ids = [p.stem for p in paths]
    if not paths:
        raise SystemExit(f"no .wav files in {predict_dir}")
    return ids, paths


def read_clips(paths):
    arrs = []
    for p in paths:
        a, sr = read_wav(p)
        if sr != wdata.SR:
            raise SystemExit(f"{p} is {sr} Hz, expected {wdata.SR}")
        arrs.append(a)
    return arrs


def report_mix(out: pd.DataFrame, args, extra_free=None) -> None:
    print(f"\n=== Phase 2 language mix ({len(out):,} clips) ===")
    print(f"{'lang':<6}{'clips':>7}{'share':>8}{'mean conf':>11}")
    for lang, n in out.language.value_counts().items():
        c = out.loc[out.language == lang, "confidence"].mean()
        print(f"{lang:<6}{n:>7}{100*n/len(out):>7.1f}%{c:>11.3f}")

    if extra_free is not None:
        outside = [l for l in extra_free if l not in set(args.langs)]
        print(f"\nunrestricted predictions outside the corpus's 19: "
              f"{len(outside):,}/{len(extra_free):,} "
              f"({100*len(outside)/max(len(extra_free),1):.1f}%)")
        if outside:
            top = collections.Counter(outside).most_common(8)
            print("  " + ", ".join(f"{l} {n}" for l, n in top))
            print("  A large share here means Phase 2 contains languages the")
            print("  corpus cannot supply labels for -- no amount of training on")
            print("  these 19 would reach those clips.")

    low = out.confidence < args.min_conf
    print(f"\nbelow --min-conf {args.min_conf}: {low.sum():,} clips "
          f"({100*low.mean():.1f}%)")
    print(f"confidence: p10 {out.confidence.quantile(.1):.3f}  "
          f"median {out.confidence.median():.3f}  "
          f"p90 {out.confidence.quantile(.9):.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


def predict_dir_mms(args, fe, model, device) -> int:
    ids, paths = resolve_clips(args.predict_dir, args.test_csv)
    print(f"\nlabelling {len(paths):,} clips from {args.predict_dir}")
    free, restr, conf = mms_predict(read_clips(paths), fe, model, device,
                                    args.langs, max(1, args.batch_size // 2))
    out = pd.DataFrame({"ID": ids, "language": restr, "confidence": conf,
                        "unrestricted": free})
    report_mix(out, args, extra_free=free)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(wdata.ALL_LANGS))
    ap.add_argument("--per-lang", type=int, default=150,
                    help="training clips per language; LID saturates fast")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict-dir", type=Path, default=None,
                    help="directory of wavs to label, e.g. data/phase_2/audio")
    ap.add_argument("--test-csv", type=Path, default=None,
                    help="restrict prediction to these ids, in this order")
    ap.add_argument("--out", type=Path, default=Path("data/phase_2/languages.csv"))
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="clips below this are reported separately -- a language "
                         "outside the 19 can only show up as low confidence")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backend", choices=("mms", "probe"), default="mms",
                    help="'mms' uses facebook/mms-lid-* directly, which is "
                         "trained for language ID and needs no fitting. 'probe' "
                         "fits a logistic regression on frozen w2v-BERT "
                         "embeddings -- measured at only 52.9% on held-out "
                         "speakers, kept for comparison")
    ap.add_argument("--mms-model", default=MMS)
    args = ap.parse_args()

    if len(args.langs) == 1 and args.langs[0] == "all":
        args.langs = list(wdata.ALL_LANGS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if hw.supports_bf16() else torch.float16
    print(f"{hw.describe()}\n")

    arrays, labels, speakers = collect_training(
        args.langs, args.per_lang, args.shards, args.seed)
    y = np.array(labels)

    if args.backend == "mms":
        # No training: MMS-LID is already a language classifier. The corpus
        # clips are used only to measure how far to trust it, on exactly the
        # audio the probe scored 52.9% on.
        print(f"\nloading {args.mms_model}")
        fe, model = load_mms(args.mms_model, device)
        print(f"scoring {len(arrays):,} labeled clips to measure accuracy")
        free, restr, _ = mms_predict(arrays, fe, model, device, args.langs,
                                     max(1, args.batch_size // 2))
        acc = float(np.mean(np.array(restr) == y))
        acc_free = float(np.mean(np.array(free) == y))
        print(f"\naccuracy on labeled corpus clips: {100*acc:.1f}% "
              f"restricted to the 19, {100*acc_free:.1f}% unrestricted")
        wrong = collections.Counter((a, b) for a, b in zip(y, restr) if a != b)
        if wrong:
            print("  most common confusions:")
            for (a, b), n in wrong.most_common(8):
                print(f"    {a} -> {b}: {n}")
        per_lang_acc = {l: float(np.mean(np.array(restr)[y == l] == l))
                        for l in sorted(set(y))}
        print("  per language: " + "  ".join(
            f"{l} {100*a:.0f}%" for l, a in sorted(per_lang_acc.items(),
                                                   key=lambda kv: kv[1])))
        if acc < 0.85:
            print("\n  WARNING: still below 85%. Treat the mix as indicative "
                  "only, and prefer the languages where per-language accuracy "
                  "above is high.")

        if not args.predict_dir:
            print("\nno --predict-dir; stopping after validation")
            return 0
        return predict_dir_mms(args, fe, model, device)

    fe, model = load_encoder(device)
    print("\nembedding training clips")
    X = embed(arrays, fe, model, device, dtype, args.batch_size)

    # Hold out whole speakers: a random split lets the same voice appear on both
    # sides, and the classifier would then be scored on speaker recognition.
    rng = np.random.default_rng(args.seed)
    spk = np.array(speakers)
    held = set()
    for lang in args.langs:
        s = sorted({v for v, l in zip(spk, y) if l == lang and v})
        if len(s) > 2:
            held.update(rng.choice(s, max(1, len(s) // 5), replace=False))
    te = np.array([s in held for s in spk])
    if te.sum() < 10 or (~te).sum() < 10:
        print("too few speakers to hold out; falling back to a random split")
        te = rng.random(len(y)) < 0.2

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[~te], y[~te])
    acc = clf.score(X[te], y[te])
    print(f"\nheld-out speaker accuracy: {100*acc:.1f}%  "
          f"({te.sum():,} clips, {len(set(y))} languages)")
    pred_te = clf.predict(X[te])
    wrong = collections.Counter(
        (a, b) for a, b in zip(y[te], pred_te) if a != b)
    if wrong:
        print("  most common confusions:")
        for (a, b), n in wrong.most_common(8):
            print(f"    {a} -> {b}: {n}")
    if acc < 0.85:
        print("\n  WARNING: accuracy this low makes the Phase 2 mix below "
              "unreliable. Raise --per-lang or --shards before trusting it.")

    if not args.predict_dir:
        print("\nno --predict-dir; stopping after validation")
        return 0

    ids, paths = resolve_clips(args.predict_dir, args.test_csv)
    print(f"\nlabelling {len(paths):,} clips from {args.predict_dir}")
    Xp = embed(read_clips(paths), fe, model, device, dtype, args.batch_size)

    proba = clf.predict_proba(Xp)
    out = pd.DataFrame({"ID": ids,
                        "language": clf.classes_[proba.argmax(1)],
                        "confidence": proba.max(1)})
    report_mix(out, args)
    print("\nUse this to set --langs / --shards: training hours should follow")
    print("this table, not our guesses about which languages are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
