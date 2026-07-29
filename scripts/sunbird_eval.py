"""Score Sunbird's 51-language African ASR model on our validation set.

Why an outside model matters here. Phase 2 turned out to be East African, and
roughly half our remaining error sits in clips MMS places outside the corpus's
19 languages -- Kinyarwanda, Luo, Lumasaba, Nyoro. We have no labeled audio for
those, so no amount of training on WaxalNLP reaches them.

Sunbird/asr-whisper-51-african-languages covers all of them, plus five of our
six, under Apache-2.0. Three things it could be worth, in increasing order of
effort:

  1. Better than ours outright -> submit it, or ensemble.
  2. Comparable -> use it as an independent teacher to pseudo-label Phase 2.
     Its errors are uncorrelated with ours, which is what makes distillation
     from it stronger than self-training on our own output.
  3. Worse on our languages but competent on kin/luo/myx/nyo -> label only the
     clips our LID flags as out-of-corpus.

This measures which. It reuses selftest_phase2.build_labeled_wavs, so the clips
are drawn exactly as our own evaluations draw them and the numbers compare
directly.

Note the corpus calls Lusoga `sog`, which is really the ISO code for Sogdian --
Lusoga is `xog`. That mismatch is why MMS-LID reported "sog not covered" and
scored it 0%. The mapping below corrects it.

    python scripts/sunbird_eval.py --langs ach lug nyn sog lin sna --n 180
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import torch                                                # noqa: E402

from infer_phase2 import read_wav                           # noqa: E402
from selftest_phase2 import build_labeled_wavs              # noqa: E402
from waxal import data as wdata                             # noqa: E402
from waxal import hw                                        # noqa: E402
from waxal.metric import score, score_by_language           # noqa: E402

MODEL = "Sunbird/asr-whisper-51-african-languages"

# From the model card: Sunbird overwrote unused Whisper language tokens, so
# these ids are not the ones base Whisper uses for the same codes.
LANGUAGE_TOKENS = {
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325,
    "som": 50326, "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353,
    "hau": 50354, "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352,
    "ber": 50351, "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346,
    "ful": 50345, "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341,
    "kin": 50340, "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336,
    "lgg": 50335, "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330,
    "myx": 50329, "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321,
    "pcm": 50320, "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315,
    "tsn": 50314, "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310,
    "zul": 50309,
}

# The corpus's language codes -> Sunbird's. Only Lusoga differs.
CORPUS_TO_SUNBIRD = {"sog": "xog"}

# Measured on this validation set with ctc-p2-avg, for direct comparison.
OURS = {"ach": 0.2631, "nyn": 0.2264, "sog": 0.2909,
        "lug": 0.1060, "lin": 0.2440, "sna": 0.1183, "all": 0.1846}


def load_model(model_id, device, dtype):
    import transformers
    processor = transformers.WhisperProcessor.from_pretrained(model_id)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype).eval().to(device)
    return processor, model


@torch.no_grad()
def transcribe(paths, langs, processor, model, device, dtype,
               batch_size, tell_language, max_new_tokens):
    """Batched, grouped by language so one forced-token prefix serves a batch.

    Whisper's decoder prefix is per-sequence, so a mixed-language batch would
    need per-row forced ids. Grouping avoids that without giving up batching.
    """
    tok = processor.tokenizer
    transcribe_tok = tok.convert_tokens_to_ids("<|transcribe|>")
    nots_tok = tok.convert_tokens_to_ids("<|notimestamps|>")

    order, out = [], []
    by_lang = {}
    for i, (p, l) in enumerate(zip(paths, langs)):
        by_lang.setdefault(l, []).append(i)

    t0, done = time.time(), 0
    for lang, idxs in by_lang.items():
        code = CORPUS_TO_SUNBIRD.get(lang, lang)
        forced = None
        if tell_language:
            if code not in LANGUAGE_TOKENS:
                print(f"  {lang}: no Sunbird token for '{code}' -- auto-detecting")
            else:
                forced = [(1, LANGUAGE_TOKENS[code]), (2, transcribe_tok),
                          (3, nots_tok)]
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            audio = [read_wav(paths[i])[0] for i in chunk]
            feats = processor(audio, sampling_rate=wdata.SR, do_normalize=True,
                              return_tensors="pt").input_features.to(device, dtype)
            ids = model.generate(feats, num_beams=1, do_sample=False,
                                 max_new_tokens=max_new_tokens,
                                 **({"forced_decoder_ids": forced} if forced else {}))
            texts = processor.batch_decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            order.extend(chunk)
            out.extend(" ".join(t.split()) for t in texts)
            done += len(chunk)
            if done % 40 < batch_size:
                r = done / (time.time() - t0)
                print(f"  {done}/{len(paths)}  ({r:.2f} clips/s, "
                      f"{(len(paths)-done)/max(r,1e-9)/60:.0f} min left)", flush=True)
    # Restore the caller's order.
    hyps = [None] * len(paths)
    for i, t in zip(order, out):
        hyps[i] = t
    return hyps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--langs", nargs="+",
                    default=["ach", "lin", "lug", "nyn", "sna", "sog"])
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--work", type=Path, default=Path("/workspace/sunbird"))
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--valid-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-s", type=float, default=18.0)
    ap.add_argument("--max-s", type=float, default=30.0,
                    help="Whisper takes 30s of audio; longer is truncated")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--tell-language", action="store_true",
                    help="force the language token. The card says accuracy is "
                         "better with it, but Phase 2 ships no language tag, so "
                         "the default (auto-detect) is the deployable number and "
                         "this is the upper bound")
    args = ap.parse_args()

    if args.max_s > 30:
        ap.error("Whisper accepts at most 30s of audio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if hw.supports_bf16() else torch.float16
    print(f"{hw.describe()}\n")

    ref, _, audio_dir = build_labeled_wavs(
        args.langs, args.n, args.work, args.shards, args.valid_frac, args.seed,
        args.min_s, args.max_s)

    print(f"\nloading {args.model}")
    processor, model = load_model(args.model, device, dtype)
    mode = "language told" if args.tell_language else "auto-detect (as Phase 2)"
    print(f"{len(ref):,} clips, {mode}\n")

    paths = [audio_dir / f"{i}.wav" for i in ref.ID]
    hyps = transcribe(paths, ref.language.tolist(), processor, model, device,
                      dtype, args.batch_size, args.tell_language,
                      args.max_new_tokens)
    refs = ref.transcription.astype(str).tolist()

    out = args.work / "sunbird_pred.csv"
    pd.DataFrame({"ID": ref.ID, "language": ref.language,
                  "reference": refs, "hypothesis": hyps}).to_csv(out, index=False)

    s = score(refs, hyps)
    per = score_by_language(refs, hyps, ref.language.tolist())
    print(f"\n=== {args.model} ({len(refs):,} clips, {mode}) ===")
    print(s)

    print(f"\n{'lang':<6}{'n':>5}{'sunbird':>10}{'ctc-p2-avg':>13}{'':>4}verdict")
    wins = 0
    for lang in sorted(per):
        n = int((ref.language == lang).sum())
        a, b = per[lang].combined, OURS.get(lang)
        if b is None:
            v, cmp = "no baseline", "  --"
        elif a < b - 0.02:
            v, cmp, wins = "SUNBIRD BETTER", f"{b:.4f}", wins + 1
        elif a > b + 0.02:
            v, cmp = "ours better", f"{b:.4f}"
        else:
            v, cmp = "comparable", f"{b:.4f}"
        print(f"{lang:<6}{n:>5}{a:>10.4f}{cmp:>13}    {v}")
    print(f"\noverall {s.combined:.4f} vs ours {OURS['all']:.4f}")

    rw = np.array([len(r.split()) for r in refs])
    hw_ = np.array([len(h.split()) for h in hyps])
    print(f"words/clip: reference {rw.mean():.1f}  hypothesis {hw_.mean():.1f} "
          f"({hw_.mean()/max(rw.mean(),1e-9):.2f}x)")
    if hw_.mean() / max(rw.mean(), 1e-9) > 1.4 or s.combined > 1.0:
        print("\n  hypotheses run long -- Whisper is probably emitting commentary")
        print("  or looping. Inspect the CSV before believing the score.")

    print("\n--- read this as ---")
    if s.combined < OURS["all"] - 0.02:
        print("  Sunbird beats our model on our own validation set. Score Phase 2")
        print("  with it directly before anything more elaborate.")
    elif wins:
        print(f"  Sunbird wins on {wins} language(s). Even losing overall, it is")
        print("  an independent teacher for those -- and it covers kin/luo/myx/nyo,")
        print("  which we cannot train at all. Pseudo-labelling Phase 2 with it is")
        print("  the move that reaches the half of the error we cannot measure.")
    else:
        print("  Ours is better everywhere measurable. The remaining case for")
        print("  Sunbird is kin/luo/myx/nyo, which this set cannot test -- run it")
        print("  over the Phase 2 clips our LID flags as out-of-corpus and read")
        print("  the outputs by hand.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
