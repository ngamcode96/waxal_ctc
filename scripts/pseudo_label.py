"""Transcribe unlabeled audio to create pseudo-labels for semi-supervised training.

The labeled set is ~180 hours across three languages; the unlabeled pool is
~78GB. Training on our own high-confidence transcriptions of that pool is the
one remaining technique with the right order of magnitude -- augmentation and
checkpoint averaging give 5-10% relative, self-training can give 20-30%.

Output is a small CSV (id, language, transcription, confidence), not features.
Features are re-extracted later on whatever GPU trains the model, which is far
cheaper than moving ~10GB of arrays between machines.

Nothing here touches the test split: the unlabeled pool has no transcriptions at
all, so there is no label to leak.

    python scripts/pseudo_label.py --model ngia/ctc-v2-avg \
        --shards 0 1 --out /kaggle/working/pseudo.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from infer import load_processor  # noqa: E402
from waxal import data as wdata  # noqa: E402

def is_degenerate(text: str) -> bool:
    """Detect CTC looping ("Mki a a a ai....") on clips with no usable speech.

    A regex for consecutive single letters misses the interleaved cases, so this
    tests the *ratio* of one-character tokens instead. Standalone punctuation is
    ignored -- counting it flagged real transcripts that merely had a spaced-out
    comma. Validated against all 38,199 training transcripts: 4 false positives
    (0.01%), and it catches every degenerate output we have seen.
    """
    toks = [t for t in (t.strip(".,!?;:'\"") for t in text.split()) if t]
    if not toks:
        return True
    single = [len(t) <= 1 for t in toks]
    if sum(single) / len(toks) >= 0.4:
        return True
    run = 0
    for s in single:
        run = run + 1 if s else 0
        if run >= 3:
            return True
    return False


def frame_batches(ds, max_frames: int, max_count: int, max_s: float):
    """Group clips so each batch has a bounded total length, not a fixed count.

    The unlabeled pool is unfiltered -- filter_usable only ever runs on labeled
    data -- so it contains clips far longer than anything in training. With a
    fixed batch size one of those blows up attention memory no matter how small
    the batch is. Budgeting frames instead means a long clip simply becomes its
    own batch, and clips over max_s are skipped entirely since training would
    drop them anyway.
    """
    buf, frames, skipped = [], 0, 0
    for row in ds:
        arr, sr = wdata.audio_array(row["audio"])
        dur = len(arr) / sr
        if dur > max_s:
            skipped += 1
            continue
        n = len(arr) // 320                     # ~50 frames/sec after stride
        if buf and (frames + n > max_frames or len(buf) >= max_count):
            yield buf
            buf, frames = [], 0
        buf.append((row["id"], row["language"], arr))
        frames += n
    if buf:
        yield buf
    if skipped:
        print(f"  skipped {skipped:,} clips over {max_s}s")


@torch.no_grad()
def transcribe(ds, model, processor, device, max_frames, max_count, max_s):
    """Returns (ids, langs, texts, confidences)."""
    model.eval().to(device)
    ids, langs, texts, confs = [], [], [], []
    done = 0
    for batch in frame_batches(ds, max_frames, max_count, max_s):
        arrays = [a for _, _, a in batch]
        feats = processor(arrays, sampling_rate=wdata.SR,
                          return_tensors="pt", padding=True).to(device)
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**feats).logits
        except torch.OutOfMemoryError:
            # A single clip can still exceed memory; drop it rather than lose
            # the whole shard, and report it so the budget can be tuned.
            torch.cuda.empty_cache()
            print(f"  OOM on a batch of {len(batch)} "
                  f"({sum(len(a)//320 for a in arrays):,} frames) -- skipped")
            continue

        probs = logits.float().softmax(-1)
        conf = probs.max(-1).values.mean(-1).cpu().numpy()
        hyps = processor.batch_decode(logits.argmax(-1).cpu().numpy())
        del logits, probs, feats

        ids.extend(i for i, _, _ in batch)
        langs.extend(l for _, l, _ in batch)
        texts.extend(hyps)
        confs.extend(conf.tolist())
        done += len(batch)
        if done % 500 < len(batch):
            print(f"  {done:,}/{len(ds):,}", flush=True)
    return ids, langs, texts, confs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--shards", type=int, nargs="+", default=[0],
                    help="which unlabeled parquet shards to pull per language "
                         "(~0.5GB each)")
    ap.add_argument("--langs", nargs="+", default=list(wdata.LANGS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="upper bound on clips per batch")
    ap.add_argument("--max-batch-frames", type=int, default=8000,
                    help="total frames per batch (~50/sec). This, not "
                         "--batch-size, is what prevents OOM: 8000 suits a T4, "
                         "24000 an A100")
    ap.add_argument("--max-s", type=float, default=30.0,
                    help="skip clips longer than this; the unlabeled pool is "
                         "unfiltered and training would drop them anyway")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop rows below this confidence (0 keeps everything; "
                         "filter later once you can see the distribution)")
    ap.add_argument("--min-words", type=int, default=4)
    args = ap.parse_args()

    processor = load_processor(args.model, args.vocab)
    import transformers
    model = transformers.Wav2Vec2BertForCTC.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = wdata.load_unlabeled(tuple(args.langs), tuple(args.shards))
    print(f"pseudo-labeling {len(ds):,} unlabeled clips on {device}")

    ids, langs, texts, confs = transcribe(
        ds, model, processor, device,
        args.max_batch_frames, args.batch_size, args.max_s)
    df = pd.DataFrame({"id": ids, "language": langs,
                       "transcription": texts, "confidence": confs})

    df["words"] = df.transcription.str.split().str.len()
    df["degenerate"] = df.transcription.map(is_degenerate)

    print(f"\n{len(df):,} transcribed")
    print(f"  confidence: p10 {df.confidence.quantile(.10):.3f}  "
          f"median {df.confidence.median():.3f}  "
          f"p90 {df.confidence.quantile(.90):.3f}")
    print(f"  degenerate: {df.degenerate.sum():,} ({100*df.degenerate.mean():.1f}%)")
    print(f"  under {args.min_words} words: {(df.words < args.min_words).sum():,}")

    keep = (~df.degenerate) & (df.words >= args.min_words) & \
           (df.confidence >= args.min_conf)
    print(f"\nkept {keep.sum():,}/{len(df):,} ({100*keep.mean():.1f}%)")
    print("\nby language and confidence decile, to choose a threshold later:")
    for lang, g in df[keep].groupby("language"):
        print(f"  {lang}: {len(g):,} rows, "
              f"conf p25 {g.confidence.quantile(.25):.3f} "
              f"median {g.confidence.median():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[keep].drop(columns=["degenerate"]).to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({keep.sum():,} rows)")
    print("\nsample:")
    for _, r in df[keep].head(3).iterrows():
        print(f"  [{r.language} conf={r.confidence:.3f}] {r.transcription[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
