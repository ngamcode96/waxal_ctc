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
import os
import sys
from pathlib import Path

# Must precede the torch/numpy imports: they read these at import time.
#
# Feature extraction runs N worker processes, and each would otherwise start
# torch/OpenMP with one thread per core -- ~256 threads on 16 cores. Measured
# here: 6.87 examples/s without this, against the 187 examples/s training gets
# doing identical work. Same fix as train_ctc.py, which this script never got.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
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


def extract(ds, processor, num_proc: int, max_s: float):
    """Decode and extract features in parallel, then drop over-long clips.

    Decoding one clip at a time leaves the GPU idle -- it is CPU work with one
    core doing it. Training reaches 187 clips/s with num_proc=16 doing exactly
    this, so pseudo-labelling should too.

    Long clips are filtered *after* extraction rather than skipped inside the
    map. Returning a placeholder for skipped rows gives Arrow two different
    types for the same column (halffloat vs double) and the shards refuse to
    concatenate. Extracting a few over-long clips and discarding them is cheap
    by comparison.
    """
    # Belt and braces: covers torch regardless of import order. Forked workers
    # inherit it, and the GPU pass afterwards does not need CPU threads.
    torch.set_num_threads(1)

    def fn(row):
        arr, sr = wdata.audio_array(row["audio"])
        f = processor(arr, sampling_rate=sr).input_features[0]
        return {"input_features": np.asarray(f, dtype=np.float16),
                "length": len(f), "uid": row["id"], "lang": row["language"]}

    out = ds.map(fn, remove_columns=ds.column_names, num_proc=num_proc,
                 writer_batch_size=100, desc="extracting features")
    cap = int(max_s * 50)          # ~50 frames per second
    before = len(out)
    out = out.filter(lambda n: 0 < n <= cap, input_columns="length",
                     desc=f"dropping clips over {max_s}s")
    if before != len(out):
        print(f"  dropped {before - len(out):,} clips over {max_s}s "
              f"({cap} frames)")
    return out


@torch.no_grad()
def transcribe(feats_ds, model, processor, device, max_frames, max_count):
    """Run the model over pre-extracted features, budgeting frames per batch.

    Sorted by length so each batch is near-uniform: minimal padding, and a long
    clip lands with other long clips instead of inflating a batch of short ones.
    """
    model.eval().to(device)
    # Read the length column once. Sorting with `feats_ds[i]["length"]` would
    # materialize every row -- the full feature array included -- to read a
    # single integer, which stalls for minutes on a few thousand clips.
    lengths = feats_ds["length"]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])

    ids, langs, texts, confs = [], [], [], []
    batch, frames = [], 0
    done = 0

    def flush(idx):
        nonlocal done
        if not idx:
            return
        rows = feats_ds[idx]
        padded = processor.pad(
            [{"input_features": np.asarray(f, dtype=np.float32)}
             for f in rows["input_features"]],
            padding=True, return_tensors="pt").to(device)
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**padded).logits
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on {len(idx)} clips -- skipped")
            return
        probs = logits.float().softmax(-1)
        confs.extend(probs.max(-1).values.mean(-1).cpu().numpy().tolist())
        texts.extend(processor.batch_decode(logits.argmax(-1).cpu().numpy()))
        ids.extend(rows["uid"])
        langs.extend(rows["lang"])
        done += len(idx)
        if done % 1000 < len(idx):
            print(f"  {done:,}/{len(feats_ds):,}", flush=True)

    for i in order:
        n = lengths[i]
        if batch and (frames + n > max_frames or len(batch) >= max_count):
            flush(batch)
            batch, frames = [], 0
        batch.append(i)
        frames += n
    flush(batch)
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
    ap.add_argument("--num-proc", type=int, default=8,
                    help="parallel workers for audio decoding -- this, not the "
                         "GPU, is what limits throughput")
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

    feats = extract(ds, processor, args.num_proc, args.max_s)
    print(f"  {len(feats):,} clips ready")
    ids, langs, texts, confs = transcribe(
        feats, model, processor, device,
        args.max_batch_frames, args.batch_size)
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
