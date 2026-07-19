"""Score a checkpoint on the cached validation split, both metric ways.

The leaderboard and our training log disagree sharply (0.78 vs 0.17). Zindi does
not publish whether it averages per-utterance rates or computes corpus-level
ones, and those diverge a lot when clip lengths vary. This prints both, plus a
per-language and per-length breakdown, so we can tell which number the
leaderboard corresponds to -- without going near the public test labels.

    python scripts/eval_checkpoint.py \
        --model /workspace/ctc-v1/checkpoint-1743 \
        --cache-dir /dev/shm/cache
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import load_processor  # noqa: E402

from waxal.metric import score, score_by_language  # noqa: E402


def load_valid(cache_dir: Path):
    shards = sorted(cache_dir.glob("valid_*_of_*.arrow")) or \
        [p for p in [cache_dir / "valid.arrow"] if p.exists()]
    if not shards:
        raise SystemExit(f"no valid shards in {cache_dir}")
    return datasets.concatenate_datasets(
        [datasets.Dataset.from_file(str(f)) for f in shards])


@torch.no_grad()
def run(ds, model, processor, device, batch_size):
    model.eval().to(device)
    hyps = []
    for start in range(0, len(ds), batch_size):
        rows = ds[start:start + batch_size]
        # These are already-extracted features; pad them exactly as the training
        # collator does. Passing them back through the feature extractor would
        # treat mel frames as a waveform and produce nonsense.
        feats = processor.pad(
            [{"input_features": np.asarray(f, dtype=np.float32)}
             for f in rows["input_features"]],
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(**feats).logits
        hyps.extend(processor.batch_decode(logits.argmax(-1).cpu().numpy()))
        if start % (batch_size * 20) == 0:
            print(f"  {min(start + batch_size, len(ds)):,}/{len(ds):,}", flush=True)
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import transformers

    processor = load_processor(args.model, args.vocab)
    model = transformers.Wav2Vec2BertForCTC.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_valid(args.cache_dir)
    print(f"validation: {len(ds):,} clips")

    refs = [processor.tokenizer.decode(ids, group_tokens=False)
            for ids in ds["labels"]]
    hyps = run(ds, model, processor, device, args.batch_size)

    s = score(refs, hyps)
    print(f"\n{s}\n")
    print("The leaderboard number should match one of those two lines.\n")

    for lang, v in score_by_language(refs, hyps, ds["language"]).items():
        print(f"  {lang}: corpus {v.combined:.4f}   mean {v.combined_mean:.4f}")

    # Short clips are where per-utterance averaging hurts most: one wrong word
    # in a five-word reference is a WER of 0.2 on its own.
    lengths = ds["length"]
    buckets = [(0, 400), (400, 800), (800, 1200), (1200, 10_000)]
    print("\nby clip length (frames, ~50/sec):")
    for lo, hi in buckets:
        idx = [i for i, n in enumerate(lengths) if lo <= n < hi]
        if not idx:
            continue
        b = score([refs[i] for i in idx], [hyps[i] for i in idx])
        print(f"  {lo:>5}-{hi:<5} n={len(idx):>5}  "
              f"corpus {b.combined:.4f}   mean {b.combined_mean:.4f}")

    empty = sum(1 for h in hyps if not h.strip())
    print(f"\nempty hypotheses: {empty}/{len(hyps)}")
    print("\nfirst 3:")
    for r, h in list(zip(refs, hyps))[:3]:
        print(f"  ref: {r[:100]}\n  hyp: {h[:100]}\n")


if __name__ == "__main__":
    main()
