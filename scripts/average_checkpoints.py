"""Average the weights of several checkpoints into one model.

v2's validation oscillates between epochs (0.1747, 0.1756, 0.1591, 0.1634),
which means the optimizer is circling a minimum rather than sitting in it.
Averaging weights across those points usually lands closer to the centre than
any single epoch does -- the per-epoch noise cancels, the shared signal does not.

This is weight averaging, not ensembling: the result is one model of the same
size and the same inference cost. It only works across checkpoints from a single
run, where the weights are all in the same basin.

    python scripts/average_checkpoints.py \
        --checkpoints /dev/shm/ctc-v2/checkpoint-{3916,4895,5874} \
        --output-dir /dev/shm/ctc-v2-avg
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Copied alongside the weights so the result loads as a complete model.
SIDECAR = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
           "vocab.json", "special_tokens_map.json", "added_tokens.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weights", type=float, nargs="*", default=None,
                    help="per-checkpoint weights; defaults to uniform")
    args = ap.parse_args()

    paths = [c / "model.safetensors" for c in args.checkpoints]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"missing weights: {missing}")
        return 1

    w = args.weights or [1.0] * len(paths)
    if len(w) != len(paths):
        print(f"got {len(w)} weights for {len(paths)} checkpoints")
        return 1
    total = sum(w)
    w = [x / total for x in w]
    print("averaging:")
    for c, x in zip(args.checkpoints, w):
        print(f"  {x:.3f}  {c}")

    acc: dict[str, torch.Tensor] = {}
    for path, weight in zip(paths, w):
        sd = load_file(str(path))
        for k, v in sd.items():
            # Accumulate in float32 even when the weights are bf16: summing many
            # low-precision tensors otherwise loses the small differences that
            # averaging is meant to capture.
            contrib = v.to(torch.float32) * weight
            acc[k] = contrib if k not in acc else acc[k] + contrib
        del sd

    # Restore each tensor's original dtype from the first checkpoint.
    ref = load_file(str(paths[0]))
    out = {k: v.to(ref[k].dtype) for k, v in acc.items()}
    if set(out) != set(ref):
        print(f"key mismatch: {set(out) ^ set(ref)}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.output_dir / "model.safetensors"))

    src = args.checkpoints[0]
    for name in SIDECAR:
        for cand in (src / name, src.parent / name):
            if cand.exists():
                shutil.copy(cand, args.output_dir / name)
                break

    have = sorted(p.name for p in args.output_dir.iterdir())
    print(f"\nwrote {args.output_dir}: {have}")
    print(f"\nevaluate it:\n"
          f"  python scripts/eval_checkpoint.py --model {args.output_dir} "
          f"--cache-dir /dev/shm/cache --batch-size 32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
