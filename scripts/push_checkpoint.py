"""Upload a training checkpoint to the Hub without disturbing the running job.

With --hub-strategy end nothing reaches the Hub until training finishes, and
checkpoints written to /dev/shm live in RAM -- a pod stop loses hours of work.
This copies a completed checkpoint up while training continues.

Safe to run against a live run: it only reads, and it refuses checkpoints that
are still being written.

    python scripts/push_checkpoint.py \
        --output-dir /dev/shm/ctc-v2 --repo ngia/ctc-v2-e2

By default it uploads only what inference needs (~2.4GB). The optimizer state is
another ~2.4GB and is only useful for resuming, so it is opt-in.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Written by save_model in roughly this order; trainer_state.json lands last,
# so its presence means the checkpoint is complete.
COMPLETE_MARKER = "trainer_state.json"

INFERENCE_FILES = [
    "model.safetensors", "config.json",
    "preprocessor_config.json", "tokenizer_config.json",
    "vocab.json", "special_tokens_map.json", "added_tokens.json",
]
RESUME_FILES = ["optimizer.pt", "scheduler.pt", "rng_state.pth",
                "trainer_state.json", "training_args.bin"]


def latest_complete(output_dir: Path) -> Path | None:
    """Newest checkpoint that has finished being written."""
    cks = sorted(output_dir.glob("checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[1]))
    for ck in reversed(cks):
        if (ck / COMPLETE_MARKER).exists():
            return ck
        print(f"skipping {ck.name}: still being written")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="training --output-dir (holds checkpoint-* and the processor)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="a specific checkpoint; defaults to the newest complete one")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--with-optimizer", action="store_true",
                    help="also upload optimizer/scheduler state (~2.4GB) so the "
                         "run can be resumed elsewhere")
    args = ap.parse_args()

    ck = args.checkpoint or latest_complete(args.output_dir)
    if ck is None:
        print(f"no complete checkpoint in {args.output_dir}")
        return 1
    print(f"checkpoint: {ck}")

    wanted = INFERENCE_FILES + (RESUME_FILES if args.with_optimizer else [])
    # The processor lives in the parent (train_ctc saves it before training);
    # the weights live in the checkpoint. Prefer the checkpoint's copy.
    uploads: dict[str, Path] = {}
    for name in wanted:
        for src in (ck / name, args.output_dir / name):
            if src.exists():
                uploads.setdefault(name, src)

    if "model.safetensors" not in uploads:
        print(f"no model.safetensors in {ck} -- is this a checkpoint directory?")
        return 1

    total = sum(p.stat().st_size for p in uploads.values())
    print(f"uploading {len(uploads)} files ({total/1e9:.2f}GB) -> {args.repo}")
    for name, src in sorted(uploads.items()):
        print(f"  {name:28s} {src.stat().st_size/1e6:8.1f}MB  ({src.parent.name})")

    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    for name, src in sorted(uploads.items()):
        api.upload_file(path_or_fileobj=str(src), path_in_repo=name,
                        repo_id=args.repo,
                        commit_message=f"{ck.name}: {name}")
        print(f"  pushed {name}")

    print(f"\ndone -> https://huggingface.co/{args.repo}")
    print(f"infer with:  --model {args.repo}")
    if not args.with_optimizer:
        print("(inference only; pass --with-optimizer to make it resumable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
