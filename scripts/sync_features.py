"""Push/pull the extracted feature cache to a private HF dataset repo.

Extraction costs ~50 minutes of CPU. Uploading the result (~16 GB) takes a few
minutes, and any later pod pulls it instead of re-extracting. That turns a
per-pod cost into a one-off.

Uploads the save_to_disk directory verbatim rather than via push_to_hub, so the
round trip is byte-identical Arrow -- no parquet conversion, and the manifest
that validates the cache key travels with it.

    python scripts/sync_features.py push --cache-dir /workspace/cache
    python scripts/sync_features.py pull --cache-dir /workspace/cache

The repo is PRIVATE by default: these features are derived from the challenge
data, and the rules forbid sharing work outside your team.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_REPO = "ngia/waxal-features"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def push(args) -> int:
    cache = args.cache_dir
    if not (cache / "train").exists():
        print(f"nothing to push: {cache/'train'} does not exist")
        print("run training once with --cache-dir to build the feature cache")
        return 1

    size = dir_size(cache)
    print(f"uploading {cache} ({human(size)}) -> {args.repo}")
    if size > 300 * 1024**3:
        print("refusing: over 300GB, that is not a feature cache")
        return 1

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(cache),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="feature cache",
        # Arrow files are already compressed audio features; no point gzipping.
        ignore_patterns=["*.lock", "**/tmp*"],
    )
    print(f"done -> https://huggingface.co/datasets/{args.repo}")
    print("pull it on a new pod with:")
    print(f"  python scripts/sync_features.py pull --cache-dir {cache} "
          f"--repo {args.repo}")
    return 0


def pull(args) -> int:
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo} -> {cache}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(cache),
        max_workers=args.workers,
    )
    for split in ("train", "valid"):
        d = cache / split
        print(f"  {split}: {'ok' if d.exists() else 'MISSING'} "
              f"({human(dir_size(d)) if d.exists() else '-'})")
    print("\ntraining will now reuse these instead of extracting -- but only if "
          "the cache key matches (model, langs, valid_frac, seed, min_s/max_s, "
          "vocab). A mismatch rebuilds and prints both keys.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("push", "pull"))
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--public", action="store_true",
                    help="publish publicly; off by default (challenge rules)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    return push(args) if args.action == "push" else pull(args)


if __name__ == "__main__":
    sys.exit(main())
