"""Find and remove truncated files in the Hugging Face download cache.

An interrupted download can leave a parquet file that looks present but is
short, which surfaces much later as "OSError: Corrupt snappy compressed data"
during Arrow generation -- pointing at the parser, not at the real cause.

This compares each cached file against the size the Hub reports and deletes the
mismatches so only those re-download, rather than the whole 12.6 GB.

    python scripts/verify_cache.py            # report only
    python scripts/verify_cache.py --fix      # delete corrupt entries
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, scan_cache_dir

REPO = "google/WaxalNLP"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="delete mismatched files")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    print(f"checking cached files for {args.repo}")
    remote = {}
    for s in HfApi().dataset_info(args.repo, files_metadata=True).siblings:
        if s.size is not None:
            remote[s.rfilename] = s.size
    print(f"  hub reports {len(remote):,} files with sizes")

    try:
        cache = scan_cache_dir()
    except Exception as e:
        print(f"cannot scan cache: {type(e).__name__}: {e}")
        return 1

    repo_id = args.repo
    entries = [r for r in cache.repos if r.repo_id == repo_id]
    if not entries:
        print(f"nothing cached for {repo_id} -- nothing to verify")
        return 0

    ok = bad = unknown = 0
    corrupt: list[Path] = []
    for repo in entries:
        for rev in repo.revisions:
            for f in rev.files:
                # file_name is the path within the repo snapshot
                rel = str(f.file_path).split("/snapshots/")[-1]
                rel = "/".join(rel.split("/")[1:]) if "/" in rel else rel
                expected = remote.get(rel)
                actual = f.size_on_disk
                if expected is None:
                    unknown += 1
                    continue
                if actual != expected:
                    bad += 1
                    corrupt.append(Path(f.file_path))
                    print(f"  CORRUPT {rel}: {actual:,} on disk, "
                          f"{expected:,} expected ({actual/expected:.1%})")
                else:
                    ok += 1

    print(f"\nok: {ok}   corrupt: {bad}   unverifiable: {unknown}")

    if not bad:
        print("cache is intact -- the corruption is elsewhere")
        return 0

    if not args.fix:
        print("\nrerun with --fix to delete the corrupt files (they re-download)")
        return 1

    for p in corrupt:
        # Resolve the symlink: the real bytes live in blobs/, and deleting only
        # the snapshot link would leave the bad blob in place.
        target = p.resolve() if p.is_symlink() else p
        for victim in {p, target}:
            try:
                victim.unlink()
                print(f"  deleted {victim}")
            except FileNotFoundError:
                pass
    print(f"\ndeleted {bad} corrupt file(s) -- rerun training to re-download them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
