"""Verify Hugging Face write access before spending GPU credits.

Trainer's per-checkpoint upload fails at the *first epoch boundary*, not at
startup -- so a read-only token or a namespace typo costs you an hour of A100
time before it surfaces. This exercises the same path (create repo, upload,
read back) in a few seconds, and needs no GPU.

    python scripts/check_hub.py --repo ngia/ctc-v1
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="target repo, e.g. ngia/ctc-v1")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="leave the probe file in place instead of deleting it")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("FAIL: HF_TOKEN is not set")
        print("      export HF_TOKEN=hf_...  (needs WRITE permission)")
        return 1

    api = HfApi(token=token)

    try:
        me = api.whoami()
    except Exception as e:
        print(f"FAIL: token rejected -- {type(e).__name__}: {e}")
        return 1

    name = me.get("name")
    print(f"authenticated as: {name}")

    # A fine-grained read-only token authenticates fine and only fails on write,
    # which is exactly the failure we're trying to catch early.
    auth = (me.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role")
    if role:
        print(f"token role: {role}")
        if role == "read":
            print("FAIL: this token is read-only. Create one with write access at")
            print("      https://huggingface.co/settings/tokens")
            return 1

    namespace = args.repo.split("/")[0]
    orgs = {o["name"] for o in me.get("orgs", [])}
    if namespace != name and namespace not in orgs:
        print(f"FAIL: you are '{name}' but the repo namespace is '{namespace}'")
        print(f"      accessible namespaces: {sorted({name} | orgs)}")
        return 1

    try:
        url = api.create_repo(args.repo, private=not args.public, exist_ok=True)
        print(f"repo ready: {url}")
    except Exception as e:
        print(f"FAIL: cannot create/access repo -- {type(e).__name__}: {e}")
        return 1

    info = api.repo_info(args.repo)
    print(f"private: {info.private}")
    if info.private is False and not args.public:
        print("WARNING: this repo is PUBLIC. The challenge rules forbid sharing")
        print("         work outside your team -- consider making it private.")

    probe = "_hub_write_check.txt"
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / probe
            p.write_text("write check\n")
            api.upload_file(path_or_fileobj=str(p), path_in_repo=probe,
                            repo_id=args.repo,
                            commit_message="verify write access")
        print(f"upload: ok ({probe})")
    except Exception as e:
        print(f"FAIL: upload rejected -- {type(e).__name__}: {e}")
        print("      the token authenticated but cannot write to this repo")
        return 1

    files = api.list_repo_files(args.repo)
    if probe not in files:
        print("FAIL: upload reported success but the file is not in the repo")
        return 1
    print("read back: ok")

    if not args.keep:
        api.delete_file(probe, repo_id=args.repo,
                        commit_message="remove write check")
        print("cleaned up probe file")

    print(f"\nPASS -- checkpoints will upload to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
