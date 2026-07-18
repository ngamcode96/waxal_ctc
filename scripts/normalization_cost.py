"""What does it cost to emit normalized text against the raw cased references?

An acoustic model can't reliably hear casing or punctuation. We can either train
on raw text (and let the model guess) or train on normalized text (and accept a
fixed penalty). This measures that penalty exactly: score normalized references
against raw references. Whatever this prints is the floor a perfect normalized
model would hit, i.e. the score we can never beat by going that route.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waxal.metric import score  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def lower(t: str) -> str:
    return t.lower()


def strip_punct(t: str) -> str:
    return re.sub(r"[^\w\s']", " ", t)


def collapse_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


VARIANTS = {
    "lowercase only": lambda t: collapse_ws(lower(t)),
    "punct stripped only": lambda t: collapse_ws(strip_punct(t)),
    "lowercase + punct stripped": lambda t: collapse_ws(strip_punct(lower(t))),
    "lower+punct, keep final .": lambda t: collapse_ws(strip_punct(lower(t))) + ".",
}


def main() -> None:
    tr = pd.read_csv(RAW / "Train.csv", escapechar="\\")
    tr = tr[tr.transcription.notna()]
    refs = tr.transcription.astype(str).tolist()

    print(f"{len(refs):,} references\n")
    print("cost of emitting each normalized form (lower = cheaper):\n")
    for name, fn in VARIANTS.items():
        s = score(refs, [fn(r) for r in refs])
        print(f"{name:>28s}  WER {s.wer:.4f}  CER {s.cer:.4f}  ->  {s.combined:.4f}")

    # Casing and punctuation are only worth modelling if they're predictable.
    # How often is a token's casing determined by position alone?
    caps = sum(1 for r in refs for w in r.split() if w[:1].isupper())
    total = sum(len(r.split()) for r in refs)
    starts = sum(1 for r in refs if r[:1].isupper())
    print(f"\ncapitalized tokens: {caps:,}/{total:,} ({100*caps/total:.1f}%)")
    print(f"references starting with a capital: {starts:,}/{len(refs):,} "
          f"({100*starts/len(refs):.1f}%)")
    ends = sum(1 for r in refs if r.rstrip().endswith((".", "!", "?")))
    print(f"references ending in terminal punctuation: {ends:,} "
          f"({100*ends/len(refs):.1f}%)")


if __name__ == "__main__":
    main()
