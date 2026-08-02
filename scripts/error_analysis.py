"""Break the word errors down by cause, to decide what will actually move WER.

Measured on the corrected Phase 2 set (2026-08-03): our CER is 0.1145 against
first place's 0.1137 -- within 0.0008 -- while our WER is 0.4020 against 0.3794.
97% of the gap is words. Since a word-boundary error costs one character but two
words, that profile points at boundaries rather than mishearing, but "points at"
is not "is". This measures it.

Every substitution, deletion and insertion in the alignment is assigned a cause:

  merge        one hypothesis word equals several reference words joined
  split        several hypothesis words join to equal one reference word
  case         right word, wrong capitalisation
  punct        right word once punctuation is stripped
  near-miss    within a couple of character edits -- a spelling slip
  wrong        genuinely a different word

The split matters because the fixes differ. Merge, split and near-miss are what
an n-gram LM beam search repairs: it scores whole word sequences and prefers
real words over pseudo-words. `wrong` and most deletions are acoustic, and only
a better model or more data touches those.

    python scripts/eval_checkpoint.py --model ngia/ctc-v2-avg \
        --cache-dir /workspace/cache-v2 --dump /workspace/dump.jsonl
    python scripts/error_analysis.py --dump /workspace/dump.jsonl
"""

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jiwer                                                # noqa: E402


def strip_punct(w: str) -> str:
    return "".join(c for c in w if not unicodedata.category(c).startswith("P"))


def edits(a: str, b: str) -> int:
    """Levenshtein distance, iterative two-row."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_regions(alignment):
    """Maximal runs of consecutive non-equal chunks, as (ref span, hyp span).

    jiwer splits a single boundary mistake across adjacent chunks -- merging
    'kabedo neni' into 'kabedoneni' shows up as substitute(kabedo -> kabedoneni)
    followed by delete(neni), and neither chunk alone looks like a merge. Only
    the joined region does.
    """
    regions, cur = [], None
    for ch in alignment:
        if ch.type == "equal":
            if cur:
                regions.append(cur)
                cur = None
            continue
        if cur is None:
            cur = [ch.ref_start_idx, ch.ref_end_idx,
                   ch.hyp_start_idx, ch.hyp_end_idx]
        else:
            cur[1], cur[3] = ch.ref_end_idx, ch.hyp_end_idx
    if cur:
        regions.append(cur)
    return regions


def classify(ref_words, hyp_words):
    """Assign each alignment error a cause. Returns Counter and examples."""
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    counts = collections.Counter()
    examples = collections.defaultdict(list)

    for rs, re_, hs, he in error_regions(out.alignments[0]):
        r, h = ref_words[rs:re_], hyp_words[hs:he]
        # An error's WER cost is the number of edits the aligner needed, which
        # for a region is the longer side.
        cost = max(len(r), len(h))

        if not h:
            counts["deletion"] += cost
            examples["deletion"].append((" ".join(r), ""))
            continue
        if not r:
            counts["insertion"] += cost
            examples["insertion"].append(("", " ".join(h)))
            continue

        # Boundary check on the joined region: same characters, different
        # spacing. Punctuation is ignored so 'neni.' vs 'neni' does not hide it.
        rj = strip_punct("".join(r)).lower()
        hj = strip_punct("".join(h)).lower()
        if rj == hj and len(r) != len(h):
            kind = "merge" if len(h) < len(r) else "split"
            counts[kind] += cost
            examples[kind].append((" ".join(r), " ".join(h)))
            continue

        for a, b in zip(r, h):
            if a == b:
                continue
            if a.lower() == b.lower():
                counts["case"] += 1
                examples["case"].append((a, b))
            elif strip_punct(a) == strip_punct(b):
                counts["punct"] += 1
                examples["punct"].append((a, b))
            elif edits(a.lower(), b.lower()) <= max(1, len(a) // 4):
                counts["near-miss"] += 1
                examples["near-miss"].append((a, b))
            else:
                counts["wrong"] += 1
                examples["wrong"].append((a, b))
        # Whatever the shorter side could not pair with is an extra edit.
        extra = abs(len(r) - len(h))
        if extra:
            counts["wrong"] += extra
            examples["wrong"].append((" ".join(r), " ".join(h)))
    return counts, examples, len(ref_words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True,
                    help="JSONL from eval_checkpoint.py --dump")
    ap.add_argument("--lang", default=None, help="restrict to one language")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.dump.read_text().splitlines() if l.strip()]
    if args.lang:
        rows = [r for r in rows if r.get("lang") == args.lang]
    if not rows:
        raise SystemExit(f"no rows in {args.dump}"
                         + (f" for lang={args.lang}" if args.lang else ""))

    total = collections.Counter()
    ex = collections.defaultdict(list)
    ref_words = 0
    for r in rows:
        c, e, n = classify(r["ref"].split(), r["hyp"].split())
        total.update(c)
        ref_words += n
        for k, v in e.items():
            if len(ex[k]) < args.examples * 3:
                ex[k].extend(v)

    errs = sum(total.values())
    print(f"{len(rows):,} clips, {ref_words:,} reference words"
          + (f", language={args.lang}" if args.lang else ""))
    print(f"total word errors: {errs:,}   WER {errs/ref_words:.4f}\n")

    LM_FIXABLE = {"merge", "split", "near-miss", "case", "punct"}
    print(f"{'cause':<12}{'errors':>9}{'% of errors':>13}{'WER cost':>11}   fixed by")
    fixable = 0
    for k, n in total.most_common():
        tag = "LM beam search" if k in LM_FIXABLE else "acoustic model / more data"
        if k in LM_FIXABLE:
            fixable += n
        print(f"{k:<12}{n:>9,}{100*n/errs:>12.1f}%{n/ref_words:>11.4f}   {tag}")

    print(f"\nLM-addressable: {fixable:,}/{errs:,} ({100*fixable/errs:.1f}% of errors, "
          f"{fixable/ref_words:.4f} WER)")
    print("An LM will not fix all of those -- it is an upper bound on what beam")
    print("search over word sequences could reach.\n")

    for k in ("merge", "split", "near-miss", "case", "punct", "wrong"):
        if not ex.get(k):
            continue
        print(f"--- {k} ---")
        seen = set()
        shown = 0
        for a, b in ex[k]:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            print(f"  ref {a!r:<38} hyp {b!r}")
            shown += 1
            if shown >= args.examples:
                break
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
