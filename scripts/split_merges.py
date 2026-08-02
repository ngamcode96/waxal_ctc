"""Split words the CTC head glued together, using the training vocabulary.

Measured on the corrected Phase 2 submission (2026-08-03): 411 of 22,909 output
tokens are out-of-vocabulary words that decompose cleanly into two *frequent*
training words -- 'yaoyo' -> 'ya oyo', 'pembeya' -> 'pembe ya'. The words being
glued are the shortest and commonest ones in the corpus: na (rank 2), ya (rank
1), ba (rank 4), ne, pe. That is CTC under-predicting the word delimiter between
a function word and its host.

The direction is one-sided -- 411 merge-implicated tokens against 2 splits -- so
this only ever splits, never joins.

Why it is worth doing at all when an n-gram LM would also fix it: this needs no
kenlm build, no beam search, no GPU, and runs in seconds. A merge costs two words
of WER but only one character of CER, and our CER is already within 0.0008 of
first place while our WER is 0.0226 behind, so word boundaries are the whole gap.

Validate before submitting. With --dump (from eval_checkpoint.py --dump) it
scores WER and CER before and after on data that has references. Do not ship a
rule that has only been eyeballed.

    python scripts/split_merges.py --dump /workspace/dump.jsonl        # measure
    python scripts/split_merges.py --sub submission.csv --out fixed.csv
"""

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                         # noqa: E402


def word_chars(s: str) -> str:
    return "".join(c for c in s if not unicodedata.category(c).startswith("P"))


def load_vocab(train_csv: Path, cache_dir: Path | None) -> collections.Counter:
    """Word frequencies from the training transcripts.

    Prefers the cached training labels, which exist on a rented box where
    Train.csv (gitignored) does not.
    """
    if cache_dir:
        import datasets                                     # noqa: F401
        import transformers
        from waxal import data as wdata
        shards = sorted(cache_dir.glob("train_*_of_*.arrow")) or \
            [p for p in [cache_dir / "train.arrow"] if p.exists()]
        if shards:
            import datasets as ds
            tok = transformers.Wav2Vec2CTCTokenizer(
                str(cache_dir.parent / "vocab.json"), unk_token="[UNK]",
                pad_token="[PAD]", word_delimiter_token="|")
            d = ds.concatenate_datasets([wdata.read_arrow_shard(f) for f in shards])
            texts = [tok.decode(i, group_tokens=False) for i in d["labels"]]
            return collections.Counter(w.lower() for t in texts for w in t.split())
    df = pd.read_csv(train_csv, escapechar="\\")
    return collections.Counter(
        w for t in df.transcription.dropna()
        for w in re.findall(r"[^\W\d_]+", str(t).lower()))


def build_splitter(vocab: collections.Counter, min_count: int, min_part: int,
                   protect_count: int = 1):
    """word -> 'a b' for glued words, or None.

    Two different thresholds, because these are two different questions:

    `protect_count` -- is this already a word? A word attested in training is
    left alone. Default 1, i.e. any occurrence protects it. Splitting a genuine
    rare word ADDS two WER errors, exactly as much as fixing a merge removes,
    so the rule has to be conservative about what it touches.

    `min_count` -- are both halves real words? Requiring only membership lets a
    rare word that happens to start with a common prefix be torn apart, so the
    halves must be frequent. Where several splits are possible the most frequent
    pair wins, which prefers the function-word boundary that caused the merge.
    """
    V = {w for w, n in vocab.items() if n >= min_count}
    protected = {w for w, n in vocab.items() if n >= protect_count}
    cache: dict[str, str | None] = {}

    def split(word: str) -> str | None:
        key = word.lower()
        if key in cache:
            return cache[key]
        out = None
        if key not in protected and len(key) >= 2 * min_part:
            best = 0
            for i in range(min_part, len(key) - min_part + 1):
                a, b = key[:i], key[i:]
                if a in V and b in V:
                    s = min(vocab[a], vocab[b])
                    if s > best:
                        best, out = s, (i,)
            if out:
                out = out[0]
        cache[key] = out
        return out

    def apply(token: str) -> str:
        """Split one token, preserving its casing and trailing punctuation."""
        core = word_chars(token)
        if not core:
            return token
        i = split(core)
        if i is None:
            return token
        # Locate the core inside the token so punctuation stays where it was.
        at = token.lower().find(core.lower())
        head, tail = token[:at], token[at + len(core):]
        return f"{head}{core[:i]} {core[i:]}{tail}"

    return apply, V


def fix(text: str, apply) -> tuple[str, int]:
    out, n = [], 0
    for tok in str(text).split():
        new = apply(tok)
        n += new != tok
        out.append(new)
    return " ".join(out), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=Path, default=None, help="submission CSV to fix")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None,
                    help="JSONL from eval_checkpoint.py --dump; measures the "
                         "rule against references instead of applying it")
    ap.add_argument("--train-csv", type=Path, default=Path("data/raw/Train.csv"))
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="read the vocabulary from cached training labels "
                         "instead, for boxes without Train.csv")
    ap.add_argument("--min-count", type=int, default=5,
                    help="each half must appear at least this often in training")
    ap.add_argument("--min-part", type=int, default=2,
                    help="shortest allowed half")
    ap.add_argument("--protect-count", type=int, default=1,
                    help="never split a word attested at least this often in training. 1 protects anything seen at all -- the conservative default, since splitting a real word costs exactly as much as fixing a merge saves")
    args = ap.parse_args()

    if not args.sub and not args.dump:
        ap.error("pass --sub to fix a submission or --dump to measure the rule")

    vocab = load_vocab(args.train_csv, args.cache_dir)
    apply, V = build_splitter(vocab, args.min_count, args.min_part,
                          args.protect_count)
    print(f"vocabulary: {len(vocab):,} types, {len(V):,} at >= {args.min_count} "
          f"occurrences")

    if args.dump:
        import jiwer
        rows = [json.loads(l) for l in args.dump.read_text().splitlines() if l.strip()]
        refs = [r["ref"] for r in rows]
        hyps = [r["hyp"] for r in rows]
        fixed, changed = zip(*(fix(h, apply) for h in hyps))
        n_changed = sum(changed)
        print(f"{len(rows):,} clips, {n_changed:,} tokens split\n")

        print(f"{'':<10}{'WER':>10}{'CER':>10}{'combined':>11}")
        for name, hs in (("before", hyps), ("after", list(fixed))):
            w, c = jiwer.wer(refs, hs), jiwer.cer(refs, hs)
            print(f"{name:<10}{w:>10.4f}{c:>10.4f}{0.5*(w+c):>11.4f}")
        w0, c0 = jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)
        w1, c1 = jiwer.wer(refs, list(fixed)), jiwer.cer(refs, list(fixed))
        print(f"{'delta':<10}{w1-w0:>+10.4f}{c1-c0:>+10.4f}{0.5*(w1+c1)-0.5*(w0+c0):>+11.4f}")

        if w1 < w0:
            print("\nWER improves -> apply it to the submission with --sub.")
            print("Note CER may worsen slightly; that is an acceptable trade here,")
            print("since CER is already at parity with first place and WER is not.")
        else:
            print("\nWER does not improve. Do not ship this. Try --min-count 20 to")
            print("split only on very common words, or drop the idea and use the LM.")

        # by language, if the dump carries it
        if rows and "lang" in rows[0]:
            print(f"\n{'lang':<6}{'n':>6}{'WER before':>12}{'WER after':>11}{'delta':>9}")
            for lang in sorted({r["lang"] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
                rr = [refs[i] for i in idx]
                a = jiwer.wer(rr, [hyps[i] for i in idx])
                b = jiwer.wer(rr, [fixed[i] for i in idx])
                print(f"{lang:<6}{len(idx):>6}{a:>12.4f}{b:>11.4f}{b-a:>+9.4f}")

        print("\nexamples:")
        shown = 0
        for h, f in zip(hyps, fixed):
            if h == f:
                continue
            for a, b in zip(h.split(), f.split()):
                pass
            print(f"  {h[:80]}\n  {f[:80]}\n")
            shown += 1
            if shown >= 3:
                break
        return 0

    sub = pd.read_csv(args.sub, escapechar="\\")
    assert list(sub.columns) == ["ID", "Target"], sub.columns
    fixed, changed = zip(*(fix(t, apply) for t in sub.Target.fillna("")))
    sub["Target"] = list(fixed)
    blank = sub.Target.astype(str).str.strip() == ""
    assert not blank.any(), "splitting produced a blank target"
    out = args.out or args.sub.with_name(args.sub.stem + "_split.csv")
    sub.to_csv(out, index=False)
    print(f"split {sum(changed):,} tokens across "
          f"{sum(1 for c in changed if c):,} of {len(sub):,} clips")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
