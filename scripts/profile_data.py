"""Profile the competition CSVs. Schema-agnostic: we don't know the columns yet.

Usage: .venv/bin/python scripts/profile_data.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def guess(df: pd.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for key, orig in lower.items():
            if cand in key:
                return orig
    return None


def profile(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"MISSING  {path.name}")
        return None
    # The CSVs escape embedded quotes with backslashes, not CSV-standard doubled
    # quotes. Without escapechar, 23 train rows silently shred into extra fields.
    df = pd.read_csv(path, escapechar="\\")
    print(f"\n{'=' * 70}\n{path.name}: {len(df):,} rows x {len(df.columns)} cols")
    print(f"columns: {list(df.columns)}")
    print(f"\nfirst 3 rows:\n{df.head(3).to_string(max_colwidth=60)}")
    nulls = df.isna().sum()
    if nulls.any():
        print(f"\nnulls:\n{nulls[nulls > 0].to_string()}")
    return df


def main() -> int:
    train = profile(RAW / "Train.csv")
    test = profile(RAW / "Test.csv")
    profile(RAW / "SampleSubmission.csv")

    if train is None:
        print("\nDownload the CSVs from Zindi into data/raw/ first.")
        return 1

    lang_col = guess(train, "lang", "locale")
    text_col = guess(train, "transcript", "text", "target", "sentence")
    id_col = guess(train, "id")
    spk_col = guess(train, "speaker")
    print(f"\n{'=' * 70}\ninferred: id={id_col} lang={lang_col} text={text_col} speaker={spk_col}")

    if lang_col:
        print(f"\nlanguage distribution (train):\n{train[lang_col].value_counts().to_string()}")
        if test is not None and lang_col in test.columns:
            print(f"\nlanguage distribution (test):\n{test[lang_col].value_counts().to_string()}")

    if spk_col and test is not None and spk_col in test.columns:
        # Speaker overlap decides how we split validation. If test speakers are
        # unseen, a random split flatters us and the leaderboard will disagree.
        tr_spk, te_spk = set(train[spk_col]), set(test[spk_col])
        print(f"\nspeakers: train={len(tr_spk)} test={len(te_spk)} overlap={len(tr_spk & te_spk)}")

    if text_col:
        txt = train[text_col].dropna().astype(str)
        words = txt.str.split().str.len()
        print(f"\ntranscript words: mean={words.mean():.1f} median={words.median():.0f} "
              f"p95={words.quantile(0.95):.0f} max={words.max()}")
        vocab = Counter(w for t in txt for w in t.split())
        print(f"vocab: {len(vocab):,} types, {sum(vocab.values()):,} tokens")
        print(f"top 20: {[w for w, _ in vocab.most_common(20)]}")

        chars = Counter(c for t in txt for c in t)
        print(f"\ncharset: {len(chars)} distinct")
        print(f"all chars: {''.join(sorted(chars))!r}")
        rare = {c: n for c, n in chars.items() if n < 50}
        print(f"rare chars (<50 occurrences, likely noise to normalize away): {sorted(rare)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
