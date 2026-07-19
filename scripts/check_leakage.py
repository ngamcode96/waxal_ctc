"""Audit for train/test contamination.

Three things could leak, in decreasing order of severity:

  1. Test clips present in training -- the model would have memorized them.
  2. Test *speakers* present in training -- not memorization, but the test set
     would be measuring voices the model has heard, not generalization.
  3. Test transcripts duplicated in training -- possible with scripted prompts,
     though WAXAL is spontaneous image description so it should be rare.

This reads only ids and speaker_ids from the test split. The transcription column
is dropped on load, so no ground-truth label is ever read -- this is an integrity
check, not a scoring script.

    python scripts/check_leakage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets

from waxal import data as wdata

META = ["id", "speaker_id", "language"]


def load_meta(lang: str, split: str) -> datasets.Dataset:
    """Ids and speakers only -- audio and transcription are never materialized."""
    ds = datasets.load_dataset(
        wdata.HF_REPO,
        data_files={split: f"data/ASR/{lang}/{lang}-{split}-*.parquet"},
        split=split,
    )
    return ds.remove_columns([c for c in ds.column_names if c not in META])


def main() -> int:
    train_ids, train_spk = set(), set()
    test_ids, test_spk = set(), set()

    for lang in wdata.LANGS:
        for split in ("train", "validation"):
            ds = load_meta(lang, split)
            train_ids.update(ds["id"])
            train_spk.update((lang, s) for s in ds["speaker_id"])
        ds = load_meta(lang, "test")
        test_ids.update(ds["id"])
        test_spk.update((lang, s) for s in ds["speaker_id"])
        print(f"  {lang}: train+val ids so far {len(train_ids):,}, "
              f"test ids {len(ds):,}")

    print(f"\ntrain+validation: {len(train_ids):,} clips, {len(train_spk):,} speakers")
    print(f"test:             {len(test_ids):,} clips, {len(test_spk):,} speakers")

    id_overlap = train_ids & test_ids
    print(f"\n1. clip id overlap: {len(id_overlap):,}")
    if id_overlap:
        print(f"   LEAK -- examples: {sorted(id_overlap)[:5]}")
    else:
        print("   clean: no test clip appears in training")

    spk_overlap = train_spk & test_spk
    pct = 100 * len(spk_overlap) / max(len(test_spk), 1)
    print(f"\n2. speaker overlap: {len(spk_overlap):,}/{len(test_spk):,} "
          f"test speakers ({pct:.1f}%)")
    if pct > 50:
        print("   test mostly reuses training voices -- it measures memorization")
        print("   of speakers more than generalization, and our speaker-disjoint")
        print("   validation is the harder, more honest measure")
    elif pct > 0:
        print("   partial overlap: test is a mix of heard and unheard voices")
    else:
        print("   clean: every test speaker is unheard, like our validation split")

    print("\n3. transcript duplication: not checked -- it would require reading")
    print("   the test labels, which this script deliberately does not do.")
    return 1 if id_overlap else 0


if __name__ == "__main__":
    sys.exit(main())
