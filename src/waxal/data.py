"""Dataset assembly for the WAXAL ASR challenge.

Two things this module is careful about:

1. The HF `test` split is the Phase 1 test set and its labels are public. We
   never load it. Phase 1 rank is meaningless and using those labels is an
   explicit rules breach; Phase 2 decides the prizes.

2. Phase 2 ships no metadata at all -- no language tag, no speaker id. So the
   model must be language-agnostic at inference, and our validation split must
   be speaker-disjoint or it will flatter us exactly where Phase 2 will punish.
"""

from dataclasses import dataclass

import datasets

LANGS = ("lin", "lug", "sna")
HF_REPO = "google/WaxalNLP"
SR = 16_000


def _files(lang: str, split: str) -> str:
    return f"data/ASR/{lang}/{lang}-{split}-*.parquet"


def load_labeled(langs=LANGS, splits=("train", "validation"), num_proc: int = 4):
    """Load the labeled portion of the target languages. Never touches `test`."""
    if "test" in splits:
        raise ValueError(
            "the HF `test` split is the Phase 1 test set with public labels -- "
            "loading it risks contaminating training and breaches the rules"
        )
    parts = []
    for lang in langs:
        for split in splits:
            ds = datasets.load_dataset(
                HF_REPO, data_files={split: _files(lang, split)}, split=split,
                num_proc=num_proc,
            )
            parts.append(ds)
    ds = datasets.concatenate_datasets(parts)
    return ds.cast_column("audio", datasets.Audio(sampling_rate=SR))


def load_test_audio(langs=LANGS, num_proc: int = 4):
    """Phase 1 test *audio only* -- the transcription column is dropped on load.

    Running our model over the Phase 1 test audio and submitting the predictions
    is legitimate: it validates the submission format and the inference path.
    What the rules forbid is using the public ground-truth labels. Dropping the
    column here means those labels never enter the process at all, so there is
    no path by which they could leak into a submission.
    """
    parts = []
    for lang in langs:
        ds = datasets.load_dataset(
            HF_REPO, data_files={"test": _files(lang, "test")}, split="test",
            num_proc=num_proc,
        )
        parts.append(ds.remove_columns([c for c in ds.column_names
                                        if c not in ("id", "audio")]))
    ds = datasets.concatenate_datasets(parts)
    return ds.cast_column("audio", datasets.Audio(sampling_rate=SR))


def load_phase2_audio(path: str, num_proc: int = 4):
    """Phase 2 evaluation audio, whatever form it arrives in.

    Phase 2 ships no metadata, so this deliberately assumes nothing beyond an id
    and an audio payload. Adjust the loader once the actual format is published.
    """
    ds = datasets.load_dataset("audiofolder", data_dir=path, split="train",
                               num_proc=num_proc)
    return ds.cast_column("audio", datasets.Audio(sampling_rate=SR))


@dataclass
class Split:
    train: datasets.Dataset
    valid: datasets.Dataset

    def __str__(self) -> str:
        return f"train={len(self.train):,}  valid={len(self.valid):,}"


def speaker_disjoint_split(ds, valid_frac: float = 0.06, seed: int = 42) -> Split:
    """Hold out whole speakers, stratified by language.

    A random row-level split leaks speakers across the boundary: the model
    memorizes voices and validation reports a score the Phase 2 set will not
    reproduce. We hold out entire speakers per language instead, so validation
    measures generalization to unheard voices -- which is what Phase 2 is.
    """
    import random

    by_lang: dict[str, set[str]] = {}
    for lang, spk in zip(ds["language"], ds["speaker_id"]):
        by_lang.setdefault(lang, set()).add(spk)

    rng = random.Random(seed)
    held: set[tuple[str, str]] = set()
    for lang, spks in by_lang.items():
        spks = sorted(spks)
        rng.shuffle(spks)
        n = max(1, round(len(spks) * valid_frac))
        held.update((lang, s) for s in spks[:n])
        print(f"  {lang}: holding out {n}/{len(spks)} speakers")

    flags = [(l, s) in held for l, s in zip(ds["language"], ds["speaker_id"])]
    idx_v = [i for i, f in enumerate(flags) if f]
    idx_t = [i for i, f in enumerate(flags) if not f]
    return Split(train=ds.select(idx_t), valid=ds.select(idx_v))


def filter_usable(ds, min_s: float = 0.5, max_s: float = 30.0, min_chars: int = 3):
    """Drop rows that would waste compute or destabilize CTC.

    CTC requires the target to be no longer than the encoder output, so clips
    that are too short for their transcript produce inf loss. Over-long clips
    blow up memory quadratically in attention for a handful of examples.
    """
    from .normalize import clean

    def ok(row) -> bool:
        txt = clean(row["transcription"] or "")
        if len(txt) < min_chars:
            return False
        n = row["audio"]["num_samples"] if "num_samples" in row["audio"] else None
        if n is None:
            return True
        dur = n / SR
        if not (min_s <= dur <= max_s):
            return False
        # w2v-BERT downsamples ~2x per 10ms frame -> ~25 frames/sec of output.
        return len(txt) <= dur * 25

    return ds.filter(ok, desc="filtering usable clips")
