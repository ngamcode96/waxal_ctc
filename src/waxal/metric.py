"""Competition metric: 0.5 * WER + 0.5 * CER.

NOTE ON DIRECTION: everything here is an *error* rate, so lower is better. The
Zindi leaderboard displays `1 - error`, where higher is better -- a leaderboard
score of 0.781 means an error of 0.219. Don't compare the two directly.

Zindi does not publish whether the metric is corpus-level (total edits / total
reference length) or the mean of per-utterance rates. They differ, sometimes by
a lot, so we compute both and treat the gap as a warning sign. Calibrate against
the public leaderboard once we have a scored submission.
"""

from dataclasses import dataclass

import jiwer


@dataclass
class Score:
    wer: float
    cer: float
    combined: float
    wer_mean: float
    cer_mean: float
    combined_mean: float

    def __str__(self) -> str:
        return (
            f"corpus: WER {self.wer:.4f}  CER {self.cer:.4f}  -> {self.combined:.4f}\n"
            f"mean:   WER {self.wer_mean:.4f}  CER {self.cer_mean:.4f}  -> {self.combined_mean:.4f}"
        )


def score(refs: list[str], hyps: list[str]) -> Score:
    if len(refs) != len(hyps):
        raise ValueError(f"length mismatch: {len(refs)} refs vs {len(hyps)} hyps")

    # jiwer errors on empty references; a blank hypothesis is legal (and is what
    # a model emits when it hears nothing), so only guard the reference side.
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if len(pairs) < len(refs):
        print(f"warning: dropped {len(refs) - len(pairs)} pairs with empty reference")
    r_ok, h_ok = [p[0] for p in pairs], [p[1] for p in pairs]

    wer = jiwer.wer(r_ok, h_ok)
    cer = jiwer.cer(r_ok, h_ok)
    per_wer = [jiwer.wer(r, h) for r, h in pairs]
    per_cer = [jiwer.cer(r, h) for r, h in pairs]
    wer_mean = sum(per_wer) / len(per_wer)
    cer_mean = sum(per_cer) / len(per_cer)

    return Score(
        wer=wer,
        cer=cer,
        combined=0.5 * wer + 0.5 * cer,
        wer_mean=wer_mean,
        cer_mean=cer_mean,
        combined_mean=0.5 * wer_mean + 0.5 * cer_mean,
    )


def score_by_language(refs: list[str], hyps: list[str], langs: list[str]) -> dict[str, Score]:
    """Per-language breakdown. The overall metric hides which language is dragging."""
    out = {}
    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        out[lang] = score([refs[i] for i in idx], [hyps[i] for i in idx])
    return out
