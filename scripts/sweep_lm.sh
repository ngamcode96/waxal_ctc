#!/usr/bin/env bash
# Sweep the LM decoding parameters on a subset of validation.
#
# The pyctcdecode defaults (alpha 0.5, beta 1.5) come from English LibriSpeech
# recipes and made things worse here: 0.1617 greedy -> 0.1776.
#
# beta is a word *insertion bonus*: positive pays the decoder to emit more,
# shorter words, negative pays it to merge. The first sweep only tried beta <= 0
# on the theory that 6-7 character words should not be split. That was half the
# story -- measured 2026-08-03 on the corrected Phase 2 set, our CER is within
# 0.0008 of first place while our WER is 0.0226 worse, which is the signature of
# boundary errors in BOTH directions:
#
#     ref 'kabedo neni'  hyp 'kabedoneni'   we merged  -> wants beta > 0
#     ref 'giit'         hyp 'gi it'        we split   -> wants beta < 0
#
# So the grid spans both signs. An n-gram LM fixes either direction on its own
# merits -- it prefers real words over pseudo-words regardless -- and beta only
# sets the overall bias.
#
#   bash scripts/sweep_lm.sh ngia/ctc-v2-avg /workspace/cache-v2 /workspace/lm 500
#
# WER is what matters here, not combined: CER is already at parity, so a setting
# that trades CER for WER is still a win. Watch the WER column.
#
# Compare every line against the greedy baseline printed first. If none beat it,
# the LM is not worth its inference cost -- say so and move on.
set -u

MODEL="${1:?model}"
CACHE="${2:?cache-dir}"
LMDIR="${3:?lm dir}"
N="${4:-500}"

echo "=== greedy baseline (${N} clips) ==="
python scripts/eval_checkpoint.py --model "$MODEL" --cache-dir "$CACHE" \
    --limit "$N" --batch-size 32 2>/dev/null | grep -E "^corpus:|^  (lin|lug|sna):"

for alpha in ${ALPHAS:-0.2 0.4 0.6}; do
  for beta in ${BETAS:--1.0 -0.5 0.0 0.5 1.0}; do
    echo
    echo "=== alpha=$alpha beta=$beta ==="
    python scripts/eval_checkpoint.py --model "$MODEL" --cache-dir "$CACHE" \
        --lm "$LMDIR/5gram.arpa" --unigrams "$LMDIR/unigrams.txt" \
        --alpha "$alpha" --beta "$beta" \
        --limit "$N" --batch-size 32 2>/dev/null \
      | grep -E "^corpus:|^  (lin|lug|sna):"
  done
done
