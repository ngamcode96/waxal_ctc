#!/usr/bin/env bash
# Sweep the LM decoding parameters on a subset of validation.
#
# The pyctcdecode defaults (alpha 0.5, beta 1.5) come from English LibriSpeech
# recipes and made things worse here: 0.1617 greedy -> 0.1776. beta is a word
# *insertion bonus*, so a positive value pays the decoder to split words -- the
# wrong direction for languages whose words run 6-7 characters.
#
#   bash scripts/sweep_lm.sh ngia/ctc-v1 /dev/shm/cache /workspace/lm 500
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

for alpha in 0.2 0.4 0.6; do
  for beta in -2.0 -1.0 0.0; do
    echo
    echo "=== alpha=$alpha beta=$beta ==="
    python scripts/eval_checkpoint.py --model "$MODEL" --cache-dir "$CACHE" \
        --lm "$LMDIR/5gram.arpa" --unigrams "$LMDIR/unigrams.txt" \
        --alpha "$alpha" --beta "$beta" \
        --limit "$N" --batch-size 32 2>/dev/null \
      | grep -E "^corpus:|^  (lin|lug|sna):"
  done
done
