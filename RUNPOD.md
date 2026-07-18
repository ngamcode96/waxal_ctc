# Running on RunPod

End-to-end guide for training the WAXAL CTC model on a rented A100.

---

## 1. Pod configuration

| setting | value | why |
|---|---|---|
| GPU | **A100 SXM 80GB** | 16 vCPU vs PCIe's 12 — feature extraction is CPU-bound |
| Template | any **PyTorch** image | torch comes preinstalled and driver-matched |
| Container disk | **40 GB** | the 20 GB default fills with pip/uv caches |
| Storage | **Network volume, 150 GB** | survives pod termination; the feature cache is worth ~40 min |

Do **not** pick "Volume disk" — it is deleted when the pod is terminated, so every
new pod re-downloads 13 GB and re-extracts features. Network volume mounts at
`/workspace` and costs ~$0.07/GB/mo (~$2.50 for a week at 150 GB).

### Disk budget

| item | size |
|---|---|
| HF dataset cache (3 languages, labeled only) | ~13 GB |
| extracted features | ~16 GB |
| checkpoints (2.3 GB x `save_total_limit=3` + best) | ~9 GB |
| **total** | **~38 GB** |
| with `--speed-perturb 0.9,1.0,1.1` | **~70 GB** |

---

## 2. Environment

Set these **before** anything downloads. By default the HF cache goes to
`~/.cache/huggingface` on the *container* disk, which will fill mid-download.

```bash
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf/datasets
export HF_TOKEN=hf_...            # WRITE permission required

# survive SSH drops -- a disconnect otherwise kills training
apt-get update -qq && apt-get install -y -qq tmux
tmux new -s train
```

To make the environment persist across shells in the pod:

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf/datasets
EOF
```

---

## 3. Install

```bash
git clone <your-repo> /workspace/waxal
cd /workspace/waxal
pip install -e ".[train]"
```

RunPod images ship `pip`, not `uv`, and nothing here needs uv. If you prefer it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --system -e ".[train]"      # --system is required, no venv here
```

No virtualenv either way: the pod is disposable and its torch is built against
the image's driver. The `train` extra deliberately excludes torch so this never
replaces it.

Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0,'src'); from waxal import hw; print(hw.describe())"
python -c "import transformers, datasets, torchcodec; print(transformers.__version__, datasets.__version__)"
```

**transformers 5.x needs torch >= 2.5** — it imports `DTensor` from
`torch.distributed.tensor`. RunPod images ship torch 2.4.1, so `pyproject.toml`
caps transformers below 5. If you see
`ImportError: cannot import name 'DTensor'`, run `pip install "transformers<5"`.
Do not upgrade torch to fix it: that is a multi-GB reinstall and risks breaking
the driver/CUDA match. `torchcodec` is likewise ABI-coupled to torch; if it
fails to import, pin it to a build matching your torch rather than moving torch.

Expect `True` and `bf16_native=True`. If bf16 shows False on an A100, something
is wrong with the driver — stop and investigate rather than training 3x slow.

---

## 4. Pre-flight checks

Cheap checks first, so failures don't surface an hour into a paid run.

```bash
# Hub write access (fails at the FIRST EPOCH otherwise, ~25 min in)
python scripts/check_hub.py --repo ngia/ctc-v1

# Full pipeline on 200 rows, ~3 min. Score will be garbage; that is fine.
python scripts/train_ctc.py --output-dir /workspace/smoke \
    --limit 200 --epochs 1 --batch-size 4 --grad-accum 1 --valid-frac 0.25
```

The smoke test exercises data loading, filtering, feature extraction, the
length-grouped sampler, the training loop, and metric computation. If it
finishes, the stack works on this box.

---

## 5. Train

```bash
python scripts/train_ctc.py \
    --output-dir /workspace/ctc-v1 \
    --cache-dir /workspace/cache \
    --push-to-hub \
    --epochs 5 \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 5e-5 \
    --num-proc 16 \
    --seed 42
```

First run spends ~30 min downloading and ~30 min extracting features before the
first training step. Both are cached; later runs start training immediately.

Bare `--push-to-hub` uploads to `ngia/<output-dir-name>` — here `ngia/ctc-v1`.
Repos are created **private** (the rules forbid sharing outside your team).

### Key flags

| flag | default | notes |
|---|---|---|
| `--epochs` | 6.0 | 5 is a reasonable first run |
| `--batch-size` | 8 | 16 on an 80 GB A100 |
| `--grad-accum` | 4 | effective batch = batch-size x grad-accum |
| `--lr` | 5e-5 | drop to 3e-5 if loss goes to nan |
| `--num-proc` | 8 | 16 on SXM; lower if extraction exhausts RAM |
| `--cache-dir` | none | **always set it**; skips ~30 min re-extraction |
| `--max-s` / `--min-s` | 30.0 / 0.5 | clip duration bounds; part of the cache key |
| `--valid-frac` | 0.06 | fraction of *speakers* held out, not rows |
| `--save-strategy` | epoch | `steps` + `--save-steps N` for finer granularity |
| `--resume` | off | continue from the last checkpoint in `--output-dir` |
| `--push-to-hub` | off | bare flag = `ngia/<output-dir-name>` |
| `--hub-public` | off | only after the challenge closes |

### Augmentation

SpecAugment is applied inside the model at train time, so these **do not
invalidate the feature cache** — sweep them freely:

```bash
--mask-time-prob 0.065 --mask-feature-prob 0.02
```

Speed perturbation changes the audio, so it **does** invalidate the cache and
multiplies extraction time and disk by the number of factors:

```bash
--speed-perturb 0.9,1.0,1.1
```

Augmentation applies to the training split only — a perturbed validation set
would not be comparable to the baseline.

---

## 6. Check where time is going

Run after extraction completes, with training stopped:

```bash
python scripts/bench.py --cache-dir /workspace/cache --batch-size 16 --grad-accum 2
```

Reports the real clip-length distribution and whether training is GPU-bound or
data-bound. If data-bound, raise `dataloader_num_workers` in
`scripts/train_ctc.py` (currently hardcoded to 4) — you have 16 vCPUs.

---

## 7. What to watch

The metric is `0.5 x WER + 0.5 x CER`, lower is better. Every epoch prints:

```
eval_combined  eval_wer  eval_cer  eval_combined_lin  eval_combined_lug  eval_combined_sna
```

- **`eval_combined` is your only honest signal.** The Phase 1 leaderboard is
  contaminated by design (public test labels) and means nothing.
- **Watch `eval_combined_lug`.** Luganda has 5,455 training rows against ~14k
  for the others. If it lags badly, that is the highest-leverage fix — and
  Phase 2 gives no language tag, so it cannot be patched at inference.
- **`nan` loss** should not happen with native bf16. If it does, lower `--lr`.
- Validation is **speaker-disjoint**, so it measures generalization to unheard
  voices — which is what Phase 2 tests.

---

## 8. Recovery

Checkpoints are written every epoch to `--output-dir` and uploaded to the Hub.

```bash
# same command plus --resume
python scripts/train_ctc.py --output-dir /workspace/ctc-v1 \
    --cache-dir /workspace/cache --push-to-hub --resume \
    --epochs 5 --batch-size 16 --grad-accum 2 --lr 5e-5 --num-proc 16
```

`--resume` restores model, optimizer, and scheduler state. Feature extraction is
skipped entirely thanks to `--cache-dir`.

If the pod itself is gone, a new pod on the same **network volume** still has both
the cache and the checkpoints. That is the whole reason for choosing it.

---

## 9. Inference

Phase 1 (format validation only — its leaderboard score is meaningless):

```bash
python scripts/infer.py \
    --model /workspace/ctc-v1/best \
    --phase 1 \
    --sample-submission data/raw/SampleSubmission.csv \
    --out /workspace/submission.csv \
    --batch-size 16
```

Phase 2 (~26 July, decides the prizes):

```bash
python scripts/infer.py \
    --model /workspace/ctc-v1/best \
    --phase 2 --phase2-dir /workspace/phase2_audio \
    --sample-submission /workspace/Phase2SampleSubmission.csv \
    --out /workspace/submission_phase2.csv
```

Phase 2 ships no metadata, so `load_phase2_audio` assumes only an id and an audio
payload. Adjust it once the real format is published.

Download results before terminating the pod:

```bash
runpodctl send /workspace/submission.csv
```

---

## 10. Cost

| phase | time | cost @ $1.49/h |
|---|---|---|
| setup + smoke | ~15 min | $0.40 |
| download + extraction (once) | ~60 min | $1.50 |
| training, 5 epochs | ~1.5–3 h | $2–4.50 |
| **first full run** | **~3–4 h** | **~$4–6** |
| later runs (cache warm) | ~1.5–3 h | $2–4.50 |
| network volume, 150 GB | per week | ~$2.50 |

**Stop the pod when not training.** Storage keeps costing while stopped
($0.20/GB/mo for volume disk) but the GPU does not.

---

## 11. Gotchas

- **Set `HF_HOME` before downloading**, or 13 GB lands on the container disk.
- **Always pass `--cache-dir`** — without it every run re-extracts (~30 min).
- **`--max-s` / `--min-s` are in the cache key.** Changing them correctly
  triggers re-extraction; changing `--lr` or `--epochs` correctly does not.
- **Never load the HF `test` split.** Those are Phase 1's public labels and using
  them breaches the rules. `waxal.data` refuses to load it.
- **Use tmux.** An SSH drop kills a foreground training run.
- **`--num-proc 16` can exhaust RAM** during extraction (117 GB, 16 workers each
  holding decoded audio). Drop to 8 if you see OOM kills — that is system RAM,
  not VRAM.
