"""Generate a self-contained Kaggle notebook from the repo source.

The rules say custom packages in a submission notebook won't be accepted, so the
notebook can't just clone this repo -- it has to carry its own source. Rather
than maintain a hand-copied duplicate that drifts, we embed the real files as
%%writefile cells at build time. The repo stays the single source of truth.

    python scripts/build_kaggle_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "waxal_kaggle.ipynb"

EMBED = [
    "src/waxal/__init__.py",
    "src/waxal/metric.py",
    "src/waxal/normalize.py",
    "src/waxal/hw.py",
    "src/waxal/data.py",
    "scripts/train_ctc.py",
    "scripts/infer.py",
    "scripts/bench.py",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


def embed_cell(rel: str) -> dict:
    body = (ROOT / rel).read_text()
    return code(f"%%writefile {rel}\n{body}")


def build() -> dict:
    cells: list[dict] = []

    cells.append(md("""
# WAXAL ASR — w2v-BERT 2.0 CTC

Joint model over Lingala / Luganda / Shona. Self-contained: every source file is
written to disk by the cells below, so there are no custom package dependencies.

**Phase 1 test labels are public and must not be used.** `waxal.data` refuses to
load the labeled test split; inference reads the test *audio* with the
transcription column dropped on load. Phase 1 leaderboard rank carries no signal —
watch the speaker-disjoint validation score instead.

Settings: **GPU T4 x2** (or P100), and **Internet ON** (the dataset streams from
Hugging Face).
"""))

    cells.append(md("## 1. Environment"))
    cells.append(code("""
!pip install -q -U "transformers>=4.44" datasets jiwer accelerate soundfile librosa

import os, pathlib
# /kaggle/working is capped at ~20 GB and the labeled audio is ~12.6 GB; keep the
# HF cache on the larger scratch volume so extraction doesn't hit the wall.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/temp/hf/datasets"
pathlib.Path("/kaggle/temp/hf").mkdir(parents=True, exist_ok=True)

!df -h /kaggle/working /kaggle/temp | head -5
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""))

    cells.append(md("""
Optional: a Hugging Face token avoids anonymous rate limits on the ~12.6 GB
download. Add it under **Add-ons → Secrets** as `HF_TOKEN`. The dataset is public,
so this only affects speed.
"""))
    cells.append(code("""
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded")
except Exception as e:
    print(f"no HF token ({type(e).__name__}) — continuing anonymously")
"""))

    cells.append(md("## 2. Source\n\nGenerated from the repo — edit there, not here."))
    # %%writefile does not create missing parent directories.
    cells.append(code("!mkdir -p src/waxal scripts"))
    for rel in EMBED:
        cells.append(embed_cell(rel))
    cells.append(code("""
import sys
sys.path.insert(0, "src")
from waxal.normalize import clean
from waxal.metric import score
assert clean("  Ndaba «x» 12 ⭐️  ") == 'Ndaba "x"'
print("modules OK")
"""))

    cells.append(md("""
## 2b. Inference only — score a trained model from the Hub

Run **just this section** (cells 1–2 then here) to produce a submission from an
already-trained model. No training, no feature extraction: it pulls the weights
from the Hub and transcribes the Phase 1 test audio (~1.3 GB download).

Needs `HF_TOKEN` in Add-ons → Secrets if the model repo is private.
"""))
    cells.append(code("""
# The only thing to change: which model to score with.
#   ngia/ctc-v1     epoch 5, validation 0.1617   <- best so far
#   ngia/ctc-v2     full-data run, epoch 2, 0.1747
#   ngia/ctc-v3     v1 + speed perturbation
MODEL = "ngia/ctc-v2"
OUT = "/kaggle/working/submission.csv"
"""))
    cells.append(code("""
!python scripts/infer.py \\
    --model {MODEL} \\
    --phase 1 \\
    --sample-submission /kaggle/input/waxal-csvs/SampleSubmission.csv \\
    --out {OUT} \\
    --batch-size 16
"""))
    cells.append(code("""
import pandas as pd
sub = pd.read_csv(OUT, escapechar="\\\\")
print(f"{MODEL}: {sub.shape} {list(sub.columns)}")
assert list(sub.columns) == ["ID", "Target"], sub.columns
empty = (sub.Target.fillna("").str.strip() == "").sum()
print(f"empty targets: {empty}/{len(sub)}")
# Degenerate CTC output ("Muta a a a a") shows up as repeated single letters.
degen = sub.Target.fillna("").str.contains(r"(?:\\b(\\w)\\b[ .]*){4,}", regex=True)
print(f"degenerate repeats: {degen.sum()} ({100*degen.mean():.1f}%)")
sub.head(10)
"""))

    cells.append(md("""
## 2c. Continue training an earlier run

Warm-starts from a finished model on the Hub: loads its weights, then trains
with a **fresh optimizer and LR schedule**. That is what you want after a run
completes — `--resume` would restore the old schedule, which has already decayed
to zero.

Pull the cached features first so this skips the 12.6 GB download and the ~10 min
feature extraction entirely. Use a **lower learning rate** than the original run:
the model is already trained, and 5e-5 would undo some of that.

Kaggle GPU sessions cap at ~9 hours, so size `--epochs` to fit.
"""))
    cells.append(code("""
!python scripts/sync_features.py pull --cache-dir /kaggle/temp/cache

!python scripts/train_ctc.py \\
    --init-from ngia/ctc-v1 \\
    --output-dir /kaggle/working/ctc-v1b \\
    --cache-dir /kaggle/temp/cache \\
    --push-to-hub ngia/ctc-v1b \\
    --hub-strategy end \\
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 2e-5 --seed 43
"""))

    cells.append(md("""
## 2d. Speed perturbation — **A100 only, will not fit a Kaggle session**

Each clip is resampled to 0.9x / 1.0x / 1.1x, tripling the training set. Pitch
shifts with speed, which is the point: each copy behaves like a different
speaker, and Phase 2 is scored on unheard voices.

Measured costs per epoch, batch 4 / accum 8:

| | rows | Kaggle T4 | A100 |
|---|---|---|---|
| normal | 31,316 | 5.4 h | 1.4 h |
| perturbed | 93,948 | **16.3 h** | 4.2 h |

A Kaggle GPU session caps around 9 hours, so one epoch overruns. It also changes
the cache key, so the pulled cache no longer matches and you would first pay a
12.6 GB download plus ~1.7 h of 3x extraction on a T4.

Run it on the A100 instead:

```bash
python scripts/train_ctc.py \\
    --init-from ngia/ctc-v2 \\
    --output-dir /dev/shm/ctc-v3 \\
    --cache-dir /dev/shm/cache-sp \\
    --speed-perturb 0.9,1.0,1.1 \\
    --push-to-hub ngia/ctc-v3 --hub-strategy end \\
    --epochs 2 --batch-size 4 --grad-accum 8 --lr 2e-5 --seed 44
```

Note the separate `--cache-dir`: perturbed features are a different (3x larger,
~30 GB) cache, and keeping them apart means the unperturbed one stays valid.
"""))

    cells.append(md("""
## 3. Smoke test

A few hundred rows end-to-end first. The full run costs hours; a typo shouldn't
cost you one of them. Expect a *terrible* score here — 200 rows trains nothing.
What matters is that it completes without raising.
"""))
    cells.append(code("""
!python scripts/train_ctc.py \\
    --output-dir /kaggle/temp/smoke \\
    --limit 200 --epochs 1 --batch-size 2 --grad-accum 1 \\
    --valid-frac 0.25 --num-proc 2
"""))

    cells.append(md("""
## 4. Full training run

Kaggle sessions are capped at 12 hours (9 for GPU) and the weekly GPU quota is 30
hours, so this is sized to fit one session rather than to be optimal — treat it as
a baseline to beat on RunPod, not the final model.

Sized for a **T4 (15 GB, no bf16)**: batch 4 with 8 accumulation steps holds the
effective batch at 32 while keeping activations inside memory. A 580M model in
mixed precision spends ~10 GB on weights, the fp32 master copy, and AdamW moments
before a single activation is stored.

`CUDA_VISIBLE_DEVICES=0` pins this to one GPU on purpose. With both visible, the
Trainer silently switches to DataParallel, which changes the effective batch size
and adds a failure mode to debug on the first real run. Drop the prefix to use
both once a single-GPU run is known good.

`group_by_length` matters here: these clips vary a lot in duration, and batching
similar lengths together cuts padding waste substantially.
"""))
    cells.append(code("""
# --cache-dir keeps the ~40min feature extraction across retries. /kaggle/temp
# dies with the session but /kaggle/working is capped at 20GB, and features
# (~11GB) plus checkpoints (~7GB) would leave no headroom there.
!CUDA_VISIBLE_DEVICES=0 python scripts/train_ctc.py \\
    --output-dir /kaggle/working/ctc-v1 \\
    --cache-dir /kaggle/temp/features \\
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 5e-5 \\
    --num-proc 4 --valid-frac 0.06 --seed 42
"""))
    cells.append(md("""
**If loss goes `nan` and stays there:** that's fp16 underflow in the CTC loss, not
a data problem. T4 can't do bf16, so the fixes are to lower the learning rate to
`3e-5`, or raise warmup to `--warmup-ratio 0.2`. If it persists, training in fp32
works but roughly halves throughput.

**If it OOMs:** drop to `--batch-size 2 --grad-accum 16`.
"""))

    cells.append(md("""
## 4b. Where is the time going?

Run this **with training stopped** — it needs the GPU to itself. It times
forward+backward on synthetic tensors and, if a feature cache exists, the real
dataloader, then says which one dominates.

A faster GPU only helps the first number. If the dataloader dominates, renting an
A100 buys nothing.
"""))
    cells.append(code("""
!python scripts/bench.py --batch-size 4 --grad-accum 8 \\
    --cache-dir /kaggle/temp/features
"""))

    cells.append(md("""
## 5. Validation

The honest number. `combined` is the competition metric on **held-out speakers**;
the per-language breakdown shows which language is dragging — Luganda has the
least data (5,455 rows vs ~14k each for the others), so expect it to lag.
"""))
    cells.append(code("""
import json, pathlib
state = json.loads(pathlib.Path("/kaggle/working/ctc-v1/best/trainer_state.json").read_text()) \\
    if pathlib.Path("/kaggle/working/ctc-v1/best/trainer_state.json").exists() else None
if state:
    rows = [h for h in state["log_history"] if "eval_combined" in h]
    for h in rows[-5:]:
        per = {k.replace("eval_combined_", ""): round(v, 4)
               for k, v in h.items() if k.startswith("eval_combined_")}
        print(f"step {h['step']:>6}  combined {h['eval_combined']:.4f}  "
              f"wer {h['eval_wer']:.4f}  cer {h['eval_cer']:.4f}  {per}")
else:
    print("no trainer_state.json — run training first")
"""))

    cells.append(md("""
## 6. Submission

Phase 1 predictions — for format validation and pipeline confidence only. The
leaderboard score it returns is not meaningful, since others may be submitting
lookups against the public labels.

When Phase 2 lands (~26 July), switch to `--phase 2 --phase2-dir <path>`.
"""))
    cells.append(code("""
!python scripts/infer.py \\
    --model /kaggle/working/ctc-v1/best \\
    --phase 1 \\
    --sample-submission /kaggle/input/waxal-csvs/SampleSubmission.csv \\
    --out /kaggle/working/submission.csv \\
    --batch-size 8
"""))
    cells.append(code("""
import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv", escapechar="\\\\")
sample = pd.read_csv("/kaggle/input/waxal-csvs/SampleSubmission.csv", escapechar="\\\\")
assert list(sub.columns) == ["ID", "Target"], sub.columns
assert len(sub) == len(sample), (len(sub), len(sample))
assert sub.ID.tolist() == sample.ID.tolist(), "ID order must match SampleSubmission"
print(f"submission valid — {len(sub):,} rows")
sub.head()
"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}  ({len(nb['cells'])} cells)")
