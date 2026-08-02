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
    "scripts/infer.py",            # eval_checkpoint and pseudo_label import this
    "scripts/infer_phase2.py",     # imports infer.py and pseudo_label.py
    "scripts/selftest_phase2.py",  # invokes infer_phase2.py
    "scripts/gemma_zeroshot.py",   # imports selftest_phase2.py
    "scripts/lid.py",              # imports infer_phase2.py
    "scripts/bench.py",
    "scripts/eval_checkpoint.py",
    "scripts/sync_features.py",
    "scripts/pseudo_label.py",
    "scripts/push_checkpoint.py",
    "scripts/average_checkpoints.py",
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

**Phase 2 is live** (landed 27 July): 1,500 clips, 18–30 s each, no language tag
and no speaker id. Scoring is on Phase 2 alone — go to **section 6**.

**Phase 1 test labels are public and must not be used.** `waxal.data` refuses to
load the labeled test split; inference reads the test *audio* with the
transcription column dropped on load. Phase 1 leaderboard rank carries no signal —
watch the speaker-disjoint validation score instead.

Settings: **GPU T4 x2** (or P100), and **Internet ON** (the dataset streams from
Hugging Face).
"""))

    cells.append(md("## 1. Environment"))
    cells.append(code("""
!pip install -q -U "transformers>=4.44" datasets jiwer accelerate soundfile librosa scipy

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
## 2b. Phase 1 inference — pipeline check only

Produces a Phase 1 submission from a trained model on the Hub. Keep this for
validating the inference path end-to-end; **it is not what gets scored** — see
section 6 for the Phase 2 submission.

Needs `HF_TOKEN` in Add-ons → Secrets if the model repo is private.
"""))
    cells.append(code("""
# The only thing to change: which model to score with.
#   ngia/ctc-v1     epoch 5, validation 0.1617
#   ngia/ctc-v2     full-data run, epoch 2, 0.1747
#   ngia/ctc-v2-avg last three checkpoints averaged, 0.1572  <- best so far
#   ngia/ctc-v3     v1 + speed perturbation
MODEL = "ngia/ctc-v2-avg"
OUT = "/kaggle/working/submission_phase1.csv"
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
## 2e. Pseudo-labelling the unlabeled pool

The labeled set is ~180 hours; the unlabeled pool is ~78 GB across the three
languages. Transcribing it with our best model and training on the confident
outputs is the one remaining technique with the right order of magnitude —
augmentation and checkpoint averaging give 5–10% relative, self-training can
give 20–30%.

Output is a small CSV (id, language, transcription, confidence), **not**
features: re-extracting features on the training GPU is far cheaper than moving
~10 GB of arrays off Kaggle.

Nothing here goes near the test split — the unlabeled pool has no transcriptions
at all, so there is no label to leak.

Each shard is ~0.5 GB per language. Start with 2 shards to see the confidence
distribution before committing a whole session, then raise `--shards` and run
again. On a T4 expect roughly 25–35 minutes per 4,000 clips.
"""))
    cells.append(code("""
!python scripts/pseudo_label.py \\
    --model ngia/ctc-v2-avg \\
    --shards 0 1 \\
    --out /kaggle/working/pseudo_00.csv \\
    --batch-size 16
"""))
    cells.append(md("""
### Using both T4s

Kaggle gives two GPUs. Rather than `DataParallel` — which splits each batch and
pays a gather on every step — give each GPU its own shards and concatenate the
CSVs. That scales near-linearly and needs no code changes.

Caveat: the instance has only 4 CPU cores and audio decoding is CPU-bound, so
expect ~1.6x rather than a clean 2x.
"""))
    cells.append(code("""
import os, subprocess, time

JOBS = [("0", ["0", "1"], "/kaggle/working/pseudo_gpu0.csv"),
        ("1", ["2", "3"], "/kaggle/working/pseudo_gpu1.csv")]

procs = []
for gpu, shards, out in JOBS:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    cmd = ["python", "scripts/pseudo_label.py",
           "--model", "ngia/ctc-v2-avg", "--shards", *shards,
           "--out", out, "--batch-size", "16"]
    log = open(out.replace(".csv", ".log"), "w")
    procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=log), out, log))
    print(f"GPU {gpu}: shards {shards} -> {out}")

t0 = time.time()
for p, out, log in procs:
    p.wait(); log.close()
    print(f"{out}: exit {p.returncode}  ({time.time()-t0:.0f}s elapsed)")
"""))
    cells.append(code("""
!tail -25 /kaggle/working/pseudo_gpu0.log
"""))
    cells.append(code("""
import pandas as pd, glob
parts = [pd.read_csv(f) for f in sorted(glob.glob("/kaggle/working/pseudo_gpu*.csv"))]
p = pd.concat(parts, ignore_index=True)
# Shards are disjoint, but a rerun could double up -- make that impossible.
before = len(p)
p = p.drop_duplicates(subset="id")
print(f"{before:,} rows -> {len(p):,} after dedup by id")
p.to_csv("/kaggle/working/pseudo_00.csv", index=False)
"""))
    cells.append(code("""
import pandas as pd
p = pd.read_csv("/kaggle/working/pseudo_00.csv")
print(p.shape)
print(p.groupby("language").confidence.describe()[["count","25%","50%","75%"]])
# Pick a threshold from this: high confidence means the model was decisive at
# every frame, which correlates with the transcription being right.
for q in (0.5, 0.6, 0.7, 0.8):
    print(f"  conf >= {q}: {(p.confidence >= q).sum():,} rows "
          f"({100*(p.confidence >= q).mean():.0f}%)")
p.sort_values("confidence", ascending=False).head(3)[["language","confidence","transcription"]]
"""))
    cells.append(code("""
# Push to the Hub so the training pod can pull it.
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("ngia/waxal-pseudo", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="/kaggle/working/pseudo_00.csv",
                path_in_repo="pseudo_00.csv",
                repo_id="ngia/waxal-pseudo", repo_type="dataset")
print("pushed -> https://huggingface.co/datasets/ngia/waxal-pseudo")
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
## 6. Phase 2 submission — **this is the one that scores**

**Zindi replaced the Phase 2 test data on 2 August** — the set released the week
before was confirmed wrong. This section uses the corrected files. Everything
about the old set (1,500 clips, 16 kHz, 18–30 s) is obsolete, and no id carries
over.

    https://storage.googleapis.com/waxalphase2/newaudios.zip   (1.09 GB, 892 wavs)

| | old (wrong) | corrected |
|---|---|---|
| clips | 1,500 | **892** |
| sample rate | 16 kHz | **48 kHz** |
| duration | 18.0–30.0 s | **1.0–35.2 s** |
| total audio | 9.28 h | **5.00 h** |

**The 48 kHz matters more than it looks.** The model is trained at 16 kHz, and
feeding it 48 kHz does not raise — it produces audio that still sounds like
speech and still transcribes, just wrongly. `waxal.data.resample` downsamples
with a polyphase anti-aliasing filter, which needs **scipy** installed.

`Test_phase2.csv` is not at that URL, so attach the corrected one as a Kaggle
dataset. The cell below finds it under `/kaggle/input` whatever you named the
dataset. **Make sure it is the new CSV** — the old one's ids match nothing.

`scripts/infer_phase2.py` is used rather than `infer.py --phase 2`: Phase 2 is a
directory of wavs plus a CSV of ids, and the audiofolder loader behind
`--phase 2` returns an `audio` column with no `id`, so there is no way to say
which prediction belongs to which clip. Here the CSV is the authority for both
the id set and the row order.
"""))
    cells.append(code("""
import glob, os, pathlib, subprocess

AUDIO_URL = "https://storage.googleapis.com/waxalphase2/newaudios.zip"
WORK = pathlib.Path("/kaggle/temp/phase2")
WORK.mkdir(parents=True, exist_ok=True)

# Find the attached Phase 2 dataset, whatever it was named.
csvs = glob.glob("/kaggle/input/**/Test_phase2.csv", recursive=True)
assert csvs, "attach a dataset containing Test_phase2.csv (Add data -> your upload)"
PHASE2_CSV = csvs[0]
src = pathlib.Path(PHASE2_CSV).parent
print(f"csv:  {PHASE2_CSV}")

# Audio, in order of preference: already extracted in the dataset, a zip in the
# dataset, or the organizers' URL. /kaggle/input is read-only, so any extract
# has to land on /kaggle/temp.
# The corrected archive extracts to newaudios/, not audio/.
dirs = [d for d in glob.glob(str(src / "**" / "newaudios"), recursive=True)
        if os.path.isdir(d)]
zips = glob.glob(str(src / "**" / "*.zip"), recursive=True)

if dirs:
    PHASE2_AUDIO = dirs[0]
    print(f"using the extracted audio already in the dataset")
else:
    PHASE2_AUDIO = str(WORK / "newaudios")
    if zips:
        archive = zips[0]
        print(f"using the zip from the dataset: {archive}")
    else:
        archive = str(WORK / "newaudios.zip")
        # -C - resumes a partial file, so a dropped connection costs only the
        # remainder rather than another 1.09 GB. --retry covers a flaky start.
        print(f"downloading {AUDIO_URL}")
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-C", "-",
                        "-o", archive, AUDIO_URL], check=True)
        # Checked against the URL on 2026-08-02. A truncated download would
        # otherwise surface as a confusing unzip error several minutes later.
        got = os.path.getsize(archive)
        assert got == 1_086_719_156, f"expected 1,086,719,156 bytes, got {got:,}"
        print(f"downloaded {got / 1e6:.0f} MB")

    # -x '__MACOSX/*' matters: the archive carries a ._<ID>.wav resource fork
    # beside every clip, and they are not audio.
    subprocess.run(["unzip", "-q", "-o", archive, "-x", "__MACOSX/*",
                    "-d", str(WORK)], check=True)

import pandas as pd
N_EXPECTED = len(pd.read_csv(PHASE2_CSV, escapechar="\\\\"))
n_wav = len(glob.glob(os.path.join(PHASE2_AUDIO, "*.wav")))
print(f"audio: {PHASE2_AUDIO}  ({n_wav:,} wavs)   csv rows: {N_EXPECTED:,}")
assert n_wav == N_EXPECTED, (
    f"{n_wav:,} wavs but {N_EXPECTED:,} csv rows -- if the csv has 1,500 rows "
    f"it is the OLD one, whose ids match nothing in the corrected audio")
assert n_wav == 892, f"expected 892 corrected Phase 2 clips, found {n_wav:,}"
"""))
    cells.append(code("""
# Which model to score with. The corrected test set reopened this question:
#
#   ngia/ctc-v2-avg   3 languages (lin lug sna).  On those three: 0.150
#   ngia/ctc-p2-avg   6 languages (+ ach nyn sog). On those three: 0.156,
#                     but far better on Acholi / Runyankole / Lusoga
#
# ctc-p2 traded a little lin/lug/sna accuracy for a lot of East African, which
# was right for the old (wrong) test set. Whether it is right now depends on
# what the corrected set actually contains -- run 6d first and let the measured
# language mix decide. If it comes back lin/lug/sna, use ctc-v2-avg.
P2_MODEL = "ngia/ctc-v2-avg"
P2_OUT = "/kaggle/working/submission.csv"
"""))
    cells.append(md("""
Smoke test on 16 clips first. The full pass is ~9.3 hours of audio; a bad model
path or a missing vocab should cost seconds to discover, not the whole run.
"""))
    cells.append(code("""
!python scripts/infer_phase2.py \\
    --model {P2_MODEL} \\
    --test-csv {PHASE2_CSV} \\
    --audio-dir {PHASE2_AUDIO} \\
    --out /kaggle/temp/p2_smoke.csv \\
    --limit 16 --batch-size 4
"""))
    cells.append(md("""
### Full pass, both T4s

Each GPU takes every other clip and writes its own partial CSV, which is then
concatenated — the same shard-and-merge shape as pseudo-labelling, and for the
same reason: `DataParallel` splits each batch and pays a gather on every step.

Clips are 18–30 s, so `--batch-size 8` rather than Phase 1's 16. The script
sorts long-to-short and bisects a batch that OOMs instead of dropping it, so no
clip can go missing from the submission.

Drop to a single GPU by setting `JOBS = [("0", 0, 1, P2_OUT)]`.
"""))
    cells.append(code("""
import os, subprocess, time

# (gpu, shard, num_shards, out)
JOBS = [("0", 0, 2, "/kaggle/temp/p2_gpu0.csv"),
        ("1", 1, 2, "/kaggle/temp/p2_gpu1.csv")]

procs = []
for gpu, shard, n, out in JOBS:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    cmd = ["python", "scripts/infer_phase2.py",
           "--model", P2_MODEL,
           "--test-csv", PHASE2_CSV,
           "--audio-dir", PHASE2_AUDIO,
           "--out", out,
           "--shard", str(shard), "--num-shards", str(n),
           "--batch-size", "8", "--num-proc", "2"]
    log = open(out.replace(".csv", ".log"), "w")
    procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=log), out, log))
    print(f"GPU {gpu}: shard {shard}/{n} -> {out}")

t0 = time.time()
failed = []
for p, out, log in procs:
    p.wait(); log.close()
    print(f"{out}: exit {p.returncode}  ({time.time()-t0:.0f}s elapsed)")
    if p.returncode != 0:
        failed.append(out)
for out in failed:
    print(f"\\n--- tail of {out.replace('.csv', '.log')} ---")
    print(open(out.replace(".csv", ".log")).read()[-2000:])
assert not failed, f"shards failed: {failed}"
"""))
    cells.append(code("""
import glob, pandas as pd

parts = [pd.read_csv(f, escapechar="\\\\") for f in sorted(glob.glob("/kaggle/temp/p2_gpu*.csv"))]
preds = pd.concat(parts, ignore_index=True).drop_duplicates(subset="ID")

# Reindex onto the official id list so row order and membership come from the
# CSV, not from whichever shard happened to finish first.
test = pd.read_csv(PHASE2_CSV, escapechar="\\\\")
sub = test[["ID"]].copy()
sub["Target"] = sub.ID.map(dict(zip(preds.ID, preds.Target)))

# Zindi reports a blank Target as a *missing entry* and rejects the file, so no
# row may be left empty -- not one a shard never covered, and not one the model
# transcribed as nothing. "ya" is the most frequent training token (4.4% of all
# tokens) and two characters long, so it scores no worse than the blank.
FILL = "ya"
missing = sub.Target.isna().sum()
if missing:
    print(f"WARNING: {missing} ids had no prediction — filling with {FILL!r}")
sub["Target"] = sub.Target.fillna(FILL)
blank = sub.Target.astype(str).str.strip() == ""
if blank.any():
    print(f"WARNING: {blank.sum()} blank targets — filling with {FILL!r}: "
          f"{', '.join(sub.loc[blank, 'ID'].head(10))}")
    sub.loc[blank, "Target"] = FILL

sub.to_csv(P2_OUT, index=False)
print(f"wrote {P2_OUT}  ({len(sub):,} rows)")
"""))
    cells.append(md("""
### 6b. Is the path or the audio to blame? — **run this first**

First Phase 2 submission: **0.662 error** (leaderboard 0.3378, CER 0.4406, WER
0.8840), against 0.157 on local validation and 0.205 on the Phase 1 test set.

Ruled out already, measured on the real files:

| suspicion | finding |
|---|---|
| Phase 2 clips too long | training averages **20.6 s**, Phase 2 **22.3 s** — 64% of training clips are already ≥18 s |
| codec / channel mismatch | training is 128 kbps mono MP3, near-transparent once resampled |
| clips truncated on load | decoded lengths match the wav headers exactly |

What is left is either the Phase 2 audio being genuinely different, or
`infer_phase2.py` — a code path that had never been run on a GPU before that
submission. This cell separates them: it writes held-out **labeled** clips out as
16 kHz PCM wavs plus an id CSV, the exact shape Phase 2 arrives in, and pushes
them through `infer_phase2.py` as a subprocess.

- **~0.16–0.25** → the path is sound; the Phase 2 audio really is different.
- **~0.60+** → the regression is in the path, and the audio is a red herring.

Restricted to 18–30 s clips so the test set matches Phase 2's range.
"""))
    cells.append(code("""
!python scripts/selftest_phase2.py \\
    --model {P2_MODEL} \\
    --n 200 \\
    --work /kaggle/temp/selftest \\
    --shards 1
"""))
    cells.append(md("""
### 6c. The corpus has 19 languages; we trained on 3

`google/WaxalNLP` carries `ach aka amh dag dga ewe ful kpo lin lug mas mlg nyn
orm sid sna sog tir wal`. Evidence that Phase 2 is not confined to our three:

* 62% of predicted tokens are outside the model's own lin/lug/sna vocabulary,
  and of those, 9.5% are real **Runyankole** words and 10.5% real **Maasai** —
  against under 2.8% for every other language. Runyankole is the Ugandan Bantu
  language closest to Luganda, and our output skewed 63% Luganda.
* WER 0.884 with CER 0.441 is the signature of phonetically-close,
  lexically-wrong output — what a model does on a related language it never saw.
* 6b showed the path scores 0.138 on lin/lug/sna at Phase 2 lengths, so the
  model and the code are both fine.
* The arithmetic closes: at 3-of-19 languages (16%), the unseen remainder would
  need to score 0.761 to produce the 0.662 we saw. Above ~35% in-domain it
  becomes impossible — the rest would need an error over 1.0, which our
  deletion-heavy output cannot reach.

This cell measures the unseen-language error directly. Expect ~0.75–0.85.
"""))
    cells.append(code("""
!python scripts/selftest_phase2.py \\
    --model {P2_MODEL} \\
    --langs nyn mas sog \\
    --n 150 \\
    --work /kaggle/temp/selftest_unseen \\
    --shards 1
"""))

    cells.append(md("""
### 6d. What language is each Phase 2 clip? — **run this before committing to a training plan**

Coverage, not model quality, is the dominant unknown in every plan on the table:
at a fixed error on the languages we cover, going from 99% to 70% coverage costs
~0.20 on the leaderboard. The current mix estimate comes from character n-grams
over our own ASR output, which inherits the model's biases and cannot see
Ethiopic at all. This measures it from the audio.

Language ID is far easier than transcription — languages separate on phonotactics
long before words are recoverable — so a **frozen** w2v-BERT encoder plus a
logistic regression on mean-pooled embeddings is enough. No fine-tuning.

The classifier trains on the corpus itself, so it is in-domain with Phase 2:
same collection, same task, same recording conditions. Its matching weakness is
that it can only answer with one of the 19 languages it was shown, so watch the
confidence distribution — a language outside the corpus can only surface as low
confidence.

Default backend is **MMS-LID** (`facebook/mms-lid-4017`), which is trained for
language ID and emits ISO 639-3 codes — already how the corpus names its
languages. It needs no fitting; the corpus clips are used only to *measure* it.

Measured 2026-07-28: the alternative `--backend probe` (logistic regression on
frozen w2v-BERT embeddings) reached only **52.9%** on held-out speakers, failing
on exactly the related pairs that matter — `sog→lug`, `ewe→kpo`, `dga→ewe`. Not
good enough to plan a training run on.

Accuracy on labeled clips is printed **before** the Phase 2 mix. Below ~85%,
treat the mix as indicative and lean on the per-language accuracies. The report
also shows how often MMS picks a language *outside* the corpus's 19 — the check
the in-domain probe structurally cannot make.

This cell needs `PHASE2_AUDIO` and `PHASE2_CSV` from the section 6 data-prep
cell; run that first or the paths arrive as literal `{...}` strings.
"""))
    cells.append(code("""
!python scripts/lid.py \\
    --langs all \\
    --per-lang 150 \\
    --shards 1 \\
    --backend mms \\
    --predict-dir {PHASE2_AUDIO} \\
    --test-csv {PHASE2_CSV} \\
    --out /kaggle/working/phase2_languages.csv
"""))
    cells.append(md("""
Cross-check against `facebook/mms-lid-4017`, which is trained on entirely
different data. Two independent methods agreeing is far stronger evidence than
either alone — and if they disagree, the in-domain result is the one to doubt,
since it cannot represent a language outside the corpus.
"""))
    cells.append(code("""
import pandas as pd
mix = pd.read_csv("/kaggle/working/phase2_languages.csv")
print(mix.language.value_counts(normalize=True).mul(100).round(1).to_string())
print(f"\\nlow-confidence clips: {(mix.confidence < 0.5).sum()} of {len(mix)}")
# Training hours should follow this table, not our guesses. Feed the top
# languages into --langs / --shards for the run in section 7.
top = mix.language.value_counts()
print("\\nsuggested --langs:", " ".join(top[top >= 0.02*len(mix)].index))
"""))

    cells.append(md("""
### 6e. Does Gemma 4 already know the other languages?

The decision this answers: a model trained on lin/lug/sna caps at **0.359** on the
leaderboard, and we are at 0.338. Fine-tuning Gemma on those three inherits the
same cap. What would break it is Gemma's own pretraining — 140+ languages against
our three. If it can already transcribe `nyn` and `mas`, then fine-tuning it on
the three we have labels for can transfer to the sixteen we don't.

No fine-tuning, no QA data, no new labels. Same clips as 6b/6c, so the numbers
are directly comparable to `w2v-BERT: 0.138 in-domain, 0.585 on nyn`.

- **nyn well under 0.45** → the transfer is real; commit the remaining days to Gemma.
- **nyn near 0.585** → Gemma inherits the ceiling too.

E2B needs ~15 GB in fp16, so `--four-bit` is mandatory on a T4. Budget ~1 hour.
"""))
    cells.append(code("""
# Gemma 4 needs a newer transformers than the one section 1 installs, plus
# bitsandbytes for 4-bit. Restart the session after this if imports misbehave.
!pip install -q -U transformers accelerate bitsandbytes librosa
"""))
    cells.append(code("""
!python scripts/gemma_zeroshot.py \\
    --model google/gemma-4-E2B-it \\
    --langs lin lug sna nyn mas \\
    --n 100 \\
    --work /kaggle/temp/gemma0 \\
    --four-bit
"""))

    cells.append(md("""
### Final check

Every assertion here is one that would otherwise be caught by the leaderboard,
which costs a submission.

The **blank target** check is the one that has actually rejected a submission:
Zindi reports a blank `Target` as `Missing entries for IDs ...` even though the
row is present. Eight clips decoded to nothing on the first Phase 2 run.
"""))
    cells.append(code("""
import pandas as pd, sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")     # is_degenerate lives in pseudo_label.py
from pseudo_label import is_degenerate

sub = pd.read_csv(P2_OUT, escapechar="\\\\")
test = pd.read_csv(PHASE2_CSV, escapechar="\\\\")

assert list(sub.columns) == ["ID", "Target"], sub.columns
assert len(sub) == len(test), f"{len(sub):,} rows vs {len(test):,} in the csv"
assert sub.ID.tolist() == test.ID.tolist(), "ID order must match Test_phase2.csv"
assert sub.ID.duplicated().sum() == 0, "duplicate ids"

text = sub.Target.fillna("")
blank = text.str.strip() == ""
assert not blank.any(), (
    f"{blank.sum()} blank targets — Zindi rejects these as missing entries: "
    f"{sub.loc[blank, 'ID'].head(10).tolist()}")

degen = text.map(is_degenerate).sum()
print(f"submission valid — {len(sub):,} rows, no blank targets")
print(f"  degenerate output: {degen:,} ({100*degen/len(sub):.1f}%)")
print(f"  mean words/clip:   {text.str.split().str.len().mean():.1f}")
# A high empty or degenerate rate means the model failed on this audio, not that
# the submission is malformed. Phase 2 clips are 18-30s; if training data was
# mostly short, expect looping on the long tail.
sub.head(10)
"""))

    cells.append(md("""
## 7. Feature extraction for the A100 run

Phase 2 is East African — measured in 6d and corroborated by the per-language
breakdown of our own submission:

| language | share of Phase 2 | what our model does now |
|---|---|---|
| `lug` (+ `sog`, which MMS cannot label separately) | 43.9% | 24.8 words/clip, only **43.7%** real words |
| `ach` Acholi | 30.4% | **5.8** words/clip, 21.6% real, **12.3% degenerate** |
| `nyn` Runyankole | 16.7% | 25.5 words/clip, 46.4% real |
| `lin` + `sna` | **4.6%** | kept at full size anyway — see below |

The run adds `sog ach nyn` at full size to the `lin lug sna` we already train on,
and keeps **every** language at its full available data. Acholi is the biggest
single win: it is Nilotic, gets no transfer from a Bantu-trained encoder, and
currently emits a fifth of the words that are there.

`lin` and `sna` are only ~4.6% of the measured mix, so they could be cut to a
couple of shards for a much cheaper run (208 h instead of 344 h). They are kept
in full deliberately: the language classifier validated at 70.4%, not 95%, so the
mix is good enough to decide *where to add* data but not good enough to justify
*throwing data away* — and keeping them guards against the model forgetting what
it already does well. The cost is ~65% more compute per epoch.

**Extraction is CPU-bound**, so it runs here for free and the A100 pulls the
result. `--extract-only` downloads the audio for exactly these languages,
extracts features, and stops before touching a GPU.
"""))
    cells.append(code("""
# One definition, used for extraction here and for training on the pod. The cache
# key covers langs/shards/vocab/init-from, so any drift between the two means the
# pod silently re-extracts and the credits are spent anyway.
LANGS  = "ach lin lug nyn sna sog"          # sorted; the script canonicalises it
SHARDS = "0"                                # 0 = every shard, for every language
INIT   = "ngia/ctc-v2-avg"
CACHE  = "/kaggle/temp/features-p2"
REPO   = "ngia/waxal-features-p2"

COMMON = f"--langs {LANGS} --shards {SHARDS} --init-from {INIT} --extend-vocab"
print(COMMON)

# Preflight: the download and the features compete for the same disk, and
# extraction fails at the final flush with a bare "OSError: [Errno 5]" if it runs
# out -- an hour in, with nothing to show.
import shutil, sys
sys.path.insert(0, "src")
from waxal import data as wdata

hrs = 0.0
caps = {}
default = 0
for tok in SHARDS.split():
    if "=" in tok:
        k, v = tok.split("="); caps[k] = int(v)
    else:
        default = int(tok)
for l in LANGS.split():
    avail = (len(wdata.available_shards(l, "train")) +
             len(wdata.available_shards(l, "validation")))
    cap = caps.get(l, default)
    hrs += (avail if cap == 0 else min(avail, cap)) * wdata.HOURS_PER_SHARD

download_gb = hrs * 3600 * 16_000 / 1e9        # 128 kbps mono mp3
features_gb = download_gb * 1.3                # arrow features run a bit larger
free_gb = shutil.disk_usage("/kaggle/temp").free / 1e9
print(f"\\n{hrs:.0f} h of audio across {len(LANGS.split())} languages")
print(f"  download   ~{download_gb:.0f} GB")
print(f"  features   ~{features_gb:.0f} GB")
print(f"  free now    {free_gb:.0f} GB on /kaggle/temp")
if free_gb < download_gb + features_gb:
    print("  --free-download-cache is doing real work here: it deletes the")
    print("  parquet once it is Arrow, before extraction writes its output.")
else:
    print("  comfortable either way")
"""))
    cells.append(code("""
# --free-download-cache deletes the ~12 GB of downloaded parquet once it has been
# converted to Arrow, before extraction writes its own output to the same disk.
!python scripts/train_ctc.py \\
    --extract-only {COMMON} \\
    --output-dir /kaggle/temp/p2-extract \\
    --cache-dir {CACHE} \\
    --free-download-cache \\
    --num-proc 4 --load-proc 1 --valid-frac 0.06 --seed 42
"""))
    cells.append(md("""
Verify before pushing — an upload of a half-written cache is worse than no
upload, because the pod will pull it, fail the key check, and re-extract.
"""))
    cells.append(code("""
import glob, json, os
train = sorted(glob.glob(f"{CACHE}/train_*_of_*.arrow")) or glob.glob(f"{CACHE}/train.arrow")
valid = sorted(glob.glob(f"{CACHE}/valid_*_of_*.arrow")) or glob.glob(f"{CACHE}/valid.arrow")
size = lambda fs: sum(os.path.getsize(f) for f in fs) / 1e9
print(f"train: {len(train)} shard(s), {size(train):.1f} GB")
print(f"valid: {len(valid)} shard(s), {size(valid):.1f} GB")
assert train and valid, "extraction did not finish -- do not push"

man = f"{CACHE}/train.json"
assert os.path.exists(man), "no manifest: the pod cannot validate the cache key"
key = json.load(open(man))
print(f"\\nlangs:  {key.get('langs')}")
print(f"shards: {key.get('shards')}")
print(f"vocab:  {len(key.get('vocab', {}))} symbols")
print("\\nthese three must match on the pod, or it re-extracts")
"""))
    cells.append(md("""
Push the cache. It goes to a **private** dataset repo — the rules forbid sharing
work outside your team, and these features derive from the competition data.
"""))
    cells.append(code("""
!python scripts/sync_features.py push --cache-dir {CACHE} --repo {REPO}
"""))
    cells.append(md("""
### On the A100

Pull, then train. The arguments after `--init-from` must match the extraction
cell **exactly**, or the cache key misses and the pod re-extracts on rented time
— which is the whole thing we are avoiding. The cell below prints the command
built from `COMMON` rather than having you retype it.

`--hub-strategy every_save` pushes each epoch, so a pod that dies mid-run still
leaves a usable checkpoint.
"""))
    cells.append(code("""
POD = '''# on the pod, in the repo root:
pip install -q -U transformers datasets jiwer accelerate soundfile

python scripts/sync_features.py pull --cache-dir /workspace/cache --repo {repo}

python scripts/train_ctc.py \\\\
    {common} \\\\
    --output-dir /workspace/ctc-p2 \\\\
    --cache-dir /workspace/cache \\\\
    --push-to-hub ngia/ctc-p2 --hub-strategy every_save \\\\
    --epochs 3 --batch-size 8 --grad-accum 4 --lr 2e-5 \\\\
    --num-proc 8 --valid-frac 0.06 --seed 42

# ~2.7 h/epoch on an A100 at ~60k rows -> ~8 h for 3 epochs.
# batch 8 / accum 4 holds the effective batch at 32 on an 80GB card;
# use 4 / 8 on a 40GB one.
# If credits get tight, 2 epochs (~5.4 h) from a warm start is a real option --
# or re-run the extraction with --shards "2 lug=0 sog=0 ach=0 nyn=0", which
# caps lin/sna and brings an epoch back down to ~1.6 h.'''
print(POD.format(repo=REPO, common=COMMON))
"""))
    cells.append(md("""
If the pod prints a cache-key mismatch and begins extracting, **stop it** — it is
about to spend an hour of credits redoing this cell's work. It prints both keys;
the differing field is almost always `langs`, `shards` or `vocab`, meaning an
argument drifted between the two machines.
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


def check_referenced_scripts(nb: dict) -> list[str]:
    """Every scripts/X.py a cell invokes must also be written by a cell.

    Missing one produces "No such file or directory" only when that cell runs,
    which in a long notebook can be an hour in. pseudo_label.py and
    sync_features.py were both referenced but not embedded.
    """
    import re

    written = {rel for rel in EMBED}
    referenced = set()
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.startswith("%%writefile"):
            continue
        referenced.update(re.findall(r"(scripts/[a-z_]+\.py)", src))
    return sorted(referenced - written)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    missing = check_referenced_scripts(nb)
    if missing:
        raise SystemExit(
            f"these scripts are invoked by a cell but never written to disk: "
            f"{missing}\nadd them to EMBED")
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}  ({len(nb['cells'])} cells)")
