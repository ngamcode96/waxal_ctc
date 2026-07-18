# WAXAL ASR Challenge

Joint w2v-BERT 2.0 CTC model over Lingala (`lin`), Luganda (`lug`) and Shona (`sna`)
for the [Google WAXAL ASR Challenge](https://zindi.africa/competitions/google-waxal-asr-challenge).

## Setup

Laptop (analysis only — no CUDA wheels):

```bash
uv venv --python 3.11
uv pip install -e .
```

RunPod / any GPU box:

```bash
uv venv --python 3.11
uv pip install -e ".[train]"
```

## Training

```bash
export HF_TOKEN=hf_...        # needs write permission
python scripts/train_ctc.py \
    --output-dir /workspace/ctc-v1 \
    --cache-dir /workspace/cache \
    --push-to-hub \
    --epochs 5 --batch-size 16 --grad-accum 2 --lr 5e-5 --num-proc 16
```

The bare `--push-to-hub` flag uploads each checkpoint to
`ngia/<output-dir-name>` — so the command above pushes to `ngia/ctc-v1`, and
naming the output dir after the experiment keeps runs separate. Repos are
created **private**; the rules forbid sharing work outside your team during the
challenge. Pass a value to override the repo, or `--hub-public` to publish.

`--cache-dir` persists the ~40 minute feature extraction across runs. Put it on a
volume that survives the pod. Add `--resume` to continue from the last checkpoint.

## Scripts

| script | purpose |
|---|---|
| `scripts/profile_data.py` | dataset shape, language split, charset |
| `scripts/normalization_cost.py` | what each text-normalization choice costs against the metric |
| `scripts/bench.py` | is training GPU-bound or data-bound? |
| `scripts/train_ctc.py` | fine-tune w2v-BERT 2.0 with a CTC head |
| `scripts/infer.py` | write a Zindi submission |
| `scripts/build_kaggle_notebook.py` | regenerate `notebooks/waxal_kaggle.ipynb` from this source |

## Things that will bite you

- **The Zindi CSVs are backslash-escaped, not CSV-standard.** Read them with
  `pd.read_csv(..., escapechar="\\")` or 23 training rows silently shred.
- **Phase 1 test labels are public** (the competition test split is the public HF
  `test` split). Using them is a rules breach; `waxal.data` refuses to load that
  split. Only Phase 2 decides the final ranking, so Phase 1 leaderboard position
  carries no signal — watch the speaker-disjoint validation score instead.
- **Phase 2 ships no metadata**, including no language tag. Hence one joint model
  over a shared alphabet rather than per-language models.
- **Don't pick precision with `torch.cuda.is_bf16_supported()`.** It returns True
  on a T4, where bf16 is emulated and measured 3.6x slower than fp16. See
  `src/waxal/hw.py`.
- **Train on raw cased, punctuated text.** Emitting normalized text costs 0.10
  combined against the references even with perfect acoustics
  (`scripts/normalization_cost.py`).

## Editing the Kaggle notebook

Don't. Edit the source here and regenerate:

```bash
python scripts/build_kaggle_notebook.py
```

The notebook embeds this source as `%%writefile` cells because the rules reject
custom packages in submission notebooks.
