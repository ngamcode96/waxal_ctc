"""Fine-tune w2v-BERT 2.0 with a CTC head on Lingala / Luganda / Shona.

Why CTC over the starter notebook's Gemma 3n + LoRA:
  * The metric is 50% CER. A generative decoder that hallucinates a fluent wrong
    sentence is catastrophic on both halves; CTC degrades into local character
    noise, which the CER half forgives.
  * Phase 2 provides no language tag. The three languages share <1% of their
    vocabulary, so one joint model over a shared alphabet infers the language
    from acoustics -- no LID stage, no metadata dependency.
  * CTC output can be rescored with an n-gram LM (pyctcdecode) for a cheap
    further gain. A generative model offers no equivalent.

Run:
    python scripts/train_ctc.py --output-dir out/ctc-v1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Must precede the torch/numpy imports: they read these at import time.
#
# Feature extraction runs N worker processes, and each would otherwise start
# torch/OpenMP with one thread per core. With 8 workers on 16 cores that is ~128
# threads competing for 16 cores, and throughput stops responding to --num-proc
# at all (measured: 8.86 ex/s at 16 workers, 8.46 at 8). One thread per worker
# lets the process count actually do the parallelism.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets
import numpy as np
import torch
import transformers

from waxal import data as wdata
from waxal import hw
from waxal.metric import score, score_by_language
from waxal.normalize import clean

MODEL_ID = "facebook/w2v-bert-2.0"
HUB_USER = "ngia"


def build_tokenizer(texts: list[str], out_dir: Path) -> transformers.Wav2Vec2CTCTokenizer:
    chars = sorted({c for t in texts for c in clean(t)})
    # "|" stands in for space so the tokenizer can treat it as a normal symbol.
    vocab = {c if c != " " else "|": i for i, c in enumerate(chars)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)          # doubles as the CTC blank
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1))
    print(f"vocab: {len(vocab)} symbols -> {out_dir/'vocab.json'}")
    return transformers.Wav2Vec2CTCTokenizer(
        str(out_dir / "vocab.json"),
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )


class Collator:
    """Pads audio features and labels independently; masks label padding to -100."""

    def __init__(self, processor):
        self.p = processor

    def __call__(self, features):
        # Features are cached as float16 to halve disk and read bandwidth;
        # restore float32 here so the model and padding logic are unaffected.
        audio = self.p.pad(
            [{"input_features": np.asarray(f["input_features"], dtype=np.float32)}
             for f in features],
            padding=True, return_tensors="pt",
        )
        labels = self.p.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            padding=True, return_tensors="pt",
        )
        # CTC loss ignores -100, so padded label positions contribute nothing.
        audio["labels"] = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        return audio


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Batch clips of similar duration together.

    transformers 5.x dropped `group_by_length`, and without it every batch is
    padded to its longest member -- with clips ranging from 0.5s to 30s that is a
    large fraction of the compute spent on padding. Shuffle, cut into megabatches,
    sort each by length, then shuffle the batch order: near-uniform batches while
    keeping enough randomness that the model doesn't see the same grouping twice.
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 42,
                 megabatch_mult: int = 50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.megabatch_size = batch_size * megabatch_mult
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch      # reshuffles grouping each epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=g).tolist()

        megabatches = [indices[i:i + self.megabatch_size]
                       for i in range(0, len(indices), self.megabatch_size)]
        # Longest-first inside each megabatch surfaces the worst-case batch early,
        # so an OOM shows up in the first minute rather than an hour in.
        megabatches = [sorted(mb, key=lambda i: self.lengths[i], reverse=True)
                       for mb in megabatches]

        batches = [mb[i:i + self.batch_size]
                   for mb in megabatches
                   for i in range(0, len(mb), self.batch_size)]
        order = torch.randperm(len(batches), generator=g).tolist()
        for b in order:
            yield from batches[b]


class LengthGroupedTrainer(transformers.Trainer):
    """Trainer that uses LengthGroupedSampler when the dataset carries lengths."""

    def _get_train_sampler(self, *args, **kwargs):
        ds = self.train_dataset
        if ds is None or "length" not in getattr(ds, "column_names", []):
            return super()._get_train_sampler(*args, **kwargs)
        return LengthGroupedSampler(
            ds["length"], self.args.per_device_train_batch_size, self.args.seed
        )


def make_training_args(**kwargs) -> transformers.TrainingArguments:
    """Build TrainingArguments, dropping keys this transformers version rejects.

    The argument set churns across releases (v5 removed `group_by_length`, v4.46
    renamed `evaluation_strategy` to `eval_strategy`). Kaggle, Colab and RunPod
    will not agree on a version, so filter against the actual signature rather
    than pinning -- and say out loud what got dropped, since a silently ignored
    argument is how you end up wondering why training is slow.
    """
    import inspect

    sig = inspect.signature(transformers.TrainingArguments.__init__)
    accepted = set(sig.parameters)

    # Renames, newest name first: try each until one is accepted.
    aliases = {
        "eval_strategy": ("eval_strategy", "evaluation_strategy"),
        "warmup_ratio": ("warmup_ratio",),
    }
    resolved, dropped = {}, []
    for key, value in kwargs.items():
        for name in aliases.get(key, (key,)):
            if name in accepted:
                resolved[name] = value
                break
        else:
            dropped.append(key)

    if dropped:
        print(f"note: transformers {transformers.__version__} does not accept "
              f"{dropped} -- proceeding without")
        if "group_by_length" in dropped:
            print("      (length grouping off: more padding waste, same accuracy)")
    return transformers.TrainingArguments(**resolved)


def _resample(arr: np.ndarray, factor: float) -> np.ndarray:
    """Speed up (factor>1) or slow down the waveform, shifting pitch with it.

    Resampling without correcting pitch is deliberate -- it is the classic Kaldi
    speed-perturb recipe, and the pitch shift is what makes it act like extra
    speakers rather than just extra tempo. That is what Phase 2 tests.
    """
    if factor == 1.0:
        return arr
    n = int(round(len(arr) / factor))
    # Linear interpolation onto the resampled grid; good enough for augmentation
    # and avoids a scipy/librosa dependency in the hot path.
    idx = np.linspace(0, len(arr) - 1, n)
    return np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)


def prepare(ds, processor, num_proc: int, speeds: tuple[float, ...] = (1.0,),
            writer_batch_size: int = 100):
    """Extract features. Uses the cheaper non-batched path unless augmenting.

    writer_batch_size matters more than it looks: each feature array is ~480KB,
    so the datasets default of 1000 buffers ~480MB per worker before flushing.
    With 16 workers that is ~7.7GB of buffering and very bursty writes -- painful
    on network storage. 100 keeps the flushes small and steady.
    """
    # Belt and braces: the env vars above cover libraries that read them at
    # import, this covers torch regardless of import order. Forked workers
    # inherit it. GPU training does not need CPU threads, so leaving it at 1
    # costs nothing later.
    torch.set_num_threads(1)

    if speeds == (1.0,):
        # No augmentation: one row in, one row out. Avoids the per-row dict
        # wrapping that batched=True with batch_size=1 imposes.
        def fn_single(row):
            arr, sr = wdata.audio_array(row["audio"])
            feats = processor(arr, sampling_rate=sr).input_features[0]
            return {
                # float16 halves the cache (~16GB -> ~8GB) and halves the bytes
                # read per training step. Training runs in bf16 regardless, and
                # the collator restores float32 before the model sees it, so
                # nothing downstream changes.
                "input_features": np.asarray(feats, dtype=np.float16),
                "labels": processor.tokenizer(clean(row["transcription"])).input_ids,
                "language": row["language"],
                "length": len(feats),
            }

        return ds.map(fn_single, remove_columns=ds.column_names, num_proc=num_proc,
                      writer_batch_size=writer_batch_size,
                      desc="extracting features")

    def fn(batch):
        # batched=True with batch_size=1: every field arrives as a 1-element list.
        arr, sr = wdata.audio_array(batch["audio"][0])
        labels = processor.tokenizer(clean(batch["transcription"][0])).input_ids
        feats, lens = [], []
        for f in speeds:
            x = processor(_resample(arr, f), sampling_rate=sr).input_features[0]
            feats.append(x)
            lens.append(len(x))
        return {
            "input_features": feats,
            "labels": [labels] * len(speeds),      # transcript is unchanged
            "language": [batch["language"][0]] * len(speeds),
            "length": lens,                        # drives LengthGroupedSampler
        }

    # batched with a 1-row batch: lets each input emit len(speeds) output rows.
    return ds.map(fn, remove_columns=ds.column_names, num_proc=num_proc,
                  batched=True, batch_size=1,
                  writer_batch_size=writer_batch_size,
                  desc=f"extracting features (speeds={list(speeds)})")


# ~750 frames x 160 mels x 2 bytes (float16) for a typical 15s clip.
BYTES_PER_ROW = 750 * 160 * 2


def check_disk_space(rows: int, cache_dir: Path) -> None:
    """Fail now rather than at writer.finalize() ten minutes in.

    datasets writes the map output under HF_DATASETS_CACHE and save_to_disk then
    copies it into cache_dir, so the peak is roughly twice the final size.
    """
    import shutil

    need = rows * BYTES_PER_ROW * 2
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    print(f"disk: {free/1e9:.1f}GB free, ~{need/1e9:.1f}GB needed "
          f"for {rows:,} rows (map output + cache copy)")
    if free < need:
        raise SystemExit(
            f"\nNot enough space: {free/1e9:.1f}GB free, ~{need/1e9:.1f}GB needed.\n"
            f"  - free the downloaded parquet with --free-download-cache (~13GB)\n"
            f"  - or point --cache-dir at a larger disk\n"
            f"  - or provision a bigger volume\n"
            f"Extraction would otherwise fail at the final flush with "
            f"'OSError: [Errno 5] Input/output error'."
        )


def prepare_cached(ds, processor, num_proc: int, cache_dir: Path | None, tag: str,
                   key: dict, speeds: tuple[float, ...] = (1.0,)):
    """Feature extraction with an explicit on-disk cache.

    datasets.map caches by fingerprint, but the fingerprint hashes the mapped
    function *and its closure* -- which here includes the processor. Any edit to
    this file, or an unstable hash of the processor, silently invalidates it and
    you pay full extraction again. At ~40 minutes a run that is too expensive to
    leave to chance, so we save explicitly and validate against the parameters
    that actually affect the output.
    """
    if cache_dir is None:
        return prepare(ds, processor, num_proc, speeds)

    check_disk_space(len(ds) * len(speeds), cache_dir)

    path = cache_dir / tag
    manifest = cache_dir / f"{tag}.json"
    if path.exists() and manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored == key:
            print(f"reusing cached features: {path}")
            return datasets.load_from_disk(str(path))
        # Stale cache is worse than none -- it trains on the wrong thing silently.
        print(f"cache at {path} was built with different settings, rebuilding")
        print(f"  cached: {stored}\n  wanted: {key}")

    out = prepare(ds, processor, num_proc, speeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Default sharding writes ~21 files for this dataset. Networked filesystems
    # (RunPod's MooseFS) fail with Errno 5 partway through that; fewer, larger
    # writes are more likely to survive. Local disks do not care either way.
    out.save_to_disk(str(path), max_shard_size="2GB")
    manifest.write_text(json.dumps(key, indent=1, sort_keys=True))
    print(f"cached features -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-proc", type=int, default=8,
                    help="workers for feature extraction (CPU-bound; use all cores)")
    ap.add_argument("--gradient-checkpointing", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="default: off when the GPU has >40GB, on otherwise")
    ap.add_argument("--free-download-cache", action="store_true",
                    help="delete the downloaded parquet (~13GB) once it has been "
                         "converted to Arrow, before extraction writes its output. "
                         "Re-downloads if you need it again")
    ap.add_argument("--load-proc", type=int, default=1,
                    help="workers for dataset download/Arrow generation. Keep low: "
                         "it is I/O-bound, and high values make datasets' forked "
                         "workers fail with 'I/O operation on closed file'")
    ap.add_argument("--valid-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows loaded")
    ap.add_argument("--max-s", type=float, default=30.0,
                    help="drop clips longer than this. Attention is quadratic in "
                         "length, so a few long clips cost disproportionately")
    ap.add_argument("--min-s", type=float, default=0.5)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="persist extracted features here so a failed run doesn't "
                         "repeat the ~40min extraction; put it on a persistent volume")
    ap.add_argument("--save-strategy", choices=("steps", "epoch"), default="epoch",
                    help="epoch: one checkpoint per epoch (default)")
    ap.add_argument("--save-steps", type=int, default=500,
                    help="only used with --save-strategy steps")
    ap.add_argument("--push-to-hub", type=str, nargs="?", const="auto", default="",
                    help=f"upload checkpoints as they are saved, surviving pod "
                         f"loss. Bare flag uses {HUB_USER}/<output-dir-name>; pass "
                         f"a value to override, e.g. {HUB_USER}/waxal-ctc-v1")
    ap.add_argument("--hub-public", action="store_true",
                    help="publish the Hub repo publicly. Off by default: the rules "
                         "forbid sharing work outside your team during the challenge")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in --output-dir")
    # SpecAugment. Applied inside the model at training time only, so it costs
    # nothing extra and does NOT invalidate the feature cache -- tune these freely.
    ap.add_argument("--mask-time-prob", type=float, default=0.05,
                    help="fraction of time steps masked (0.05-0.10 typical)")
    ap.add_argument("--mask-time-length", type=int, default=10)
    ap.add_argument("--mask-feature-prob", type=float, default=0.0,
                    help="frequency masking; 0.01-0.05 helps on unseen speakers")
    ap.add_argument("--mask-feature-length", type=int, default=64)
    # Speed perturbation. Changes the audio, so it DOES invalidate the cache and
    # multiplies extraction time and disk by len(factors).
    ap.add_argument("--speed-perturb", type=str, default="",
                    help="comma-separated factors, e.g. 0.9,1.0,1.1 -- triples the "
                         "training set with faster/slower copies")
    args = ap.parse_args()

    transformers.set_seed(args.seed)          # rules require reproducibility

    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = hw.wants_gradient_checkpointing()
    print(f"{hw.describe()}  vram={hw.vram_gb():.0f}GB  "
          f"gradient_checkpointing={args.gradient_checkpointing}")

    if args.push_to_hub == "auto":
        args.push_to_hub = f"{HUB_USER}/{args.output_dir.name}"
    if args.push_to_hub:
        vis = "public" if args.hub_public else "private"
        print(f"checkpoints -> huggingface.co/{args.push_to_hub} ({vis})")

    print("loading labeled data (train+validation only; test is off-limits)")
    # With --limit, fetch one parquet shard per language instead of all ~12.6GB.
    ds = wdata.load_labeled(num_proc=args.load_proc, shards=1 if args.limit else 0)
    if args.limit:
        # Shuffle first: the shards concatenate language by language, so taking
        # the head would give an all-Lingala "smoke test" that never exercises
        # the other two languages or the joint vocabulary.
        ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))
    print(f"  {len(ds):,} rows")

    ds = wdata.filter_usable(ds, min_s=args.min_s, max_s=args.max_s)
    print(f"  {len(ds):,} usable (clips {args.min_s}-{args.max_s}s)")

    if args.free_download_cache:
        # load_dataset has already materialized everything into Arrow, so the
        # downloaded parquet is dead weight -- and it is ~13GB competing for the
        # same disk that extraction is about to write to.
        import shutil
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        repo = hub / f"datasets--{wdata.HF_REPO.replace('/', '--')}"
        if repo.exists():
            freed = sum(f.stat().st_size for f in repo.rglob("*") if f.is_file())
            shutil.rmtree(repo, ignore_errors=True)
            print(f"freed {freed/1e9:.1f}GB of downloaded parquet ({repo})")
        else:
            print(f"nothing to free at {repo}")

    print("building speaker-disjoint validation split")
    split = wdata.speaker_disjoint_split(ds, args.valid_frac, args.seed)
    print(f"  {split}")

    tokenizer = build_tokenizer(split.train["transcription"], args.output_dir)
    fe = transformers.AutoFeatureExtractor.from_pretrained(MODEL_ID)
    processor = transformers.Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tokenizer)

    # Everything that changes the extracted features or which rows they cover.
    # The learning rate and epoch count deliberately aren't here -- they don't
    # affect features, so re-tuning them should reuse the cache.
    speeds = tuple(float(s) for s in args.speed_perturb.split(",")) \
        if args.speed_perturb else (1.0,)

    key = {
        "model": MODEL_ID, "langs": list(wdata.LANGS), "sr": wdata.SR,
        "valid_frac": args.valid_frac, "seed": args.seed, "limit": args.limit,
        "min_s": args.min_s, "max_s": args.max_s,   # change which rows survive
        "vocab": sorted(processor.tokenizer.get_vocab()),
    }
    # Augment training only. A perturbed validation set would measure the model
    # on audio the Phase 2 set will never contain, and would not be comparable
    # to the un-augmented baseline.
    train_ds = prepare_cached(split.train, processor, args.num_proc,
                              args.cache_dir, "train",
                              {**key, "speeds": list(speeds)}, speeds)
    valid_ds = prepare_cached(split.valid, processor, args.num_proc,
                              args.cache_dir, "valid", key)
    if speeds != (1.0,):
        print(f"speed perturbation {list(speeds)}: train {len(split.train):,} "
              f"-> {len(train_ds):,} rows")
    valid_langs = valid_ds["language"]
    valid_refs = [clean(t) for t in split.valid["transcription"]]

    model = transformers.Wav2Vec2BertForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
        # SpecAugment-style masking is the main regularizer here; the labeled
        # set is small relative to the 580M encoder.
        mask_time_prob=args.mask_time_prob,
        mask_time_length=args.mask_time_length,
        mask_feature_prob=args.mask_feature_prob,
        mask_feature_length=args.mask_feature_length,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    def compute_metrics(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        hyps = processor.batch_decode(ids)
        s = score(valid_refs, hyps)
        per = score_by_language(valid_refs, hyps, valid_langs)
        out = {"wer": s.wer, "cer": s.cer, "combined": s.combined}
        out.update({f"combined_{l}": v.combined for l, v in per.items()})
        return out

    targs = make_training_args(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        # NOT torch.cuda.is_bf16_supported(): that reports True on a T4, where
        # bf16 is emulated and much slower than the card's fp16 tensor cores.
        bf16=hw.supports_bf16(),
        fp16=not hw.supports_bf16(),
        # Checkpointing trades ~35% throughput for memory. It is mandatory on a
        # 16GB T4 (measured: OOM without it at batch 4) and pure waste on an 80GB
        # A100, so decide from the card rather than hardcoding the small-GPU case.
        gradient_checkpointing=args.gradient_checkpointing,
        # Evaluate and save on the same cadence so load_best_model_at_end always
        # has a metric for every checkpoint it might pick.
        eval_strategy=args.save_strategy,
        save_strategy=args.save_strategy,
        **({"eval_steps": args.save_steps, "save_steps": args.save_steps}
           if args.save_strategy == "steps" else {}),
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="combined",
        greater_is_better=False,          # lower WER/CER wins
        # We group by length ourselves (LengthGroupedTrainer); keep the extra
        # columns it needs, since the collator already selects what the model sees.
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=[],
        # "every_save" uploads each checkpoint as it is written, so a pod that
        # dies mid-run costs one epoch rather than the whole run.
        **({"push_to_hub": True,
            "hub_model_id": args.push_to_hub,
            "hub_strategy": "every_save",
            "hub_private_repo": not args.hub_public} if args.push_to_hub else {}),
    )

    trainer = LengthGroupedTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
    )
    resume = None
    if args.resume:
        last = transformers.trainer_utils.get_last_checkpoint(str(args.output_dir))
        if last:
            print(f"resuming from {last}")
            resume = last
        else:
            print(f"--resume given but no checkpoint in {args.output_dir}; "
                  "starting fresh")

    trainer.train(resume_from_checkpoint=resume)

    best = args.output_dir / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    print(f"saved -> {best}")

    if args.push_to_hub:
        # The per-checkpoint uploads carry model weights but not the processor;
        # without it the repo can't tokenize or extract features on its own.
        processor.push_to_hub(args.push_to_hub, private=not args.hub_public)
        trainer.push_to_hub(commit_message="final best checkpoint")
        print(f"pushed -> https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
