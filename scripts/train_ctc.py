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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets
import numpy as np
import torch
import transformers

from waxal import data as wdata
from waxal.metric import score, score_by_language
from waxal.normalize import clean

MODEL_ID = "facebook/w2v-bert-2.0"


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
        audio = self.p.pad(
            [{"input_features": f["input_features"]} for f in features],
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


def prepare(ds, processor, num_proc: int):
    def fn(batch):
        arr, sr = wdata.audio_array(batch["audio"])
        feats = processor(arr, sampling_rate=sr).input_features[0]
        return {
            "input_features": feats,
            "labels": processor.tokenizer(clean(batch["transcription"])).input_ids,
            "language": batch["language"],
            "length": len(feats),      # drives LengthGroupedSampler
        }

    return ds.map(fn, remove_columns=ds.column_names, num_proc=num_proc,
                  desc="extracting features")


def prepare_cached(ds, processor, num_proc: int, cache_dir: Path | None, tag: str,
                   key: dict):
    """Feature extraction with an explicit on-disk cache.

    datasets.map caches by fingerprint, but the fingerprint hashes the mapped
    function *and its closure* -- which here includes the processor. Any edit to
    this file, or an unstable hash of the processor, silently invalidates it and
    you pay full extraction again. At ~40 minutes a run that is too expensive to
    leave to chance, so we save explicitly and validate against the parameters
    that actually affect the output.
    """
    if cache_dir is None:
        return prepare(ds, processor, num_proc)

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

    out = prepare(ds, processor, num_proc)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(path))
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
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--valid-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows loaded")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="persist extracted features here so a failed run doesn't "
                         "repeat the ~40min extraction; put it on a persistent volume")
    ap.add_argument("--save-strategy", choices=("steps", "epoch"), default="epoch",
                    help="epoch: one checkpoint per epoch (default)")
    ap.add_argument("--save-steps", type=int, default=500,
                    help="only used with --save-strategy steps")
    ap.add_argument("--push-to-hub", type=str, default="",
                    help="HF repo id, e.g. ngam/waxal-ctc-v1. Checkpoints upload "
                         "as they are saved, surviving pod loss")
    ap.add_argument("--hub-public", action="store_true",
                    help="publish the Hub repo publicly. Off by default: the rules "
                         "forbid sharing work outside your team during the challenge")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in --output-dir")
    args = ap.parse_args()

    transformers.set_seed(args.seed)          # rules require reproducibility

    print("loading labeled data (train+validation only; test is off-limits)")
    ds = wdata.load_labeled(num_proc=args.num_proc)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds):,} rows")

    ds = wdata.filter_usable(ds)
    print(f"  {len(ds):,} usable")

    print("building speaker-disjoint validation split")
    split = wdata.speaker_disjoint_split(ds, args.valid_frac, args.seed)
    print(f"  {split}")

    tokenizer = build_tokenizer(split.train["transcription"], args.output_dir)
    fe = transformers.AutoFeatureExtractor.from_pretrained(MODEL_ID)
    processor = transformers.Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tokenizer)

    # Everything that changes the extracted features or which rows they cover.
    # The learning rate and epoch count deliberately aren't here -- they don't
    # affect features, so re-tuning them should reuse the cache.
    key = {
        "model": MODEL_ID, "langs": list(wdata.LANGS), "sr": wdata.SR,
        "valid_frac": args.valid_frac, "seed": args.seed, "limit": args.limit,
        "vocab": sorted(processor.tokenizer.get_vocab()),
    }
    train_ds = prepare_cached(split.train, processor, args.num_proc,
                              args.cache_dir, "train", key)
    valid_ds = prepare_cached(split.valid, processor, args.num_proc,
                              args.cache_dir, "valid", key)
    valid_langs = valid_ds["language"]
    valid_refs = [clean(t) for t in split.valid["transcription"]]

    model = transformers.Wav2Vec2BertForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
        # SpecAugment-style masking is the main regularizer here; the labeled
        # set is small relative to the 580M encoder.
        mask_time_prob=0.05, mask_time_length=10,
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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
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
