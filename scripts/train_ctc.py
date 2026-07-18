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

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader

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


def prepare(ds, processor, num_proc: int):
    def fn(batch):
        arr, sr = wdata.audio_array(batch["audio"])
        feats = processor(arr, sampling_rate=sr).input_features[0]
        return {
            "input_features": feats,
            "labels": processor.tokenizer(clean(batch["transcription"])).input_ids,
            "language": batch["language"],
        }

    return ds.map(fn, remove_columns=ds.column_names, num_proc=num_proc,
                  desc="extracting features")


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

    train_ds = prepare(split.train, processor, args.num_proc)
    valid_ds = prepare(split.valid, processor, args.num_proc)
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

    targs = transformers.TrainingArguments(
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
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="combined",
        greater_is_better=False,          # lower WER/CER wins
        group_by_length=True,             # big throughput win on variable-length audio
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=[],
    )

    trainer = transformers.Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
    processor.save_pretrained(str(args.output_dir / "best"))
    print(f"saved -> {args.output_dir/'best'}")


if __name__ == "__main__":
    main()
