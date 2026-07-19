"""Run a trained CTC model over evaluation audio and write a Zindi submission.

Phase 1 mode reads the public test audio but never its labels (see
waxal.data.load_test_audio). Phase 2 mode points at whatever directory the
organizers publish.

    python scripts/infer.py --model out/ctc-v1/best --phase 1 --out sub.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
import transformers

from waxal import data as wdata


MODEL_ID = "facebook/w2v-bert-2.0"


def load_processor(model_dir: Path, vocab: Path | None):
    """Load the processor, rebuilding it if the checkpoint lacks one.

    Trainer's intermediate checkpoints hold only model weights -- the processor
    is written by save_pretrained at the end of training. Since the feature
    extractor is just the pretrained one and the tokenizer is fully determined
    by vocab.json (which training writes to --output-dir), we can reconstruct it
    from the checkpoint's parent rather than requiring a finished run.
    """
    try:
        return transformers.Wav2Vec2BertProcessor.from_pretrained(str(model_dir))
    except OSError:
        pass

    candidates = [vocab] if vocab else []
    candidates += [model_dir / "vocab.json", model_dir.parent / "vocab.json"]
    found = next((p for p in candidates if p and p.exists()), None)
    if found is None:
        raise SystemExit(
            f"no processor in {model_dir} and no vocab.json in it or its parent.\n"
            f"Pass --vocab /path/to/vocab.json (training writes it to --output-dir)."
        )

    print(f"checkpoint has no processor; rebuilding from {found}")
    tokenizer = transformers.Wav2Vec2CTCTokenizer(
        str(found), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    fe = transformers.AutoFeatureExtractor.from_pretrained(MODEL_ID)
    return transformers.Wav2Vec2BertProcessor(feature_extractor=fe,
                                              tokenizer=tokenizer)


@torch.no_grad()
def transcribe(ds, model, processor, device, batch_size: int = 8) -> dict[str, str]:
    model.eval().to(device)
    out: dict[str, str] = {}
    for start in range(0, len(ds), batch_size):
        rows = ds[start:start + batch_size]
        arrays = [wdata.audio_array(a)[0] for a in rows["audio"]]
        feats = processor(
            arrays, sampling_rate=wdata.SR, return_tensors="pt", padding=True,
        ).to(device)
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**feats).logits
        hyps = processor.batch_decode(logits.argmax(-1).cpu().numpy())
        out.update(zip(rows["id"], hyps))
        if start % (batch_size * 25) == 0:
            print(f"  {min(start + batch_size, len(ds)):,}/{len(ds):,}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--phase", type=int, choices=(1, 2), default=1)
    ap.add_argument("--phase2-dir", type=str, default="")
    ap.add_argument("--sample-submission", type=Path,
                    default=Path("data/raw/SampleSubmission.csv"))
    ap.add_argument("--vocab", type=Path, default=None,
                    help="vocab.json to rebuild the tokenizer from, if the "
                         "checkpoint has no processor. Defaults to looking in the "
                         "model dir and its parent")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataset loading workers; keep low (see train_ctc.py)")
    args = ap.parse_args()

    processor = load_processor(args.model, args.vocab)
    model = transformers.Wav2Vec2BertForCTC.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.phase == 1:
        ds = wdata.load_test_audio(num_proc=args.num_proc)
    else:
        if not args.phase2_dir:
            ap.error("--phase 2 requires --phase2-dir")
        ds = wdata.load_phase2_audio(args.phase2_dir, num_proc=args.num_proc)
    print(f"transcribing {len(ds):,} clips on {device}")

    preds = transcribe(ds, model, processor, device, args.batch_size)

    if args.sample_submission and args.sample_submission.exists():
        sub = pd.read_csv(args.sample_submission, escapechar="\\")
        sub["Target"] = sub.ID.map(preds)
    else:
        # SampleSubmission.csv is gitignored, so it is often absent on a rented
        # box. The ids we just transcribed are the same set, and Zindi matches
        # on the ID column rather than row order.
        print(f"no sample submission at {args.sample_submission} -- "
              "building from the transcribed ids")
        sub = pd.DataFrame({"ID": list(preds), "Target": list(preds.values())})

    missing = sub.Target.isna().sum()
    if missing:
        # A blank target still scores (badly) -- a missing row may not score at all.
        print(f"WARNING: {missing} ids had no prediction; filling with empty string")
        sub["Target"] = sub.Target.fillna("")

    empty = (sub.Target.str.strip() == "").sum()
    print(f"empty predictions: {empty:,}/{len(sub):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(sub):,} rows)")
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()
