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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataset loading workers; keep low (see train_ctc.py)")
    args = ap.parse_args()

    processor = transformers.Wav2Vec2BertProcessor.from_pretrained(str(args.model))
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

    sub = pd.read_csv(args.sample_submission, escapechar="\\")
    sub["Target"] = sub.ID.map(preds)

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
