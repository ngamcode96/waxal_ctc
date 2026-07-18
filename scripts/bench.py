"""Find out where training time actually goes before renting a bigger GPU.

Observed 106 s/it on a T4 is far off what the arithmetic predicts, so the
bottleneck may not be compute at all. This times the two halves separately:

  * GPU: forward+backward on synthetic tensors, no data pipeline involved.
  * Data: iterating the real dataloader, no model involved.

If GPU time dominates, a faster card fixes it. If data time dominates, a faster
card changes nothing and the fix is in the pipeline.

    python scripts/bench.py --cache-dir /workspace/cache --batch-size 4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets
import torch
import transformers

from waxal import hw

MODEL_ID = "facebook/w2v-bert-2.0"


def bench_gpu(batch_size: int, frames: int, vocab: int, steps: int,
              checkpointing: bool, dtype: torch.dtype) -> float:
    """Seconds per optimizer step, synthetic data, nothing else in the way."""
    dev = torch.device("cuda")
    model = transformers.Wav2Vec2BertForCTC.from_pretrained(
        MODEL_ID, vocab_size=vocab, ctc_loss_reduction="mean",
        add_adapter=True, pad_token_id=vocab - 1,
    ).to(dev)
    if checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    amp = dtype
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)

    x = torch.randn(batch_size, frames, 160, device=dev)
    y = torch.randint(0, vocab - 2, (batch_size, frames // 12), device=dev)

    for i in range(steps + 2):          # first two warm up cudnn/allocator
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp):
            loss = model(input_features=x, labels=y).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def bench_data(cache_dir: Path, batch_size: int, workers: int, steps: int) -> tuple:
    """Seconds per batch pulled through the real dataloader, model excluded."""
    ds = datasets.load_from_disk(str(cache_dir / "train"))
    lengths = ds["length"]

    def collate(feats):
        import numpy as np
        m = max(len(f["input_features"]) for f in feats)
        out = np.zeros((len(feats), m, 160), dtype=np.float32)
        for i, f in enumerate(feats):
            a = np.asarray(f["input_features"], dtype=np.float32)
            out[i, :len(a)] = a
        return torch.from_numpy(out)

    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=collate,
        num_workers=workers, shuffle=True,
    )
    it = iter(dl)
    next(it)                              # warm up workers
    t0 = time.perf_counter()
    n = 0
    for _ in range(steps):
        try:
            next(it)
        except StopIteration:
            break
        n += 1
    return (time.perf_counter() - t0) / max(n, 1), lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=85)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible")
        return
    print(f"GPU: {hw.describe()}\n")

    have_cache = args.cache_dir and (args.cache_dir / "train").exists()
    if args.cache_dir and not have_cache:
        print(f"no feature cache at {args.cache_dir/'train'} -- "
              "running the GPU benchmark only\n")
    if have_cache:
        per_batch, lengths = bench_data(args.cache_dir, args.batch_size,
                                        args.workers, 20)
        import statistics
        print("clip length in frames (100 frames ~= 1s of audio):")
        print(f"  mean {statistics.mean(lengths):.0f}  median "
              f"{statistics.median(lengths):.0f}  "
              f"p95 {sorted(lengths)[int(len(lengths)*0.95)]}  max {max(lengths)}")
        data_step = per_batch * args.grad_accum
        print(f"\ndata only: {per_batch:.3f}s/batch -> {data_step:.1f}s per "
              f"optimizer step ({args.grad_accum} accum)\n")
        frames = int(statistics.mean(lengths))
    else:
        data_step, frames = 0.0, 750
        print("(no --cache-dir: skipping data benchmark)\n")

    total = 0.0
    for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        for ckpt in (True, False):
            try:
                s = bench_gpu(args.batch_size, frames, args.vocab, args.steps,
                              ckpt, dtype)
                total = s * args.grad_accum
                print(f"gpu {label} checkpointing={str(ckpt):5s}: "
                      f"{s:.3f}s/microbatch -> {total:.1f}s per optimizer step")
            except torch.OutOfMemoryError:
                # Expected without checkpointing on a small card; not a failure.
                print(f"gpu {label} checkpointing={str(ckpt):5s}: OOM at "
                      f"batch {args.batch_size} x {frames} frames")
            finally:
                torch.cuda.empty_cache()

    if data_step:
        print(f"\nverdict: data {data_step:.1f}s vs gpu {total:.1f}s per step -> "
              f"{'DATA-BOUND (a faster GPU will not help)' if data_step > total else 'GPU-BOUND (a faster card helps)'}")


if __name__ == "__main__":
    main()
