"""Measure raw feature-extraction throughput, one process, no datasets machinery.

Tells you what a single core can actually do. Multiply by --num-proc for the
ceiling; if the real extraction is far below that, the loss is in parallelism
(thread oversubscription, I/O contention), not in the feature extractor.

    python scripts/bench_extract.py
    python scripts/bench_extract.py --threads 16   # show the oversubscription cost
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads per process (1 is right when using "
                         "many worker processes)")
    ap.add_argument("--n", type=int, default=50, help="clips to time")
    ap.add_argument("--seconds", type=float, default=15.0, help="clip duration")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    import numpy as np
    import torch
    import transformers

    torch.set_num_threads(args.threads)

    fe = transformers.AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    audio = np.random.randn(int(16_000 * args.seconds)).astype(np.float32)

    fe(audio, sampling_rate=16_000)          # warm up
    t0 = time.perf_counter()
    for _ in range(args.n):
        fe(audio, sampling_rate=16_000)
    dt = time.perf_counter() - t0

    per = dt / args.n
    print(f"torch threads      : {args.threads}")
    print(f"per clip           : {per*1000:.1f} ms ({args.seconds}s audio)")
    print(f"one process        : {1/per:.1f} examples/s")
    for nproc in (4, 8, 16):
        print(f"  x{nproc:2d} processes    -> {nproc/per:.0f} examples/s ceiling")
    print(f"\n31,316 clips at 8 procs: {31316/(8/per)/60:.1f} min (if it scales)")


if __name__ == "__main__":
    main()
