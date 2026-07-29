"""Transcribe the Phase 2 evaluation set and write a Zindi submission.

Separate from infer.py because Phase 2 arrives as a plain directory of wavs plus
a CSV of ids, which the audiofolder loader in `waxal.data.load_phase2_audio`
cannot serve: audiofolder yields an `audio` column and nothing else, so the id
each prediction belongs to is lost. Here the CSV is the authority -- it fixes
both the id set and the row order -- and each wav is read directly from
`<audio-dir>/<ID>.wav`. That also drops the datasets/torchcodec dependency from
the submission path entirely.

Two other differences from Phase 1, both measured on the real data (2026-07-27):

* Clips run 18-30 s (mean 22.3 s) against Phase 1's much shorter and wider
  spread, so batches are formed over length-sorted clips. Padding a 30 s clip
  out with an 18 s one wastes 40% of the batch.
* A clip that fails to transcribe cannot simply be dropped: every id in the CSV
  needs a row. An OOM bisects the batch and retries rather than skipping.

Precision is chosen from the hardware, not from torch.cuda.is_bf16_supported()
-- see waxal.hw.

    python scripts/infer_phase2.py --model ngia/ctc-v2-avg \
        --test-csv data/phase_2/Test_phase2.csv \
        --audio-dir data/phase_2/audio \
        --out submission.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Must precede the torch/numpy imports: they read these at import time. Decoding
# runs in N DataLoader workers, each of which would otherwise start OpenMP with
# one thread per core. Same fix as pseudo_label.py and train_ctc.py.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import torch                                                # noqa: E402

from infer import build_lm_decoder, load_processor          # noqa: E402
from pseudo_label import is_degenerate                      # noqa: E402
from waxal import hw                                        # noqa: E402
from waxal.data import SR                                   # noqa: E402


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 waveform and its sample rate.

    soundfile if it is installed, else the stdlib `wave` module -- the Phase 2
    clips are 16 kHz mono int16 PCM, which `wave` reads natively. Keeping the
    fallback means the submission path has no hard audio dependency.
    """
    try:
        import soundfile as sf
        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise SystemExit(
                    f"{path} is {8 * w.getsampwidth()}-bit; install soundfile "
                    f"to read anything other than 16-bit PCM")
            sr, channels = w.getframerate(), w.getnchannels()
            raw = w.readframes(w.getnframes())
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            arr = arr.reshape(-1, channels)

    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), int(sr)


def wav_seconds(path: Path) -> float:
    """Duration from the header alone, for length sorting.

    Reading 1,500 headers costs milliseconds; decoding 1,500 clips to measure
    them costs minutes, and the decode would then be thrown away.
    """
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        # Not a canonical PCM wav -- fall back to size, which only has to be
        # monotonic in duration for sorting to help.
        return path.stat().st_size / 32000.0


class ClipDataset(torch.utils.data.Dataset):
    """Decode and extract one clip; parallelism comes from DataLoader workers."""

    def __init__(self, items, processor):
        self.items, self.processor = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        uid, path = self.items[i]
        arr, sr = read_wav(path)
        if sr != SR:
            raise SystemExit(
                f"{path} is {sr} Hz but the model expects {SR} Hz. Resample the "
                f"Phase 2 audio before transcribing -- silently feeding the "
                f"wrong rate produces plausible-looking garbage.")
        feats = self.processor(arr, sampling_rate=sr).input_features[0]
        return {"uid": uid, "feats": np.asarray(feats, dtype=np.float32)}


def collate(items):
    return items


@torch.no_grad()
def _logits(batch, model, processor, device, dtype):
    padded = processor.pad([{"input_features": b["feats"]} for b in batch],
                           padding=True, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=device.type == "cuda"):
        return model(**padded).logits


@torch.no_grad()
def run_batch(batch, model, processor, device, dtype, decoder):
    """Transcribe one batch, bisecting on OOM instead of dropping clips.

    Phase 1 could skip a clip that would not fit; here every id in the CSV needs
    a row, and an empty target scores zero for that clip. Halving the batch and
    retrying costs a little time on the rare long batch and loses nothing.
    """
    try:
        logits = _logits(batch, model, processor, device, dtype)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            print(f"  OOM on a single clip ({batch[0]['uid']}) -- retrying on CPU",
                  flush=True)
            cpu = torch.device("cpu")
            model.to(cpu)
            try:
                out = run_batch(batch, model, processor, cpu, dtype, decoder)
            finally:
                model.to(device)
            return out
        mid = len(batch) // 2
        print(f"  OOM on {len(batch)} clips -- splitting", flush=True)
        return (run_batch(batch[:mid], model, processor, device, dtype, decoder)
                + run_batch(batch[mid:], model, processor, device, dtype, decoder))

    if decoder is None:
        hyps = processor.batch_decode(logits.argmax(-1).cpu().numpy())
    else:
        # Beam search needs the full distribution and pyctcdecode is
        # float32-only. This runs on CPU, so it is the slow part.
        lp = logits.float().cpu().numpy()
        hyps = [decoder.decode(lp[i]) for i in range(lp.shape[0])]

    del logits
    return list(zip((b["uid"] for b in batch), hyps))


def transcribe(items, model, processor, device, dtype, batch_size, workers,
               decoder) -> dict[str, str]:
    model.eval().to(device)
    loader = torch.utils.data.DataLoader(
        ClipDataset(items, processor), batch_size=batch_size,
        num_workers=workers, collate_fn=collate, shuffle=False)

    preds: dict[str, str] = {}
    for batch in loader:
        preds.update(run_batch(batch, model, processor, device, dtype, decoder))
        if len(preds) % 200 < batch_size:
            print(f"  {len(preds):,}/{len(items):,}", flush=True)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, default=None,
                    help="vocab.json to rebuild the tokenizer from, if the "
                         "checkpoint carries no processor")
    ap.add_argument("--test-csv", type=Path,
                    default=Path("data/phase_2/Test_phase2.csv"),
                    help="the id list, and the row order of the submission")
    ap.add_argument("--audio-dir", type=Path,
                    default=Path("data/phase_2/audio"),
                    help="directory of <ID>.wav files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="clips per batch. Phase 2 clips are 18-30s, so this is "
                         "lower than Phase 1 would need; an OOM bisects anyway")
    ap.add_argument("--num-proc", type=int, default=4,
                    help="DataLoader workers for decoding. Kaggle has 4 cores")
    ap.add_argument("--shard", type=int, default=0,
                    help="with --num-shards, transcribe only this slice, so two "
                         "GPUs can split the set (see the notebook)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap clips")
    ap.add_argument("--fallback-text", default="ya",
                    help="text for clips the model transcribes as empty. Zindi "
                         "rejects a blank Target as a missing entry, so a blank "
                         "costs the whole submission rather than one clip. The "
                         "default is the most frequent training token (4.4%% of "
                         "all tokens) and is two characters, so it scores no "
                         "worse than the blank it replaces")
    ap.add_argument("--lm", type=Path, default=None,
                    help="KenLM .arpa from build_lm.py; switches greedy decoding "
                         "for beam search. Needs pyctcdecode and kenlm")
    ap.add_argument("--unigrams", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    ap.add_argument("--beta", type=float, default=1.5, help="word insertion bonus")
    args = ap.parse_args()

    if not args.test_csv.exists():
        raise SystemExit(f"no test csv at {args.test_csv}")
    if not args.audio_dir.is_dir():
        raise SystemExit(f"no audio directory at {args.audio_dir}")

    test = pd.read_csv(args.test_csv, escapechar="\\")
    if "ID" not in test.columns:
        raise SystemExit(f"{args.test_csv} has no ID column: {list(test.columns)}")
    ids = test.ID.astype(str).tolist()

    # The zip ships __MACOSX/._<ID>.wav resource forks alongside the real files.
    # Resolving each id through the CSV rather than globbing the directory means
    # they are never picked up, however the archive was extracted.
    paths = {i: args.audio_dir / f"{i}.wav" for i in ids}
    absent = [i for i, p in paths.items() if not p.exists()]
    if absent:
        raise SystemExit(
            f"{len(absent)} of {len(ids)} ids have no wav in {args.audio_dir} "
            f"(first few: {absent[:5]}).\nIf the archive is still zipped: "
            f"unzip -q audio.zip -x '__MACOSX/*' -d <dest>")
    print(f"{len(ids):,} clips in {args.test_csv.name}, all present")

    items = [(i, paths[i]) for i in ids]
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(items):,} clips")
    if args.limit:
        items = items[:args.limit]
        print(f"--limit {args.limit}: {len(items):,} clips")

    # Sort long-to-short so batches hold clips of similar length. Descending
    # puts the biggest batch first, so an OOM surfaces in the first seconds
    # rather than an hour in. Submission order is restored from the CSV below.
    items.sort(key=lambda it: wav_seconds(it[1]), reverse=True)

    import transformers
    processor = load_processor(args.model, args.vocab)
    model = transformers.Wav2Vec2BertForCTC.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if hw.supports_bf16() else torch.float16
    print(f"{hw.describe()} -- autocast {str(dtype).replace('torch.', '')}")

    decoder = None
    if args.lm:
        decoder = build_lm_decoder(processor, args.lm, args.unigrams,
                                   args.alpha, args.beta)
        print(f"beam decoding with {args.lm} "
              f"(alpha={args.alpha}, beta={args.beta})")

    preds = transcribe(items, model, processor, device, dtype,
                       args.batch_size, args.num_proc, decoder)

    # Reindex onto the CSV, so the submission carries every id in its original
    # order whatever order the length-sorted pass produced them in.
    sub = pd.DataFrame({"ID": ids})
    sub["Target"] = sub.ID.map(preds)

    if args.num_shards > 1 or args.limit:
        # A partial run: keep only what this shard produced, for concatenation.
        sub = sub[sub.Target.notna()]
        print(f"partial run -- writing {len(sub):,} rows for merging")
    else:
        missing = sub.Target.isna().sum()
        if missing:
            print(f"WARNING: {missing} ids had no prediction; filling with ''")
            sub["Target"] = sub.Target.fillna("")

    # A blank Target is reported by Zindi as a missing entry and rejects the
    # whole file, so it can never be left in. Measured on the first Phase 2 run:
    # 8 of 1,500 clips decoded to nothing, and their audio was not unusual --
    # RMS at or above the corpus median -- so this is the CTC head collapsing to
    # all-blank, not silent recordings.
    text = sub.Target.fillna("")
    blank = text.str.strip() == ""
    empty = int(blank.sum())
    if empty and args.fallback_text:
        sub.loc[blank, "Target"] = args.fallback_text
        text = sub.Target.fillna("")
        print(f"\n{empty} clip(s) decoded to nothing -- filled with "
              f"{args.fallback_text!r}: "
              f"{', '.join(sub.loc[blank, 'ID'].head(10))}")

    degen = text.map(is_degenerate).sum()
    print(f"\nempty predictions:  {empty:,}/{len(sub):,}")
    print(f"degenerate output:  {degen:,}/{len(sub):,} "
          f"({100 * degen / max(len(sub), 1):.1f}%)")
    print(f"mean words/clip:    {text.str.split().str.len().mean():.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(sub):,} rows)")
    for _, r in sub.head(3).iterrows():
        print(f"  {r.ID}  {str(r.Target)[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
