"""Score the Phase 2 inference path against labeled audio, to localize a regression.

The first Phase 2 submission scored 0.662 error against 0.157 on local
validation and 0.205 on the Phase 1 test set. Two explanations fit: the Phase 2
audio is genuinely harder (different speakers, channel, or language mix), or
`infer_phase2.py` is doing something different from the path that produced those
earlier numbers. They call for opposite responses, so guessing is expensive.

This settles it. Held-out labeled clips are written out as 16 kHz PCM wavs plus
an id CSV -- byte-for-byte the shape Phase 2 arrives in -- and pushed through
`infer_phase2.py` as a subprocess, exactly as the notebook invokes it. The
result is scored against the known transcripts.

    ~0.16-0.25  the path is sound; the Phase 2 audio is the difference
    ~0.60+      the path is the regression, and the audio is a red herring

Note the labeled set is MP3 (128 kbps, 16-48 kHz) while Phase 2 is 16-bit PCM,
so the wav round-trip here also exercises the decode-and-resample step.

    python scripts/selftest_phase2.py --model ngia/ctc-v2-avg --n 200
"""

import argparse
import collections
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from waxal import data as wdata                             # noqa: E402
from waxal.metric import score, score_by_language           # noqa: E402


def write_wav(path: Path, arr: np.ndarray, sr: int) -> None:
    """16-bit mono PCM, matching the Phase 2 clips exactly."""
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def build_labeled_wavs(langs, n, work: Path, shards, valid_frac, seed,
                       min_s, max_s):
    """Write labeled clips out as Phase-2-shaped wavs plus an id CSV.

    Returns (reference DataFrame, csv path, audio dir). Shared with
    gemma_zeroshot.py so both evaluations measure the identical clip set --
    comparing two models on differently-sampled audio would prove nothing.
    """
    audio_dir = Path(work) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ds = wdata.load_labeled(tuple(langs), shards=shards)
    unseen = [l for l in langs if l not in wdata.LANGS]
    if unseen:
        # None of it was in training, so a speaker split would be meaningless
        # theatre -- every clip is already held out.
        valid = ds
        print(f"languages the model never saw: {unseen} -- using all {len(ds):,} clips")
    else:
        valid = wdata.speaker_disjoint_split(ds, valid_frac, seed).valid
        print(f"held-out: {len(valid):,} clips")

    # Round-robin across languages. Taking the first N in dataset order silently
    # returns a single language, because the splits are concatenated per language
    # -- which is how a "nyn mas sog" run once measured 150 clips of pure nyn.
    per_lang = max(1, n // max(len(langs), 1))
    taken = collections.Counter()
    rows = []
    for i in range(len(valid)):
        if len(rows) >= n:
            break
        r = valid[i]
        lang = r["language"]
        if taken[lang] >= per_lang:
            continue
        arr, sr = wdata.audio_array(r["audio"])
        secs = len(arr) / sr
        if not (min_s <= secs <= max_s):
            continue
        uid = str(r["id"])
        write_wav(audio_dir / f"{uid}.wav", arr, sr)
        rows.append({"ID": uid, "transcription": r["transcription"],
                     "language": lang, "secs": secs})
        taken[lang] += 1

    if not rows:
        raise SystemExit(
            f"no held-out clips between {min_s}s and {max_s}s -- "
            f"widen the range or raise --shards")

    ref = pd.DataFrame(rows)
    csv = Path(work) / "Test_selftest.csv"
    ref[["ID"]].assign(Target="").to_csv(csv, index=False)
    print(f"wrote {len(ref):,} wavs to {audio_dir}  "
          f"(mean {ref.secs.mean():.1f}s, {ref.secs.min():.1f}-{ref.secs.max():.1f}s)")
    print(f"  by language: {ref.language.value_counts().to_dict()}")
    return ref, csv, audio_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--n", type=int, default=200, help="clips to score")
    ap.add_argument("--work", type=Path, default=Path("/kaggle/temp/selftest"))
    ap.add_argument("--shards", type=int, default=1,
                    help="labeled parquet files per language (~0.5 GB each). "
                         "With fewer than the full set the speaker split differs "
                         "from training's, so treat the number as indicative")
    ap.add_argument("--langs", nargs="+", default=list(wdata.LANGS),
                    help="which corpus languages to score. The corpus holds 19 "
                         "(ach aka amh dag dga ewe ful kpo lin lug mas mlg nyn "
                         "orm sid sna sog tir wal) and the model was trained on "
                         "three. Pass unseen ones to measure what the model does "
                         "with a language it has never heard -- the Phase 2 score "
                         "is only explicable if the test set contains some")
    ap.add_argument("--valid-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-proc", type=int, default=2)
    ap.add_argument("--min-s", type=float, default=18.0,
                    help="keep only clips at least this long, so the test set "
                         "matches Phase 2's 18-30s range rather than the "
                         "training average")
    ap.add_argument("--max-s", type=float, default=30.0)
    args = ap.parse_args()

    unseen = [l for l in args.langs if l not in wdata.LANGS]
    ref, csv, audio_dir = build_labeled_wavs(
        args.langs, args.n, args.work, args.shards, args.valid_frac, args.seed,
        args.min_s, args.max_s)

    out = args.work / "selftest_pred.csv"
    cmd = [sys.executable, str(Path(__file__).with_name("infer_phase2.py")),
           "--model", str(args.model),
           "--test-csv", str(csv),
           "--audio-dir", str(audio_dir),
           "--out", str(out),
           "--batch-size", str(args.batch_size),
           "--num-proc", str(args.num_proc)]
    if args.vocab:
        cmd += ["--vocab", str(args.vocab)]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("infer_phase2.py failed")

    pred = pd.read_csv(out, escapechar="\\")
    merged = ref.merge(pred, on="ID", how="left")
    hyps = merged.Target.fillna("").astype(str).tolist()
    refs = merged.transcription.astype(str).tolist()

    s = score(refs, hyps)
    print(f"\n=== Phase 2 path on held-out labeled audio ({len(refs):,} clips) ===")
    print(s)
    for lang, v in score_by_language(refs, hyps, merged.language.tolist()).items():
        print(f"  {lang}: corpus {v.combined:.4f}   mean {v.combined_mean:.4f}")

    rw = np.array([len(r.split()) for r in refs])
    hw = np.array([len(h.split()) for h in hyps])
    print(f"\nwords/clip: reference {rw.mean():.1f}  hypothesis {hw.mean():.1f} "
          f"({100 * hw.mean() / max(rw.mean(), 1e-9):.0f}% of reference)")
    print("  Phase 2 submission emitted 17.6 words/clip on 22.3s audio; the "
          "labeled set averages 26.6 words on 20.6s.")

    print("\n--- read this as ---")
    print(f"  combined {s.combined:.4f}")
    if unseen:
        # Phase 2 scored 0.6623 and lin/lug/sna score 0.1384 through this same
        # path, so any two-way mix of those languages with these ones lands
        # between the two. If 0.6623 sits above this measurement, no such mix
        # explains it and the test set must also hold harder languages.
        P2, IN = 0.6623, 0.1384
        x = s.combined
        print(f"  lin/lug/sna score {IN:.3f} through this same path; Phase 2 scored {P2:.3f}")
        if x <= IN:
            print("  this is no harder than the trained languages -- unexpected; re-check")
        elif x >= P2:
            f = (x - P2) / (x - IN)
            print(f"  a set that is {f:.0%} lin/lug/sna and {1-f:.0%} these languages")
            print(f"  would score {P2:.3f}. Compare against 3 of 19 languages = 16%.")
        else:
            print(f"  {P2:.3f} is ABOVE this {x:.3f}, so no mix of these languages with")
            print(f"  lin/lug/sna can produce it -- every mixture lands in [{IN:.3f}, {x:.3f}].")
            print("  Phase 2 must also contain languages harder than these. Likely the")
            print("  distant ones: amh and tir are Ethiopic script, which a Latin CTC")
            print("  alphabet cannot spell at all, so they score near 1.0.")
    elif s.combined < 0.30:
        print("  in line with local validation -> the inference path is sound,")
        print("  and the Phase 2 audio is genuinely different. Look at the data.")
    else:
        print("  far above local validation on audio the model has seen the like")
        print("  of -> the regression is in this code path, not in the Phase 2")
        print("  audio. Diff it against eval_checkpoint.py's feature handling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
