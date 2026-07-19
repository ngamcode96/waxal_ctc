"""Score a checkpoint on the cached validation split, both metric ways.

The leaderboard and our training log disagree sharply (0.78 vs 0.17). Zindi does
not publish whether it averages per-utterance rates or computes corpus-level
ones, and those diverge a lot when clip lengths vary. This prints both, plus a
per-language and per-length breakdown, so we can tell which number the
leaderboard corresponds to -- without going near the public test labels.

    python scripts/eval_checkpoint.py \
        --model /workspace/ctc-v1/checkpoint-1743 \
        --cache-dir /dev/shm/cache
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import load_processor  # noqa: E402

from waxal.metric import score, score_by_language  # noqa: E402


def load_valid(cache_dir: Path):
    shards = sorted(cache_dir.glob("valid_*_of_*.arrow")) or \
        [p for p in [cache_dir / "valid.arrow"] if p.exists()]
    if not shards:
        raise SystemExit(f"no valid shards in {cache_dir}")
    return datasets.concatenate_datasets(
        [datasets.Dataset.from_file(str(f)) for f in shards])


def build_lm_decoder(processor, lm: Path, unigrams: Path | None,
                     alpha: float, beta: float):
    """A pyctcdecode beam decoder over the same alphabet the model emits.

    pyctcdecode wants the alphabet as a list ordered by token id, with the CTC
    blank as "" and the word delimiter as a literal space.
    """
    from pyctcdecode import build_ctcdecoder

    vocab = processor.tokenizer.get_vocab()
    labels = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    labels = ["" if t == "[PAD]" else " " if t == "|" else t for t in labels]

    words = None
    if unigrams and unigrams.exists():
        words = [w for w in unigrams.read_text().split("\n") if w]

    return build_ctcdecoder(labels, kenlm_model_path=str(lm), unigrams=words,
                            alpha=alpha, beta=beta)


@torch.no_grad()
def run(ds, model, processor, device, batch_size, decoder=None):
    model.eval().to(device)
    hyps = []
    for start in range(0, len(ds), batch_size):
        rows = ds[start:start + batch_size]
        # These are already-extracted features; pad them exactly as the training
        # collator does. Passing them back through the feature extractor would
        # treat mel frames as a waveform and produce nonsense.
        feats = processor.pad(
            [{"input_features": np.asarray(f, dtype=np.float32)}
             for f in rows["input_features"]],
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(**feats).logits
        if decoder is None:
            hyps.extend(processor.batch_decode(logits.argmax(-1).cpu().numpy()))
        else:
            # Beam search needs the full distribution, not the argmax path, and
            # pyctcdecode is float32-only.
            lp = logits.float().cpu().numpy()
            hyps.extend(decoder.decode(lp[i]) for i in range(lp.shape[0]))
        if start % (batch_size * 20) == 0:
            print(f"  {min(start + batch_size, len(ds)):,}/{len(ds):,}", flush=True)
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lm", type=Path, default=None,
                    help="KenLM .arpa from build_lm.py; enables beam decoding")
    ap.add_argument("--unigrams", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="LM weight; higher trusts the language model more")
    ap.add_argument("--beta", type=float, default=1.5,
                    help="word insertion bonus; higher produces longer output")
    args = ap.parse_args()

    import transformers

    processor = load_processor(args.model, args.vocab)
    model = transformers.Wav2Vec2BertForCTC.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_valid(args.cache_dir)
    print(f"validation: {len(ds):,} clips")

    decoder = None
    if args.lm:
        decoder = build_lm_decoder(processor, args.lm, args.unigrams,
                                   args.alpha, args.beta)
        print(f"beam decoding with {args.lm} (alpha={args.alpha}, "
              f"beta={args.beta})")
    else:
        print("greedy decoding (pass --lm to compare against beam search)")

    refs = [processor.tokenizer.decode(ids, group_tokens=False)
            for ids in ds["labels"]]
    hyps = run(ds, model, processor, device, args.batch_size, decoder)

    s = score(refs, hyps)
    print(f"\n{s}\n")
    print("The leaderboard number should match one of those two lines.\n")

    for lang, v in score_by_language(refs, hyps, ds["language"]).items():
        print(f"  {lang}: corpus {v.combined:.4f}   mean {v.combined_mean:.4f}")

    # Short clips are where per-utterance averaging hurts most: one wrong word
    # in a five-word reference is a WER of 0.2 on its own.
    lengths = ds["length"]
    buckets = [(0, 400), (400, 800), (800, 1200), (1200, 10_000)]
    print("\nby clip length (frames, ~50/sec):")
    for lo, hi in buckets:
        idx = [i for i, n in enumerate(lengths) if lo <= n < hi]
        if not idx:
            continue
        b = score([refs[i] for i in idx], [hyps[i] for i in idx])
        print(f"  {lo:>5}-{hi:<5} n={len(idx):>5}  "
              f"corpus {b.combined:.4f}   mean {b.combined_mean:.4f}")

    empty = sum(1 for h in hyps if not h.strip())
    print(f"\nempty hypotheses: {empty}/{len(hyps)}")
    print("\nfirst 3:")
    for r, h in list(zip(refs, hyps))[:3]:
        print(f"  ref: {r[:100]}\n  hyp: {h[:100]}\n")


if __name__ == "__main__":
    main()
