"""Train an n-gram language model over the training transcripts, for CTC beam decoding.

Greedy CTC decoding picks the best symbol per frame independently, which is how
you get output like "Muta a a a a" -- a sequence with no linguistic support at
all (zero such patterns exist in the 38k training transcripts). A language model
scores whole word sequences during beam search and rejects those paths.

This should help Shona most: its words average 7.2 characters against Lingala's
4.5, so a single character slip destroys a whole word. WER is half the metric,
so recovering those words is worth more there than anywhere else.

The corpus comes from the *cached training labels*, decoded back to text -- not
from the validation split, which would leak, and not from Train.csv, which is
gitignored and absent on a rented box.

One language model covers all three languages: they share under 1% of their
vocabulary, so words rarely collide, and Phase 2 gives no language tag to route
on anyway.

    python scripts/build_lm.py --cache-dir /dev/shm/cache --out /workspace/lm
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets


def resolve_vocab(cache_dir: Path, vocab_path: Path | None) -> Path:
    """Find a vocab.json, falling back to the one inside the cache manifest.

    The manifest stores the exact token->id map the cached labels were written
    against, so the cache is self-sufficient -- no training run required.
    """
    if vocab_path and vocab_path.exists():
        return vocab_path

    manifest = cache_dir / "train.json"
    if not manifest.exists():
        raise SystemExit(
            f"no vocab.json at {vocab_path} and no manifest at {manifest}")

    vocab = json.loads(manifest.read_text()).get("vocab")
    if not isinstance(vocab, dict):
        raise SystemExit(
            f"{manifest} has no token->id map (old-format cache). "
            f"Pass --vocab explicitly.")

    out = cache_dir / "vocab.json"
    out.write_text(json.dumps(vocab, ensure_ascii=False, indent=1))
    print(f"vocab from cache manifest -> {out} ({len(vocab)} symbols)")
    return out


def corpus_from_cache(cache_dir: Path, vocab_path: Path) -> list[str]:
    """Decode the cached training labels back into transcripts."""
    import transformers

    tok = transformers.Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|")

    shards = sorted(cache_dir.glob("train_*_of_*.arrow")) or \
        [p for p in [cache_dir / "train.arrow"] if p.exists()]
    if not shards:
        raise SystemExit(f"no train shards in {cache_dir}")

    ds = datasets.concatenate_datasets(
        [datasets.Dataset.from_file(str(f)) for f in shards])
    print(f"decoding {len(ds):,} training transcripts")
    return [tok.decode(ids, group_tokens=False) for ids in ds["labels"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, default=None,
                    help="vocab.json from training; defaults to the "
                         "one stored in the cache manifest")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--order", type=int, default=5, help="n-gram order")
    ap.add_argument("--lmplz", default="lmplz",
                    help="path to KenLM's lmplz binary")
    ap.add_argument("--memory", default="4G",
                    help="lmplz sort buffer (-S). Its default of 80%% of physical "
                         "RAM gets OOM-killed inside a container")
    ap.add_argument("--temp-dir", type=Path, default=Path("/tmp/lmplz"),
                    help="lmplz scratch space (-T)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    vocab_path = resolve_vocab(args.cache_dir, args.vocab)
    texts = corpus_from_cache(args.cache_dir, vocab_path)

    # Keep the original casing. Tempting to lowercase -- it would nearly halve
    # the type count and sharpen the estimates -- but pyctcdecode scores the
    # words the acoustic model actually emits, and our alphabet is cased. A
    # lowercase LM would fail to match "Ndaba" and silently score every
    # sentence-initial word as unknown.
    words = [w for t in texts for w in t.split()]
    counts = Counter(words)
    print(f"corpus: {len(texts):,} sentences, {len(words):,} tokens, "
          f"{len(counts):,} types")
    print(f"  hapax: {sum(1 for w, n in counts.items() if n == 1):,} "
          f"({100*sum(1 for w, n in counts.items() if n == 1)/len(counts):.1f}%)")

    corpus = args.out / "corpus.txt"
    corpus.write_text("\n".join(texts) + "\n")
    print(f"corpus -> {corpus}")

    unigrams = args.out / "unigrams.txt"
    unigrams.write_text("\n".join(sorted(counts)) + "\n")
    print(f"unigrams -> {unigrams} ({len(counts):,})")

    if shutil.which(args.lmplz) is None:
        print(f"\n'{args.lmplz}' not found -- build KenLM to train the ARPA:\n"
              "  apt-get update && apt-get install -y cmake libboost-all-dev "
              "libeigen3-dev zlib1g-dev\n"
              "  git clone https://github.com/kpu/kenlm /opt/kenlm\n"
              "  cmake -S /opt/kenlm -B /opt/kenlm/build && "
              "make -C /opt/kenlm/build -j8\n"
              f"  python scripts/build_lm.py ... --lmplz /opt/kenlm/build/bin/lmplz")
        return 1

    arpa = args.out / f"{args.order}gram.arpa"
    # prune: keep all unigrams/bigrams, drop singleton 3+ grams. With 65% hapax
    # words, unpruned high-order counts are mostly noise and bloat the model.
    prune = ["0", "0"] + ["1"] * (args.order - 2)
    # -S caps the sort buffer. lmplz otherwise claims 80% of *physical* RAM,
    # which inside a container is the host's, not the cgroup limit -- it then
    # gets OOM-killed (SIGKILL) partway through counting. This corpus is ~6MB,
    # so a few GB is generous.
    # -T keeps temp files off /tmp, which is on the small container disk.
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.lmplz, "-o", str(args.order), "--prune", *prune,
           "--discount_fallback",
           "-S", args.memory, "-T", str(args.temp_dir)]
    print(f"\n$ {' '.join(cmd)} < {corpus} > {arpa}")
    with corpus.open() as fin, arpa.open("w") as fout:
        subprocess.run(cmd, stdin=fin, stdout=fout, check=True)

    size = arpa.stat().st_size
    print(f"\narpa -> {arpa} ({size/1e6:.1f}MB)")
    print("\nuse it with:")
    print(f"  python scripts/eval_checkpoint.py --model ... --cache-dir "
          f"{args.cache_dir} --lm {arpa} --unigrams {unigrams}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
