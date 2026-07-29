"""Does Gemma 4 already know the languages our CTC model has never heard?

This is the one measurement that decides whether a Gemma-based plan can beat the
ceiling. Phase 2 is at most ~35% lin/lug/sna, so any model trained on those three
caps at 0.359 on the leaderboard however good it gets -- we are already at 0.338.
Fine-tuning Gemma on three languages inherits that same cap.

What would break it is Gemma's own pretraining: 140+ languages, against our
model's three. If it has real competence in nyn, mas, sog and the rest, then
fine-tuning it on the three we have labels for can transfer to the sixteen we do
not, which a CTC model with a fixed Latin alphabet structurally cannot do.

No fine-tuning, no QA data, no labels beyond what the corpus already ships. It
scores zero-shot Gemma on the identical clip set selftest_phase2.py builds, so
the numbers are directly comparable to the w2v-BERT ones:

    w2v-BERT on held-out lin/lug/sna   0.1384
    w2v-BERT on unseen nyn             0.5845
    w2v-BERT on Phase 2                0.6623

Read the result as: if Gemma's nyn error is far below 0.585, its multilingual
pretraining is real and the plan has a path. If it is comparable or worse, Gemma
inherits the ceiling and the QA + transcription work buys ~0.02.

    python scripts/gemma_zeroshot.py --langs lin lug sna nyn mas --n 100
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import torch                                                # noqa: E402

from selftest_phase2 import build_labeled_wavs              # noqa: E402
from waxal import data as wdata                             # noqa: E402
from waxal.metric import score, score_by_language           # noqa: E402

# Measured with scripts/selftest_phase2.py on 2026-07-27/28, same clip shape.
BASELINE = {"in_domain": 0.1384, "nyn": 0.5845, "phase2": 0.6623}

# The documented ASR prompt. Google's own example transcribes without naming the
# language and lets the model detect it, which is the only form usable here:
# Phase 2 ships no language tag, so a prompt that needs one cannot be deployed.
PROMPT_AGNOSTIC = ("Transcribe the following speech segment in its original "
                   "language. Output only the transcription, with no commentary, "
                   "no translation and no newlines.")
PROMPT_NAMED = ("Transcribe the following speech segment in {lang} into {lang} "
                "text. Output only the transcription, with no commentary and no "
                "newlines.")

LANG_NAMES = {
    "ach": "Acholi", "aka": "Akan", "amh": "Amharic", "dag": "Dagbani",
    "dga": "Dagaare", "ewe": "Ewe", "ful": "Fulfulde", "kpo": "Kabiye",
    "lin": "Lingala", "lug": "Luganda", "mas": "Maasai", "mlg": "Malagasy",
    "nyn": "Runyankole", "orm": "Oromo", "sid": "Sidamo", "sna": "Shona",
    "sog": "Lusoga", "tir": "Tigrinya", "wal": "Wolaytta",
}


_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is)[^:\n]*|the\s+transcription[^:\n]*|"
    r"sure[^:\n]*|okay[^:\n]*|transcription)\s*:\s*", re.I)


def strip_preamble(text: str) -> str:
    """Drop conversational scaffolding an instruction-tuned model adds anyway.

    The prompt asks for the transcription alone, but instruction tuning is
    persistent: "Here is the transcription: ..." scores every one of those words
    as an insertion. Only a leading label is removed, and only up to the first
    colon -- anything more aggressive would start editing the transcript itself.
    Also unwraps a fully-quoted answer and keeps the first paragraph, since
    trailing commentary ("This appears to be Luganda...") is pure insertion.
    """
    t = " ".join(str(text).split())
    for _ in range(2):                    # at most two stacked labels
        t2 = _PREAMBLE.sub("", t, count=1).strip()
        if t2 == t:
            break
        t = t2
    if len(t) > 1 and t[0] in "\"'“" and t[-1] in "\"'”":
        t = t[1:-1].strip()
    return t


def load_gemma(model_id: str, four_bit: bool):
    """Gemma 4 with audio input.

    AutoModelForMultimodalLM is what the model card uses; older transformers
    releases do not have it, so the failure is reported as a version problem
    rather than an opaque attribute error.
    """
    import transformers

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as e:
        raise SystemExit(
            f"this transformers ({transformers.__version__}) has no "
            f"AutoModelForMultimodalLM ({e}).\n"
            f"    pip install -U transformers accelerate librosa") from e

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    kwargs = {"device_map": "auto", "dtype": "auto"}
    if four_bit:
        # E2B needs ~15 GB in fp16 and a Kaggle T4 has 16 GB, so anything other
        # than 4-bit leaves no room for the audio encoder's activations.
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs.pop("dtype")
        print("loading in 4-bit (nf4, fp16 compute)")
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
    return processor, model.eval()


@torch.no_grad()
def transcribe(paths, langs, processor, model, prompt_style: str,
               max_new_tokens: int) -> list[str]:
    """One clip per call. Gemma's audio path is not batched here on purpose:
    the point is a clean measurement, not throughput, and 30 s clips at 25
    tokens/s already fill a good part of the context."""
    out = []
    t0 = time.time()
    for i, (p, lang) in enumerate(zip(paths, langs)):
        text = (PROMPT_AGNOSTIC if prompt_style == "agnostic"
                else PROMPT_NAMED.format(lang=LANG_NAMES.get(lang, lang)))
        messages = [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "audio", "audio": str(p)},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True).to(model.device)
        n_in = inputs["input_ids"].shape[-1]
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
        # The model card decodes WITH special tokens and then calls
        # parse_response, which strips whatever scaffolding the chat template
        # wraps the answer in. Decoding straight to text instead scores that
        # scaffolding as if it were transcription -- which is how a first run
        # produced WER 4.33, four times the reference length.
        raw = processor.decode(gen[0][n_in:], skip_special_tokens=False)
        hyp = raw
        if hasattr(processor, "parse_response"):
            try:
                parsed = processor.parse_response(raw, prefix=inputs["input_ids"])
                hyp = parsed if isinstance(parsed, str) else (
                    parsed.get("text") or parsed.get("content") or str(parsed))
            except Exception as e:
                if i == 0:
                    print(f"  parse_response failed ({type(e).__name__}: {e}); "
                          f"falling back to plain decode + strip", flush=True)
                hyp = processor.decode(gen[0][n_in:], skip_special_tokens=True)
        out.append(strip_preamble(hyp))
        if i < 2:
            print(f"  [sample {i}] {out[-1][:160]}", flush=True)
        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(paths)}  ({rate:.2f} clips/s, "
                  f"{(len(paths)-i-1)/max(rate,1e-9)/60:.0f} min left)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--langs", nargs="+",
                    default=["lin", "lug", "sna", "nyn", "mas"],
                    help="mix trained and untrained languages -- the comparison "
                         "between them is the whole point")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--work", type=Path, default=Path("/kaggle/temp/gemma0"))
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--valid-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-s", type=float, default=18.0)
    ap.add_argument("--max-s", type=float, default=30.0,
                    help="Gemma 4 accepts at most 30 s of audio")
    ap.add_argument("--prompt", choices=("agnostic", "named"), default="agnostic",
                    help="'agnostic' matches Phase 2, which ships no language "
                         "tag; 'named' is the upper bound if we could route")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--four-bit", action="store_true",
                    help="required on a 16 GB T4")
    args = ap.parse_args()

    if args.max_s > 30:
        ap.error("Gemma 4 accepts at most 30 s of audio")

    ref, _, audio_dir = build_labeled_wavs(
        args.langs, args.n, args.work, args.shards, args.valid_frac, args.seed,
        args.min_s, args.max_s)

    processor, model = load_gemma(args.model, args.four_bit)
    print(f"\n{args.model}, {args.prompt} prompt, {len(ref):,} clips\n")

    paths = [audio_dir / f"{i}.wav" for i in ref.ID]
    hyps = transcribe(paths, ref.language.tolist(), processor, model,
                      args.prompt, args.max_new_tokens)
    refs = ref.transcription.astype(str).tolist()

    out = args.work / "gemma_pred.csv"
    pd.DataFrame({"ID": ref.ID, "language": ref.language,
                  "reference": refs, "Target": hyps}).to_csv(out, index=False)

    s = score(refs, hyps)
    per = score_by_language(refs, hyps, ref.language.tolist())
    print(f"\n=== zero-shot {args.model} ({len(refs):,} clips) ===")
    print(s)

    print(f"\n{'lang':<6}{'n':>5}{'gemma':>9}{'w2v-BERT':>11}   verdict")
    for lang in sorted(per):
        n = int((ref.language == lang).sum())
        g = per[lang].combined
        known = lang in wdata.LANGS
        base = BASELINE["in_domain"] if known else (
            BASELINE["nyn"] if lang == "nyn" else None)
        cmp = f"{base:.4f}" if base is not None else "  --"
        if base is None:
            verdict = "untrained, no w2v-BERT number"
        elif g < base - 0.05:
            verdict = "GEMMA BETTER"
        elif g > base + 0.05:
            verdict = "gemma worse"
        else:
            verdict = "comparable"
        print(f"{lang:<6}{n:>5}{g:>9.4f}{cmp:>11}   {verdict}")

    rw = np.array([len(r.split()) for r in refs])
    hw = np.array([len(h.split()) for h in hyps])
    ratio = hw.mean() / max(rw.mean(), 1e-9)
    print(f"\nwords/clip: reference {rw.mean():.1f}  hypothesis {hw.mean():.1f} "
          f"({ratio:.1f}x reference)")

    trained = [l for l in per if l in wdata.LANGS]
    untrained = [l for l in per if l not in wdata.LANGS]

    # An error above 1.0 cannot mean "does not know the language": transcribing
    # nothing at all scores 1.0. It means the hypothesis is far longer than the
    # reference, i.e. we are scoring commentary or scaffolding as transcription.
    # Saying anything about language competence from that would be wrong.
    if s.combined > 1.0 or ratio > 1.5:
        print("\n--- INCONCLUSIVE: this is an output-format problem ---")
        print(f"  combined {s.combined:.2f} with hypotheses {ratio:.1f}x the")
        print("  reference length. Emitting nothing at all would score 1.0, so a")
        print("  score above that measures insertions, not language ability.")
        print("  Gemma is returning something other than a bare transcript --")
        print("  preamble, commentary, or chat-template scaffolding.")
        print("  Inspect the samples above and the saved CSV, then re-run.")
        print("  Nothing can be concluded about Gemma's language coverage yet.")
        print(f"\nwrote {out}")
        return 0

    print("\n--- read this as ---")
    if untrained:
        u = np.mean([per[l].combined for l in untrained])
        print(f"  Gemma on languages our model never saw: {u:.4f}")
        print(f"  our CTC model on nyn, its easiest such language: {BASELINE['nyn']:.4f}")
        if u < 0.45:
            print("  -> Gemma's multilingual pretraining reaches these languages.")
            print("     Fine-tuning it on the three we have labels for can plausibly")
            print("     transfer to the rest, which breaks the 0.359 ceiling. The QA")
            print("     + transcription plan is worth the remaining days.")
        else:
            print("  -> Gemma does not meaningfully know these languages either, so")
            print("     fine-tuning it on three inherits the same 0.359 ceiling.")
            print("     The QA + transcription work would be competing for ~0.02.")
    if trained:
        t = np.mean([per[l].combined for l in trained])
        print(f"\n  On lin/lug/sna, Gemma zero-shot {t:.4f} vs our fine-tuned "
              f"{BASELINE['in_domain']:.4f}.")
        print("  Zero-shot against fine-tuned is not a fair fight -- this number is")
        print("  a floor for Gemma, not its ceiling. The untrained-language column")
        print("  is the one that decides the plan.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
