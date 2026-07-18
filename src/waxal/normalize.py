"""Transcript cleanup for CTC training.

Deliberately conservative. We measured the cost of emitting normalized text
against the raw cased references (scripts/normalization_cost.py):

    lowercase + strip punctuation          -> 0.1011 combined
    ... plus deterministic recasing        -> 0.0703 combined
    raw cased + punctuated                 -> 0.0000

So case and punctuation stay. They're largely positional (89.3% of references
start with a capital, 83.8% end in terminal punctuation), which CTC learns
cheaply. All this module does is fold away the long tail of junk characters so
the CTC alphabet stays small and every symbol has enough training signal.
"""

import re
import unicodedata

# Rare characters that are almost certainly transcription noise or encoding
# damage, mapped to their intended form. Counts are from the 38k train rows.
CHAR_MAP = {
    "\xa0": " ", "​": "", "﻿": "",
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'", "‛": "'", "`": "'", "´": "'",
    "–": "-", "—": "-", "‑": "-",
    "ᵑ": "ŋ",            # superscript n -> the real Luganda velar nasal
    "Ŋ": "ŋ",            # only a handful of uppercase forms; fold to lowercase
    "Ķ": "K", "ķ": "k", "Ĺ": "L", "ĺ": "l", "ĝ": "g", "ā": "a",
    "Œ": "OE", "œ": "oe", "þ": "th", "×": "x",
    "⭐": "", "️": "",     # emoji + variation selector
    "…": ".",
}

# The alphabet we actually train on. Anything outside this is dropped.
KEEP = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ŋ"                       # Luganda
    "àáâçèéêëìíîïòóôùúûü"      # Lingala/Shona diacritics (lowercase only)
    " '-.,!?;:()\""
)

_ACCENT_UPPER = str.maketrans("ÀÁÂÇÈÉÊËÌÍÎÏÒÓÔÙÚÛÜ", "àáâçèéêëìíîïòóôùúûü")


def clean(text: str) -> str:
    """Normalize a transcript to the training alphabet, preserving case/punctuation."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    for src, dst in CHAR_MAP.items():
        text = text.replace(src, dst)

    # Uppercase accented letters are vanishingly rare; folding them to lowercase
    # avoids spending alphabet slots on symbols with ~no training signal.
    text = text.translate(_ACCENT_UPPER)

    # Digits are read aloud as words, so a literal digit is never the right
    # target. Drop them rather than teach the model an unpronounceable symbol.
    text = re.sub(r"\d+", " ", text)

    text = "".join(c for c in text if c in KEEP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alphabet(texts: list[str]) -> list[str]:
    """The CTC vocabulary implied by a corpus, after cleaning."""
    seen = set()
    for t in texts:
        seen.update(clean(t))
    return sorted(seen)
