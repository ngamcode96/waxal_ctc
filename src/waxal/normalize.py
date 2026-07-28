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
#
# KEEP covers lin/lug/sna only. The corpus has 19 languages and Phase 2 draws
# from more than three, so training beyond LANGS needs KEEP_19 -- measured
# 2026-07-28 over one train shard per language, this KEEP silently destroys:
#
#     amh  20.5% of characters survive    tir  21.6%    <- Ethiopic, wiped out
#     kpo  80.9%    ewe  90.3%    aka  92.1%    dga  93.2%    dag  94.1%
#     ful  94.8%
#
# so eight languages lose text, not just the two in a different script. Deleting
# 'ɔ' and 'ɛ' from Akan or Ewe does not make those languages merely harder, it
# makes the target transcripts wrong.
KEEP = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ŋ"                       # Luganda
    "àáâçèéêëìíîïòóôùúûü"      # Lingala/Shona diacritics (lowercase only)
    " '-.,!?;:()\""
)

# The 19-language alphabet: every letter appearing at least 50 times across one
# train shard per language, unioned with KEEP so nothing lin/lug/sna already
# train on can be dropped. 319 characters, covering 99.96% of all letter tokens
# in the corpus -- the tail below that threshold is encoding noise and one-off
# borrowings that would get too little training signal to learn.
KEEP_19 = KEEP | set(
    # Latin beyond ASCII: African orthographies (Ewe, Dagbani, Fula, Akan,
    # Kabiye) plus the Lingala/Shona diacritics.
    "àáâãçèéêëìíîïòóôõùúûüēĩŋũūƁƆƉƊƐƒƙƴǝɑɓɔɖɗɛɡɣɩɲʊʋʒͻιԑẽỳ"
    # Ethiopic syllabary: Amharic and Tigrinya.
    "ሀሁሂሃሄህሆለሉሊላሌልሎሏሐሑሒሓሕመሙሚማሜምሞሟሠሣሥረ"
    "ሩሪራሬርሮሯሰሱሲሳሴስሶሸሹሻሽቀቁቂቃቅቆቋቐቑቓቕበቡቢ"
    "ባቤብቦቧቪተቱቲታቴትቶቷቸቹቻችኃኅኋነኑኒናኔንኖኘኙኛኝ"
    "ኞአኡኢኣኤእኦከኩኪካኬክኮኳኸኹኻኽኾወዉዊዋውዎዐዑዒዓዕ"
    "ዘዙዚዛዜዝዞዣዥየዩዪያይዮደዱዲዳዴድዶጀጁጂጃጅጆገጉጊጋ"
    "ጌግጎጐጓጕጠጡጢጣጤጥጦጧጨጫጭጴጵጸጹጺጻጽፀፁፂፃፅፈፉፊ"
    "ፋፌፍፎፏፐፒፓፕፖ"
)

_ACCENT_UPPER = str.maketrans("ÀÁÂÇÈÉÊËÌÍÎÏÒÓÔÙÚÛÜ", "àáâçèéêëìíîïòóôùúûü")

# Which alphabet clean() is currently using. A run-level setting rather than a
# per-call argument because clean() is called from training, evaluation and
# scoring alike, and every one of them has to agree -- a reference cleaned with
# one alphabet and a hypothesis with another would score nonsense.
_ACTIVE = KEEP


def set_alphabet(langs) -> set:
    """Pick the alphabet for this run from the languages being trained on.

    Called once at startup. Anything beyond lin/lug/sna needs KEEP_19, or the
    transcripts for eight of the nineteen languages arrive partly or wholly
    deleted -- see the note on KEEP.
    """
    global _ACTIVE
    extra = set(langs) - {"lin", "lug", "sna"}
    _ACTIVE = KEEP_19 if extra else KEEP
    print(f"alphabet: {'KEEP_19' if extra else 'KEEP'} "
          f"({len(_ACTIVE)} characters)"
          + (f" -- required by {sorted(extra)}" if extra else ""))
    return _ACTIVE


def active_alphabet() -> set:
    return _ACTIVE


def clean(text: str, keep: set | None = None) -> str:
    """Normalize a transcript to the training alphabet, preserving case/punctuation."""
    if not isinstance(text, str):
        return ""
    keep = _ACTIVE if keep is None else keep

    text = unicodedata.normalize("NFC", text)
    for src, dst in CHAR_MAP.items():
        text = text.replace(src, dst)

    # Uppercase accented letters are vanishingly rare; folding them to lowercase
    # avoids spending alphabet slots on symbols with ~no training signal.
    text = text.translate(_ACCENT_UPPER)

    # Digits are read aloud as words, so a literal digit is never the right
    # target. Drop them rather than teach the model an unpronounceable symbol.
    text = re.sub(r"\d+", " ", text)

    text = "".join(c for c in text if c in keep)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alphabet(texts: list[str]) -> list[str]:
    """The CTC vocabulary implied by a corpus, after cleaning."""
    seen = set()
    for t in texts:
        seen.update(clean(t))
    return sorted(seen)
