from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_AR_DIACRITICS_RE = re.compile(
    "["  # Arabic harakat + small letters
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06ED"
    "]"
)

_ARABIC_CHAR_MAP = str.maketrans(
    {
        # Alef variants
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        # Alef maqsura / ya
        "ى": "ي",
        # Ta marbuta (policy: map to ه to reduce sparsity in dialect)
        "ة": "ه",
        # Hamza-on-waw/ya variants
        "ؤ": "و",
        "ئ": "ي",
        # Tatweel
        "ـ": "",
        # Common Maghrebi letters (reduce sparsity)
        "ڨ": "ق",
        "پ": "ب",
        "ڤ": "ف",
        "چ": "ج",
        # Arabic punctuation -> latin punctuation
        "،": ",",
        "؟": "?",
        "؛": ";",
        # Arabic-Indic digits -> Western digits
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", flags=re.UNICODE)
_HASHTAG_RE = re.compile(r"#(\w+)", flags=re.UNICODE)
# More precise phone regex for Algeria: starts with 0 or +213, followed by mobile/landline patterns
_PHONE_RE = re.compile(r"\b(?:\+?213|0)[\s\-]?[567]\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}\b")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([!?.,;:])\1{1,}")
_MULTI_CHAR_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)  # 3+ repeats -> 2 repeats
_ZW_CHARS_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u2066-\u2069]")  # ZWJ/ZWNJ + bidi marks

_AR_BLOCK_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_BLOCK_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


@dataclass(frozen=True)
class TextNormConfig:
    """
    Keep this configurable so we can A/B test what actually helps Macro F1.
    """

    keep_url_token: bool = True
    keep_mention_token: bool = True
    keep_hashtag_text: bool = True
    keep_hashtag_token: bool = True
    mask_phone_numbers: bool = True
    reduce_elongations: bool = True
    lowercase_latin: bool = True
    deaccent_latin: bool = False
    keep_digits: bool = True
    arabizi_map_digits: bool = True  # 3->ع, 7->ح, 9->ق ... (common Maghrebi Arabizi)


# IMPORTANT: Arabizi mapping is DISABLED by default now.
# Mapping digits globally destroys numeric information (prices, dates, quantities).
# Only enable if you're sure the text is pure Arabizi with no real numbers.
_ARABIZI_DIGIT_MAP = str.maketrans(
    {
        "2": "ء",
        "3": "ع",
        "4": "ش",  # not always, but common enough to test
        "5": "خ",
        "6": "ط",
        "7": "ح",
        "8": "ق",  # variant; we also map 9->ق
        "9": "ق",
    }
)

# Smarter Arabizi detection: only map digits when adjacent to Latin letters
# (e.g., "sa7a", "3lach"), which avoids corrupting real numbers like "2023", "1000 DA", phone numbers.
_ARABIZI_CONTEXT_RE = re.compile(r"(?:(?<=[A-Za-z])[2-9]|[2-9](?=[A-Za-z]))")


def has_arabic(text: str) -> bool:
    return bool(_AR_BLOCK_RE.search(text or ""))


def has_latin(text: str) -> bool:
    return bool(_LATIN_BLOCK_RE.search(text or ""))


def keep_only_arabic(text: str) -> str:
    """
    Keep Arabic letters + digits + spaces + a small set of punctuation.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9\s!?.,;:]+", " ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def keep_only_latin(text: str) -> str:
    """
    Keep Latin letters (incl. accents) + digits + spaces + a small set of punctuation.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s!?.,;:]+", " ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_text(text: str, cfg: TextNormConfig | None = None) -> str:
    """
    Normalization for Darija + French social text.
    Goal: reduce sparsity without destroying meaning. Configurable to allow A/B tests.
    """
    if cfg is None:
        cfg = TextNormConfig()
    if text is None:
        return ""

    # Unicode normalization first (handles weird forms, half-width, etc.)
    text = unicodedata.normalize("NFKC", str(text))

    # Remove zero-width / bidi control characters
    text = _ZW_CHARS_RE.sub(" ", text)

    # URLs / mentions / hashtags -> either remove or replace with tokens
    if cfg.keep_url_token:
        text = _URL_RE.sub(" URL ", text)
    else:
        text = _URL_RE.sub(" ", text)

    if cfg.keep_mention_token:
        text = _MENTION_RE.sub(" USER ", text)
    else:
        text = _MENTION_RE.sub(" ", text)

    def _hashtag_repl(m: re.Match) -> str:
        tag = m.group(1) or ""
        tag = tag.replace("_", " ").replace("-", " ")
        pieces: list[str] = []
        if cfg.keep_hashtag_token:
            pieces.append(" HASHTAG ")
        if cfg.keep_hashtag_text and tag:
            pieces.append(tag)
        return " ".join(pieces) if pieces else " "

    text = _HASHTAG_RE.sub(_hashtag_repl, text)

    # Optional: mask phone numbers (common in complaints)
    if cfg.mask_phone_numbers:
        text = _PHONE_RE.sub(" PHONE ", text)

    # Arabic normalization + diacritics removal
    text = text.translate(_ARABIC_CHAR_MAP)
    text = _AR_DIACRITICS_RE.sub("", text)

    # Latin processing
    if cfg.lowercase_latin:
        text = text.lower()

    if cfg.deaccent_latin:
        # Remove latin accents but keep Arabic intact (accents are combining marks)
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))

    # Arabizi digit mapping - NOW CONTEXT-AWARE to avoid destroying real numbers
    # Only map digits that are surrounded by Latin letters (like "sa7a", "3lach")
    if cfg.arabizi_map_digits:
        # Context-aware replacement: only digits between Latin letters
        def _arabizi_repl(m: re.Match) -> str:
            digit = m.group(0)
            mapping = {"2": "ء", "3": "ع", "4": "ش", "5": "خ", "6": "ط", "7": "ح", "8": "ق", "9": "ق"}
            return mapping.get(digit, digit)
        text = _ARABIZI_CONTEXT_RE.sub(_arabizi_repl, text)

    if not cfg.keep_digits:
        text = re.sub(r"\d+", " ", text)

    # Normalize repeated punctuation
    text = _MULTI_PUNCT_RE.sub(r"\1\1", text)

    # Reduce elongations / repeated characters: "bezzzzaf" -> "bezzaf", "رااااني" -> "رااني"
    if cfg.reduce_elongations:
        text = _MULTI_CHAR_RE.sub(r"\1\1", text)

    # Trim and collapse spaces
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


