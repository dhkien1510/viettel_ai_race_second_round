"""Vietnamese text normalization helpers for ICD matching."""

from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[\.,;:!?()\[\]{}\"'`“”‘’/\\|_+=~]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_conservative(text: str | None) -> str:
    """Normalize text while preserving Vietnamese accents."""
    if not text:
        return ""
    value = unicodedata.normalize("NFC", str(text)).lower().strip()
    value = value.replace("-", " ")
    value = _PUNCT_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def strip_vietnamese_accents(text: str) -> str:
    """Remove Vietnamese diacritics, including the special đ character."""
    value = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_type_variants(value: str) -> str:
    value = re.sub(r"\b(type|typ|tuyp)\s*(ii|2)\b", "tip 2", value)
    value = re.sub(r"\b(type|typ|tuyp)\s*(i|1)\b", "tip 1", value)
    value = re.sub(r"\btip\s*ii\b", "tip 2", value)
    value = re.sub(r"\btip\s*i\b", "tip 1", value)
    return value


def normalize_aggressive(text: str | None) -> str:
    """Normalize for lookup: lowercase, punctuation-cleaned, no accents."""
    value = strip_vietnamese_accents(normalize_conservative(text))
    value = _normalize_type_variants(value)
    value = re.sub(r"\bman tinh\b", "man", value)
    return _SPACE_RE.sub(" ", value).strip()

