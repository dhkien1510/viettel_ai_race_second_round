"""Offset-safe text utilities.

Everything here operates on the RAW input string and returns character
offsets into it. We never normalize the text and then map back — offsets are
always taken directly on the raw string so that raw_text[start:end] holds.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterator, List, Tuple

# Vietnamese-aware "word character" test. We treat any Unicode letter or digit
# (which covers all Vietnamese diacritics) plus underscore as part of a token.
_WORD_RE = re.compile(r"\w", re.UNICODE)


def read_text(path: str | Path) -> str:
    """Read a file preserving its exact characters (no newline translation).

    We pass newline='' so \\r\\n is NOT collapsed to \\n; this keeps our
    character offsets identical to the on-disk file.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def is_word_char(ch: str) -> bool:
    if not ch:
        return False
    # Python's ``\w`` excludes standalone combining marks. Vietnamese input
    # may use decomposed characters (e.g. ``a`` + U+0300), so a mark must stay
    # attached to the surrounding token for boundary checks.
    return bool(_WORD_RE.match(ch)) or unicodedata.category(ch).startswith("M")


def at_word_boundary(text: str, start: int, end: int) -> bool:
    """True if [start, end) is not glued to surrounding word characters."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not is_word_char(before) and not is_word_char(after)


def iter_occurrences(
    text: str, needle: str, *, case_insensitive: bool = True
) -> Iterator[int]:
    """Yield every start index where `needle` occurs in `text`.

    Occurrences may not overlap each other (left-to-right, longest step).
    """
    if not needle:
        return
    hay = text.casefold() if case_insensitive else text
    nee = needle.casefold() if case_insensitive else needle
    start = 0
    while True:
        idx = hay.find(nee, start)
        if idx == -1:
            return
        yield idx
        start = idx + len(nee)


def find_phrase_spans(
    text: str,
    phrase: str,
    *,
    require_boundary: bool = True,
    case_insensitive: bool = True,
) -> List[Tuple[int, int]]:
    """All spans of `phrase` in `text`. When require_boundary, only spans that
    are not glued to surrounding word characters are returned (prevents e.g.
    the symptom "ho" from matching inside "khó")."""
    out: List[Tuple[int, int]] = []
    n = len(phrase)
    for idx in iter_occurrences(text, phrase, case_insensitive=case_insensitive):
        end = idx + n
        if require_boundary and not at_word_boundary(text, idx, end):
            continue
        out.append((idx, end))
    return out


def line_spans(text: str) -> List[Tuple[int, int, str]]:
    """Split into lines, returning (start, end, line_text_without_newline).

    end is the offset just past the last non-newline char of the line.
    """
    out: List[Tuple[int, int, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            line = text[pos:]
            # strip a trailing \r if present (Windows)
            core = line.rstrip("\r")
            out.append((pos, pos + len(core), core))
            break
        line = text[pos:nl]
        core = line.rstrip("\r")
        out.append((pos, pos + len(core), core))
        pos = nl + 1
    return out


# Clause / scope delimiters used by the assertion detector.
CLAUSE_BREAKERS = set(".;:\n\r()[]")


def clause_start(text: str, pos: int) -> int:
    """Index of the start of the clause containing `pos` (scan left until a
    clause breaker). Commas do NOT break a clause (negation can distribute
    over a comma-separated list)."""
    i = pos
    while i > 0 and text[i - 1] not in CLAUSE_BREAKERS:
        i -= 1
    return i


def strip_accents_lower(s: str) -> str:
    """Accent-folded, lower-cased form — for fuzzy/typo comparison only.
    Never used to compute output offsets."""
    nfkd = unicodedata.normalize("NFD", s)
    no_marks = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return no_marks.casefold()
