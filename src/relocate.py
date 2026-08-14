"""Relocate concept surfaces onto raw-text character offsets.

Used by:
  - the LLM backend (model emits {text,type,assertions} without position), and
  - word-level encoders (predicted surface contains underscores).

We must guarantee raw_text[start:end] == text, so the emitted `text` is always
the RAW slice (not the model's string, which may differ in case/underscore).
Multiple occurrences of the same surface are assigned left-to-right in the
order the model listed them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .schema import Entity, VALID_TYPES, VALID_ASSERTIONS, ASSERTABLE_TYPES
from .offsets import at_word_boundary, iter_occurrences, strip_accents_lower


def _all_spans(
    raw: str, surface: str, *, case_insensitive: bool = True
) -> List[Tuple[int, int]]:
    n = len(surface)
    return [
        (i, i + n)
        for i in iter_occurrences(raw, surface, case_insensitive=case_insensitive)
    ]


def _fuzzy_span(raw: str, surface: str) -> Optional[Tuple[int, int]]:
    """Accent-insensitive fallback locate (handles diacritic/typo drift)."""
    if not surface:
        return None
    raw_fold = strip_accents_lower(raw)
    surf_fold = strip_accents_lower(surface)
    idx = raw_fold.find(surf_fold)
    if idx == -1 or len(surf_fold) == 0:
        return None
    # strip_accents_lower preserves length (NFD marks removed but base kept?);
    # to stay safe, only trust this when lengths align 1:1
    if len(raw_fold) != len(raw) or len(surf_fold) != len(surface):
        return None
    return idx, idx + len(surface)


def relocate(
    raw: str,
    items: List[dict],
    *,
    require_boundary: bool = False,
    exact_case_for_short: bool = False,
) -> List[Entity]:
    used: set = set()
    out: List[Entity] = []
    for it in items:
        typ = it.get("type")
        if typ not in VALID_TYPES:
            continue
        surface = (it.get("text") or "").replace("_", " ").strip()
        if not surface:
            continue
        case_insensitive = not (exact_case_for_short and len(surface) <= 2)
        spans = _all_spans(raw, surface, case_insensitive=case_insensitive)
        if require_boundary:
            spans = [
                (start, end)
                for start, end in spans
                if at_word_boundary(raw, start, end)
            ]
        chosen: Optional[Tuple[int, int]] = None
        for s, e in spans:
            if (s, e) not in used:
                chosen = (s, e)
                break
        if chosen is None and not spans and len(surface) > 2:
            fuzzy = _fuzzy_span(raw, surface)
            if fuzzy is not None and (
                not require_boundary or at_word_boundary(raw, *fuzzy)
            ):
                chosen = fuzzy
        if chosen is None:
            continue  # cannot produce a valid offset -> drop (keeps invariant)
        s, e = chosen
        used.add((s, e))
        if typ in ASSERTABLE_TYPES:
            # LLM zero-shot (không ép JSON schema) đôi khi trả "assertions" sai
            # hình dạng (vd [{"isNegated": true}] thay vì ["isNegated"]) ->
            # dict không hashable sẽ làm dict.fromkeys() crash. Lọc bỏ phần tử
            # không phải string TRƯỚC khi dedupe thay vì giả định mọi phần tử
            # đều hợp lệ.
            raw_ass = it.get("assertions") or []
            if not isinstance(raw_ass, list):
                raw_ass = []
            str_ass = [a for a in raw_ass if isinstance(a, str)]
            # dedupe (giữ thứ tự) + chỉ giữ assertion hợp lệ
            ass = [a for a in dict.fromkeys(str_ass) if a in VALID_ASSERTIONS]
        else:
            ass = []
        out.append(Entity(raw[s:e], s, e, typ, assertions=ass, source="relocate"))
    return out
