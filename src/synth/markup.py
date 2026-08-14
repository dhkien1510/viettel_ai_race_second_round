"""Shared markup parser: ⟦TYPE|assert⟧surface⟦/⟧  ->  clean text + entities.

Offsets are computed on the CLEAN text (markers stripped) so raw[start:end]==text.
Used by scripts/parse_synthetic.py (synthetic) and scripts/merge_annotations.py
(real-note gold from 3 LLMs).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..schema import Entity, VALID_TYPES, VALID_ASSERTIONS, ASSERTABLE_TYPES

OPEN = re.compile(r"⟦(?P<type>[A-ZÀ-Ỹ_]+)(?:\|(?P<ass>[^⟧]*))?⟧")
CLOSE = "⟦/⟧"

# Generators occasionally garble the rare ⟦/⟧ token sequence (sampling noise, not a
# content error). Normalize known typo shapes back to ⟦/⟧ before parsing so a single
# mangled close tag doesn't swallow every entity up to the next valid one.
_MALFORMED_CLOSE = [
    (re.compile(r"⟧/⟧"), "⟦/⟧"),        # stray leading ⟧ instead of ⟦
    (re.compile(r"⟧/(?!⟧)"), "⟦/⟧"),    # missing final ⟧
    (re.compile(r"⟦/[^⟧]*⟧"), "⟦/⟧"),   # extra slash, or stray words/newline injected mid-close-tag
]


def sanitize_markup(raw: str) -> str:
    """Repair known malformed ⟦/⟧ close-tag typos. Idempotent."""
    out = raw
    for pat, repl in _MALFORMED_CLOSE:
        out = pat.sub(repl, out)
    return out


OPEN_CH, CLOSE_CH = "⟦", "⟧"


def parse_markup(marked: str) -> Tuple[str, List[Entity]]:
    """Return (clean_text, entities). Raises ValueError on an unclosed tag.

    Đóng thẻ tại DẤU HIỆU SỚM NHẤT, không phải ⟦/⟧ chuẩn kế tiếp. Generator hay
    dùng ⟧ ĐƠN làm thẻ đóng thay cho ⟦/⟧ (`⟦TÊN_XN⟧SpO2⟧ ⟦KQ⟧97%⟦/⟧`); bản cũ
    nhảy thẳng tới ⟦/⟧ sau `97%` nên surface của thẻ đầu NUỐT cả `⟧ ⟦KQ⟧` — ký tự
    markup rác lọt vào nhãn ở 83/516 note, model học ⟦/⟧ như một phần khái niệm.
    Điểm đóng của một thẻ mở = cái nào tới trước trong:
      · ⟦/⟧  (đóng chuẩn)         · ⟦TYPE⟧ mở mới (thẻ này bị quên đóng)
      · ⟧ đơn (đóng lỗi)
    """
    clean_parts: List[str] = []
    raw_ents = []
    i = 0
    n = len(marked)
    clean_len = 0
    while i < n:
        m = OPEN.match(marked, i)
        if m:
            typ = m.group("type")
            ass_raw = (m.group("ass") or "").strip()
            p_open = marked.find(OPEN_CH, m.end())    # ⟦ kế tiếp (⟦/⟧ hoặc mở mới)
            p_bar = marked.find(CLOSE_CH, m.end())    # ⟧ kế tiếp (đóng chuẩn/lỗi)
            if p_bar == -1 and p_open == -1:
                raise ValueError(f"unclosed tag at char {i}")
            if p_open != -1 and (p_bar == -1 or p_open < p_bar):
                surface_end = p_open
                next_i = p_open + len(CLOSE) if marked[p_open:p_open + len(CLOSE)] == CLOSE else p_open
            else:
                surface_end = p_bar
                next_i = p_bar + 1
            surface = marked[m.end():surface_end]
            start = clean_len
            clean_parts.append(surface)
            clean_len += len(surface)
            end = clean_len
            assertions = [a.strip() for a in ass_raw.split(",") if a.strip()] if ass_raw else []
            raw_ents.append((typ, assertions, start, end))
            i = next_i
        else:
            # Bỏ ký tự marker MỒ CÔI ở top level (⟦ / ⟧ không thuộc thẻ nào) —
            # đừng để chúng lọt vào clean text và làm lệch offset mọi span sau.
            if marked[i] not in (OPEN_CH, CLOSE_CH):
                clean_parts.append(marked[i])
                clean_len += 1
            i += 1
    clean_text = "".join(clean_parts)

    # HÀNG RÀO CỨNG: không ký tự markup nào được sống sót vào clean text.
    if OPEN_CH in clean_text or CLOSE_CH in clean_text:
        raise ValueError("markup rác còn sót trong clean text")

    out: List[Entity] = []
    for typ, assertions, s, e in raw_ents:
        if typ not in VALID_TYPES:
            continue
        assertions = [a for a in assertions if a in VALID_ASSERTIONS]
        if typ not in ASSERTABLE_TYPES:
            assertions = []
        out.append(Entity(clean_text[s:e], s, e, typ, assertions=assertions, source="markup"))
    return clean_text, out
