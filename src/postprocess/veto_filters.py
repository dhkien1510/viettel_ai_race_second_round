"""Rule-based veto filter — strips known false-positive patterns AFTER the
LLM (or any) backend has run, so the model's prompt doesn't have to carry
every nuance perfectly on every call.

Each veto below is backed by policy validation notes and manual review, not
a guess. Kept deliberately narrow and high-precision: a veto here silently
removes a prediction, so each rule only fires on patterns that were confirmed
as false positives across multiple documents. Anything context-dependent or
only weakly confirmed is left out on purpose rather than risk stripping a
real entity.
"""

from __future__ import annotations

import re
from typing import List

from ..schema import Entity, TYPE_DIAGNOSIS, TYPE_TEST_NAME


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


# --- 1. Physiological state (never a CHẨN_ĐOÁN, regardless of modifiers) ---
# Confirmed as a stable false-positive pattern in policy validation.
_PHYSIOLOGICAL_STATE = re.compile(
    r"(c[oó]\s+thai|mang\s+thai|m[aã]n\s+kinh)", re.IGNORECASE,
)


# --- 2. Redaction mask: entity text is mostly/entirely asterisks --------
# Masked/redacted spans should not be guessed as entities.
def _is_masked(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    stars = stripped.count("*")
    return stars > 0 and stars / len(stripped) >= 0.5


# --- 3. Bare generic exam/test words that are never a standalone entity --
# Deliberately NOT including "nội soi"/"siêu âm"/"x-quang" alone — those ARE
# valid standalone TÊN_XÉT_NGHIỆM per [GT9]; only bare-generic non-specific
# phrasing is safe to veto unconditionally.
_GENERIC_TEST_NAME = {
    "xét nghiệm", "khám chuyên khoa", "khám lâm sàng", "chẩn đoán hình ảnh",
    "nhìn bên ngoài",
}


# --- 4. Patient's own uncertain self-question about their OWN diagnosis --
# This fires ONLY when the patient is asking about themselves
# ("em có phải...", "tôi có phải...");
# a doctor's hedge ("bác sĩ nói khả năng...") or a generic FAQ header
# ("Bệnh dại có lây không?") must NOT be caught here — checked via a window
# around the entity span (below) rather than one regex spanning the whole
# variable-length entity text.
def _is_self_question(entity: Entity, text: str) -> bool:
    """Check whether `entity` sits inside a first-person self-diagnostic
    question ("em có phải bị X không ạ?"). Looks at a window AROUND the
    entity span rather than requiring an exact regex match spanning the
    whole (variable-length) entity text."""
    window_start = max(0, entity.start - 40)
    window_end = min(len(text), entity.end + 25)
    before = text[window_start:entity.start]
    after = text[entity.end:window_end]
    has_self_lead = re.search(
        r"(em|t[oô]i|con|ch[aá]u)\s+c[oó]\s+ph[aả]i\s+(l[aà]\s+)?(?:bị\s+)?$",
        before, re.IGNORECASE,
    )
    has_question_tail = re.search(r"^[^.?]{0,20}kh[oô]ng\s*(?:[aạ]\b|\?)", after, re.IGNORECASE)
    return bool(has_self_lead and has_question_tail)


def apply_veto_filters(entities: List[Entity], text: str) -> List[Entity]:
    """Drop entities matching a confirmed false-positive pattern. Pure
    filter — never adds, edits text, or changes type; only removes."""
    out = []
    for e in entities:
        norm = _norm(e.text)

        if e.type == TYPE_DIAGNOSIS and _PHYSIOLOGICAL_STATE.search(norm):
            continue
        if _is_masked(e.text):
            continue
        if e.type == TYPE_TEST_NAME and norm in _GENERIC_TEST_NAME:
            continue
        if e.type == TYPE_DIAGNOSIS and text and _is_self_question(e, text):
            continue

        out.append(e)
    return out


# TODO (left out deliberately: context-dependent / thin evidence; do NOT add
# as a blanket veto without fresh policy validation per pattern):
#   - "quyết định y khoa chưa chốt" (§12: thuốc/xét nghiệm mà bác sĩ SẼ
#     quyết định tiếp tục/ngưng trong tương lai) — only 1 confirmed instance
#     (aspirin, file 100), phrasing too varied to regex safely yet.
#   - mệnh đề cơ chế/định nghĩa (§10) — must be judged PER OCCURRENCE (the
#     same phrase can be a real finding elsewhere in the same file), not a
#     safe global text-match veto.
#   - mô tả chức năng/khả năng chung (§15) — only 2-3 confirmed instances,
#     all long free-text sentences with no stable shared substring to match.
