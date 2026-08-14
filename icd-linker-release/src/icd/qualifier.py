"""Small qualifier parser used by the deterministic ICD matcher.

The release matcher only requires three keys:
`base_text`, `qualifiers`, and `removed_phrases`.  Keep this conservative so
it never rewrites a diagnosis aggressively before reranker training.
"""

from __future__ import annotations

import re

_UNSPECIFIED_PATTERNS = [
    r"\bkh[oô]ng\s+đ[aã]\s*x[aá]c\s*đ[iị]nh\b",
    r"\bkh[oô]ng\s+r[oõ]\b",
    r"\bchưa\s+x[aá]c\s*đ[iị]nh\b",
    r"\bkh[oô]ng\s+đặc\s+hiệu\b",
    r"\bunspecified\b",
]


def parse_qualifiers(text: str) -> dict:
    """Return conservative qualifier metadata for `ICDMatcher`.

    We only strip very explicit "unspecified" phrases.  Other medical
    modifiers are left inside `base_text` so the downstream dictionary,
    fuzzy, dense retrieval, and reranker can decide with full context.
    """

    base = text or ""
    removed: list[str] = []
    unspecified = False
    for pattern in _UNSPECIFIED_PATTERNS:
        rx = re.compile(pattern, re.IGNORECASE)
        for match in rx.finditer(base):
            removed.append(match.group(0))
            unspecified = True
        base = rx.sub(" ", base)
    base = " ".join(base.split()) or (text or "")
    return {
        "base_text": base,
        "qualifiers": {"unspecified": unspecified},
        "removed_phrases": removed,
    }
