"""Fuzzy string matching for ICD names and aliases."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only without dependency
    fuzz = None


@dataclass(frozen=True)
class FuzzyCandidate:
    key: str
    code: str
    name_vi: str
    score: float
    source: str


def _fallback_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio() * 100


def _token_f1(left: str, right: str) -> float:
    """Token-level F1 score: precision * recall * 2 / (precision + recall)."""
    q = set(left.split())
    c = set(right.split())
    if not q or not c:
        return 0.0
    overlap = len(q & c)
    precision = overlap / len(c)
    recall = overlap / len(q)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score(left: str, right: str) -> float:
    if fuzz is None:
        return _fallback_ratio(left, right)

    left_tokens = left.split()
    right_tokens = right.split()

    # Short-name guard: candidate 1-token không được fuzzy high-score với query >= 2 token
    # vì partial_ratio("...khong...", "ho") = 100 do "ho" là substring của "khong"
    if len(right_tokens) == 1 and len(left_tokens) >= 2:
        # Chỉ cho điểm nếu token xuất hiện nguyên vẹn trong query
        if right_tokens[0] not in left_tokens:
            return 0.0

    scores = [
        fuzz.token_set_ratio(left, right),
        fuzz.token_sort_ratio(left, right),
    ]

    shorter = min(len(left), len(right))
    shorter_tokens = min(len(left_tokens), len(right_tokens))
    # partial_ratio chỉ an toàn khi cả 2 đều đủ dài (>= 4 ký tự, >= 2 token)
    if shorter >= 4 and shorter_tokens >= 2:
        scores.append(fuzz.partial_ratio(left, right))

    base = float(max(scores))

    # Nếu token F1 quá thấp, không cho high-confidence dù partial_ratio cao
    f1 = _token_f1(left, right)
    if f1 < 0.40:
        base = min(base, 84.0)

    return base


def top_fuzzy_candidates(
    query: str,
    choices: Iterable[tuple[str, str, str, str]],
    top_k: int = 5,
) -> list[FuzzyCandidate]:
    """Return top fuzzy candidates from `(key, code, name_vi, source)` choices."""
    candidates = [
        FuzzyCandidate(key=key, code=code, name_vi=name, score=_score(query, key), source=source)
        for key, code, name, source in choices
        if key
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:top_k]

