"""Assertion separation for Vietnamese medical spans.

Extracts clinical assertion (historical, negated, hypothetical, etc.)
from diagnosis text before entity linking.
"""

from __future__ import annotations

import re

# Vietnamese assertion patterns
# These are phrases that indicate the assertion status of a diagnosis,
# NOT medical qualifiers (acute/chronic, type 1/2, etc.)

_HISTORICAL_PREFIXES = [
    r"tiền sử\s+",
    r"tiền căn\s+",
    r"đã từng\s+",
    r"đã được chẩn đoán\s+",
    r"trước đây\s+",
    r"trong quá khứ\s+",
    r"có tiền sử\s+",
    r"bệnh sử\s+",
    r"từng\s+",
]

_NEGATED_PREFIXES = [
    r"không\s+có\s+",
    r"không\s+ghi\s+nhận\s+",
    r"không\s+phát\s+hiện\s+",
    r"không\s+thấy\s+",
    r"không\s+bị\s+",
    r"chưa\s+từng\s+",
    r"chưa\s+phát\s+hiện\s+",
    r"phủ\s+nhận\s+",
    r"không\s+",
]

_HYPOTHETICAL_PREFIXES = [
    r"nghi\s+ngờ\s+",
    r"nghi\s+",
    r"cần\s+loại\s+trừ\s+",
    r"có\s+thể\s+",
    r"chưa\s+loại\s+trừ\s+",
    r"theo\s+dõi\s+",
]

_FAMILY_PREFIXES = [
    r"gia\s+đình\s+có\s+",
    r"người\s+thân\s+",
]

# Generic verb prefixes to strip (not assertions, just noise)
_GENERIC_PREFIXES = [
    r"bị\s+",
    r"được\s+chẩn\s+đoán\s+",
    r"có\s+",
]

# Compile all patterns
_ALL_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("negated", [re.compile(p, re.IGNORECASE) for p in _NEGATED_PREFIXES]),
    ("historical", [re.compile(p, re.IGNORECASE) for p in _HISTORICAL_PREFIXES]),
    ("hypothetical", [re.compile(p, re.IGNORECASE) for p in _HYPOTHETICAL_PREFIXES]),
    ("family", [re.compile(p, re.IGNORECASE) for p in _FAMILY_PREFIXES]),
]

_GENERIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in _GENERIC_PREFIXES]


# Suffix patterns (appear at end of text)
_HISTORICAL_SUFFIXES = [
    r"\s+trước\s+đây",
    r"\s+đã\s+điều\s+trị",
    r"\s+cũ",
]

_NEGATED_SUFFIXES = [
    r"\s+âm\s+tính",
    r"\s+không\s+phát\s+hiện",
    r"\s+không\s+thấy",
]

_ALL_SUFFIX_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("negated", [re.compile(p, re.IGNORECASE) for p in _NEGATED_SUFFIXES]),
    ("historical", [re.compile(p, re.IGNORECASE) for p in _HISTORICAL_SUFFIXES]),
]


def separate_assertion(
    text: str,
    known_assertion: str | None = None,
) -> tuple[str, str | None]:
    """Separate assertion from diagnosis text.

    Args:
        text: Raw diagnosis span (e.g., "tiền sử nhồi máu cơ tim").
        known_assertion: Known assertion from upstream extractor, if any.

    Returns:
        (linking_text, assertion):
            linking_text: Clean text without assertion phrases.
            assertion: Normalized assertion: "negated", "historical",
                       "hypothetical", "family", or None (present/active).
    """
    if not text:
        return text, known_assertion

    original = text
    cleaned = text
    detected: str | None = None

    # If we already know the assertion from upstream, use it
    # But still strip phrases to clean the linking text
    if known_assertion:
        detected = known_assertion

    # Check prefix patterns (highest priority)
    for assertion_type, patterns in _ALL_PATTERNS:
        prev = cleaned
        for pat in patterns:
            m = pat.match(cleaned)
            if m:
                cleaned = cleaned[m.end():].strip()
                if detected is None:
                    detected = assertion_type
        if cleaned != prev:
            break

    # Strip generic prefixes (noise words)
    for pat in _GENERIC_COMPILED:
        m = pat.match(cleaned)
        if m:
            cleaned = cleaned[m.end():].strip()

    # Check suffix patterns
    if cleaned == original or True:  # always check suffixes too
        for assertion_type, patterns in _ALL_SUFFIX_PATTERNS:
            for pat in patterns:
                m = pat.search(cleaned)
                if m:
                    cleaned = cleaned[:m.start()].strip()
                    if detected is None:
                        detected = assertion_type
                    break

    # Normalize mapping
    if known_assertion:
        detected = known_assertion

    # Map to canonical forms used by ICDMatcher
    if detected == "negated":
        assertion = "negated"
    elif detected == "historical":
        assertion = None  # historical ≠ negated, allow matching
    elif detected == "hypothetical":
        assertion = None
    elif detected == "family":
        assertion = None
    else:
        assertion = None  # present / active

    return cleaned, assertion
