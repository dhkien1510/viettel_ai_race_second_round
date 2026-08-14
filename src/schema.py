"""Shared schema: entity types, assertion values, the Entity object, and the
hard offset/enum invariants that every produced item must satisfy.

Position convention (matches the official example, e.g. metoprolol -> [53, 75]):
    position == [start, end]  with END EXCLUSIVE, i.e. raw_text[start:end] == text
    indices are CHARACTER offsets into the raw (un-normalized) input text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = REPO_ROOT / "output"

# --- Enumerations (the ONLY legal values) ------------------------------------
TYPE_SYMPTOM = "TRIỆU_CHỨNG"
TYPE_TEST_NAME = "TÊN_XÉT_NGHIỆM"
TYPE_TEST_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
TYPE_DIAGNOSIS = "CHẨN_ĐOÁN"
TYPE_DRUG = "THUỐC"

VALID_TYPES = {
    TYPE_SYMPTOM,
    TYPE_TEST_NAME,
    TYPE_TEST_RESULT,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
}

ASSERT_NEGATED = "isNegated"
ASSERT_FAMILY = "isFamily"
ASSERT_HISTORICAL = "isHistorical"

VALID_ASSERTIONS = {ASSERT_NEGATED, ASSERT_FAMILY, ASSERT_HISTORICAL}

# assertions only apply to these three types (per the task spec)
ASSERTABLE_TYPES = {TYPE_SYMPTOM, TYPE_DIAGNOSIS, TYPE_DRUG}

# candidates (ICD-10 / RxNorm) only apply to these
LINKABLE_TYPES = {TYPE_DIAGNOSIS, TYPE_DRUG}


@dataclass
class Entity:
    """One extracted medical concept.

    `start`/`end` are character offsets into the raw input (end exclusive).
    `source` is a free-form tag for debugging (which extractor produced it).
    `canonical` is the dictionary key the surface resolved to (used by the
    linker); it never appears in the output.
    """

    text: str
    start: int
    end: int
    type: str
    assertions: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    source: str = ""
    canonical: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "position": [self.start, self.end],
            "type": self.type,
            "assertions": list(self.assertions),
            "candidates": list(self.candidates),
        }

    def key(self) -> tuple:
        return (self.start, self.end, self.type)


def validate_entity(e: Entity, raw_text: str) -> None:
    """Raise AssertionError if `e` violates any hard invariant."""
    assert e.type in VALID_TYPES, f"bad type: {e.type!r}"
    assert isinstance(e.start, int) and isinstance(e.end, int), "offsets must be int"
    assert 0 <= e.start < e.end <= len(raw_text), (
        f"offset out of range: [{e.start}, {e.end}] len={len(raw_text)}"
    )
    got = raw_text[e.start:e.end]
    assert got == e.text, f"text/offset mismatch: {got!r} != {e.text!r}"
    assert set(e.assertions) <= VALID_ASSERTIONS, f"bad assertions: {e.assertions}"
    if e.type not in ASSERTABLE_TYPES:
        assert not e.assertions, f"{e.type} must not carry assertions"
    if e.type not in LINKABLE_TYPES:
        assert not e.candidates, f"{e.type} must not carry candidates"


def validate_prediction(items: List[dict], raw_text: str) -> None:
    """Validate the serialized prediction list for one file."""
    for it in items:
        assert set(it.keys()) >= {"text", "position", "type", "assertions", "candidates"}
        assert isinstance(it["position"], list) and len(it["position"]) == 2
        s, en = it["position"]
        assert it["type"] in VALID_TYPES, f"bad type {it['type']!r}"
        assert 0 <= s < en <= len(raw_text), f"bad position {it['position']}"
        assert raw_text[s:en] == it["text"], (
            f"text mismatch at {it['position']}: {raw_text[s:en]!r} != {it['text']!r}"
        )
        assert set(it["assertions"]) <= VALID_ASSERTIONS
