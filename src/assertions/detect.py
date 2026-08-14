"""Assertion detector.

Assigns isNegated / isFamily / isHistorical to ASSERTABLE entities
(TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC) using section context + clause-scoped cues.

Design decisions (from the review of earlier plans):
- isHistorical is driven by GENUINE history sections (SectionMap.historical_at)
  or explicit clause cues ("có tiền sử", "đã từng"). It is NOT applied to the
  present-illness narrative merely because it contains "trước khi nhập viện".
- isNegated only fires on a negation cue that appears BEFORE the entity within
  the same clause (commas do not break scope, so "Không A, B, C" negates all).
  A negation cue immediately followed by a modifier word ("không đặc hiệu",
  "không do chấn thương", ...) is NOT a negation of a separate entity.
- isFamily fires only when a relative is the subject and the patient is not
  named between the relative and the entity (informant pattern excluded).
"""

from __future__ import annotations

from typing import List, Optional

import yaml

from ..schema import (
    DATA_DIR, Entity, ASSERTABLE_TYPES,
    ASSERT_NEGATED, ASSERT_FAMILY, ASSERT_HISTORICAL,
)
from ..offsets import clause_start, is_word_char
from ..sections import SectionMap


class AssertionDetector:
    def __init__(self, cfg: dict):
        self.historical_cues = [c.casefold() for c in cfg.get("historical_cues", [])]
        self.historical_exceptions = [c.casefold() for c in cfg.get("historical_exceptions", [])]
        self.negation_cues = sorted(
            [c.casefold() for c in cfg.get("negation_cues", [])], key=len, reverse=True
        )
        self.negation_exceptions = [c.casefold() for c in cfg.get("negation_exceptions", [])]
        self.family_subjects = sorted(
            [c.casefold() for c in cfg.get("family_subjects", [])], key=len, reverse=True
        )
        self.family_block = [c.casefold() for c in cfg.get("family_informant_block", [])]
        self.family_exclusions = [c.casefold() for c in cfg.get("family_exclusions", [])]

    @classmethod
    def load(cls, path=None) -> "AssertionDetector":
        path = path or (DATA_DIR / "assertion_cues.yaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def apply(self, entities: List[Entity], text: str, smap: SectionMap) -> None:
        text_cf = text.casefold()
        for e in entities:
            if e.type not in ASSERTABLE_TYPES:
                e.assertions = []
                continue
            a: List[str] = []
            if self._historical(e, text_cf, smap):
                a.append(ASSERT_HISTORICAL)
            if self._negated(e, text, text_cf):
                a.append(ASSERT_NEGATED)
            if self._family(e, text_cf, smap):
                a.append(ASSERT_FAMILY)
            e.assertions = a

    # --- isHistorical --------------------------------------------------------
    def _historical(self, e: Entity, text_cf: str, smap: SectionMap) -> bool:
        if smap.historical_at(e.start):
            return True
        cs = clause_start(text_cf, e.start)
        window = text_cf[cs:e.start]
        for cue in self.historical_cues:
            start = 0
            while True:
                pos = window.find(cue, start)
                if pos == -1:
                    break
                after = window[pos + len(cue):].lstrip()
                # "tiền sử GIA ĐÌNH" is the relative's history, not the
                # patient's own -> isFamily handles it, not isHistorical.
                if any(after.startswith(exc) for exc in self.historical_exceptions):
                    start = pos + len(cue)
                    continue
                return True
        return False

    # --- isNegated -----------------------------------------------------------
    def _negated(self, e: Entity, text: str, text_cf: str) -> bool:
        cs = clause_start(text_cf, e.start)
        window = text_cf[cs:e.start]
        for cue in self.negation_cues:
            pos = window.rfind(cue)
            if pos == -1:
                continue
            after = window[pos + len(cue):].lstrip()
            # skip if the cue is immediately followed by a modifier word
            if any(after.startswith(exc) for exc in self.negation_exceptions):
                continue
            return True
        return False

    # --- isFamily ------------------------------------------------------------
    def _family(self, e: Entity, text_cf: str, smap: SectionMap) -> bool:
        if smap.family_at(e.start):
            return True
        cs = clause_start(text_cf, e.start)
        window = text_cf[cs:e.start]
        # mask non-relative phrases that merely contain a family word
        # ("bác sĩ gia đình", ...) so they don't trigger a false subject
        for exc in self.family_exclusions:
            start = 0
            while True:
                i = window.find(exc, start)
                if i == -1:
                    break
                window = window[:i] + (" " * len(exc)) + window[i + len(exc):]
                start = i + len(exc)
        subj_pos = self._last_family_subject(window)
        if subj_pos is None:
            return False
        # if the patient is named between the relative and the entity, the
        # relative is only an informant -> not family
        tail = window[subj_pos:]
        if any(b in tail for b in self.family_block):
            return False
        return True

    def _last_family_subject(self, window: str) -> Optional[int]:
        best = None
        for subj in self.family_subjects:
            idx = window.rfind(subj)
            while idx != -1:
                end = idx + len(subj)
                before = window[idx - 1] if idx > 0 else ""
                after = window[end] if end < len(window) else ""
                # require a whole-word match on BOTH sides, else "chị" would
                # match inside "chịu" (in "khó chịu"), etc.
                if not is_word_char(before) and not is_word_char(after):
                    if best is None or idx > best:
                        best = idx
                    break
                idx = window.rfind(subj, 0, idx)
        return best
