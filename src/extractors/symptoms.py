"""Symptom extractor (TRIỆU_CHỨNG).

Dictionary substring matching with word-boundary guards (so "ho" does not match
inside "khó"), plus a small literal typo map. Longest-match-wins is enforced
later by the conflict resolver.
"""

from __future__ import annotations

from typing import List

import yaml

from ..schema import DATA_DIR, Entity, TYPE_SYMPTOM
from ..offsets import find_phrase_spans


class SymptomExtractor:
    def __init__(self, cfg: dict):
        self.phrases = sorted(cfg.get("symptoms", []), key=len, reverse=True)
        self.typo_map = cfg.get("typo_map", {}) or {}

    @classmethod
    def load(cls, path=None) -> "SymptomExtractor":
        path = path or (DATA_DIR / "symptoms_vi.yaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def extract(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        for ph in self.phrases:
            for s, e in find_phrase_spans(text, ph):
                ents.append(Entity(text[s:e], s, e, TYPE_SYMPTOM, source="symptom"))
        for wrong in self.typo_map:
            for s, e in find_phrase_spans(text, wrong):
                ents.append(Entity(text[s:e], s, e, TYPE_SYMPTOM, source="symptom:typo"))
        return ents
