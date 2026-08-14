"""Rule/dictionary backend — the always-on base extractor."""

from __future__ import annotations

from typing import List

from ..schema import Entity
from ..extractors.labs import LabExtractor
from ..extractors.drugs import DrugExtractor
from ..extractors.symptoms import SymptomExtractor
from ..extractors.diagnoses import DiagnosisExtractor
from .base import SpanBackend


class RuleBackend(SpanBackend):
    name = "rules"

    def __init__(self):
        self.labs = LabExtractor.load()
        self.drugs = DrugExtractor.load()
        self.symptoms = SymptomExtractor.load()
        self.diagnoses = DiagnosisExtractor.load()

    def predict(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        ents += self.labs.extract(text)
        ents += self.drugs.extract(text)
        ents += self.symptoms.extract(text)
        ents += self.diagnoses.extract(text)
        return ents
