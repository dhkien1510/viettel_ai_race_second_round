"""Diagnosis extractor (CHẨN_ĐOÁN).

Dictionary substring matching with word-boundary guards. Each surface carries a
seed ICD-10 list on the Entity.canonical/candidates for later linking; whether
those codes are actually emitted is controlled by the EMIT_CANDIDATES flag in
infer.py (currently off -> candidates=[]).

Some surfaces (flagged `imaging_finding: true`) are ambiguous: they read as a
diagnosis in an ASSESSMENT context ("chẩn đoán: tràn dịch màng phổi") but as a
plain imaging/lab FINDING when they appear under a LAB/IMAGING section
("kết quả CT ngực: tràn dịch màng phổi hai bên"). `Pipeline` downgrades these
to KẾT_QUẢ_XÉT_NGHIỆM when the section context says so (see infer.py).
"""

from __future__ import annotations

from typing import Dict, List, Set

import yaml

from ..schema import DATA_DIR, Entity, TYPE_DIAGNOSIS
from ..offsets import find_phrase_spans


class DiagnosisExtractor:
    def __init__(self, cfg: dict):
        items = cfg.get("diagnoses", [])
        self.icd_by_surface: Dict[str, List[str]] = {}
        self.imaging_finding_surfaces: Set[str] = set()
        surfaces = []
        for it in items:
            surf = it["surface"]
            surfaces.append(surf)
            self.icd_by_surface[surf.casefold()] = list(it.get("icd", []))
            if it.get("imaging_finding"):
                self.imaging_finding_surfaces.add(surf.casefold())
        self.surfaces = sorted(surfaces, key=len, reverse=True)

    @classmethod
    def load(cls, path=None) -> "DiagnosisExtractor":
        path = path or (DATA_DIR / "diagnoses_vi_icd.yaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def extract(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        for surf in self.surfaces:
            icd = self.icd_by_surface.get(surf.casefold(), [])
            for s, e in find_phrase_spans(text, surf):
                ent = Entity(text[s:e], s, e, TYPE_DIAGNOSIS, source="diagnosis",
                             canonical=surf.casefold())
                # stash seed ICD on the entity; linker decides whether to emit
                ent.candidates = list(icd)
                ents.append(ent)
        return ents
