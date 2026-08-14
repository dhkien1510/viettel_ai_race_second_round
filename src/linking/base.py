"""Linker interface + a fuzzy dictionary fast-path used by both ICD-10 and
RxNorm linkers. Embedding retrieval is a documented extension point."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from rapidfuzz import process, fuzz
    _HAVE_RAPIDFUZZ = True
except Exception:  # pragma: no cover - optional at runtime
    _HAVE_RAPIDFUZZ = False


class Linker:
    """Maps a surface string to a ranked list of standard codes."""

    def __init__(self, surface_to_codes: Dict[str, List[str]], threshold: int = 88):
        # keys are casefolded surfaces
        self.map = {k.casefold(): v for k, v in surface_to_codes.items()}
        self.keys = list(self.map.keys())
        self.threshold = threshold

    def lookup(self, surface: str, canonical: str = "") -> List[str]:
        # 1) exact canonical/surface hit
        for probe in (canonical, surface):
            if probe and probe.casefold() in self.map:
                return list(self.map[probe.casefold()])
        # 2) fuzzy fallback (typo tolerance)
        if _HAVE_RAPIDFUZZ and self.keys:
            probe = (canonical or surface).casefold()
            best = process.extractOne(probe, self.keys, scorer=fuzz.WRatio)
            if best and best[1] >= self.threshold:
                return list(self.map[best[0]])
        return []

    # Extension point: embedding retrieval over the full KB.
    def lookup_embedding(self, surface: str) -> List[str]:  # pragma: no cover
        raise NotImplementedError(
            "Wire a multilingual embedding index (LaBSE / multilingual-e5) over "
            "the ICD-10-CM / RxNorm knowledge base here."
        )
