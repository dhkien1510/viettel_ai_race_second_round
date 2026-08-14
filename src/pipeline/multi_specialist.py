"""Multi-specialist NER pipeline with genre-based routing.

Holds multiple Pipeline instances (one per genre) and routes input files
to the appropriate specialist based on genre classification.

Usage:
    pipe = MultiSpecialistPipeline()
    pipe.load_specialist("HOSPITAL", "models/xlmr_large_hospital")
    pipe.load_specialist("Q&A_FULL", "models/xlmr_large_qa")
    ...
    entities = pipe.process_text(file_id, text)  # auto-routes by genre
"""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from ..infer import Pipeline
from ..schema import Entity
from .genre_router import GenreRouter, GenreType


class MultiSpecialistPipeline:
    """Orchestrates multiple genre-specialist pipelines with automatic routing.

    Each genre has its own NER model (or can fall back to a default baseline).
    When a file is processed, its genre is determined → corresponding model
    is selected → inference runs → output is emitted.

    Initially, you can load a single "baseline" model for all genres, then
    gradually replace with specialists as they become available.
    """

    def __init__(self, baseline_model: Optional[str] = None,
                 emit_candidates: Optional[bool] = None,
                 max_len: int = 256, stride: int = 64,
                 rules_on: bool = True, raw: bool = False):
        """Init multi-specialist pipeline with optional baseline fallback.

        Args:
            baseline_model: Registry key or checkpoint dir. If set, all genres
                will use this model until explicitly overridden with
                load_specialist().
            emit_candidates, max_len, stride, rules_on, raw: Passed to
                Pipeline constructor for all loaded models.
        """
        self.router = GenreRouter()
        self.specialists: dict[GenreType, Pipeline] = {}
        self.baseline_model = baseline_model
        self.pipeline_kwargs = {
            "emit_candidates": emit_candidates,
            "max_len": max_len,
            "stride": stride,
            "rules_on": rules_on,
            "raw": raw,
        }

        # If baseline is provided, pre-load it for all genres
        if baseline_model:
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline model for all known genres."""
        print(f"[MultiSpecialistPipeline] Loading baseline model: {self.baseline_model}")
        for genre in ["Q&A_FULL", "HOSPITAL", "HYBRID_QA", "FAQ", "HYBRID_CONSULT"]:
            self.specialists[genre] = Pipeline(model=self.baseline_model, **self.pipeline_kwargs)
        print(f"[MultiSpecialistPipeline] Baseline loaded for all 5 genres")

    def load_specialist(self, genre: GenreType, model: str) -> None:
        """Load or replace a specialist model for a specific genre.

        Args:
            genre: One of "Q&A_FULL", "HOSPITAL", "HYBRID_QA", "FAQ", "HYBRID_CONSULT"
            model: Registry key or checkpoint dir
        """
        print(f"[MultiSpecialistPipeline] Loading specialist for {genre}: {model}")
        self.specialists[genre] = Pipeline(model=model, **self.pipeline_kwargs)
        print(f"[MultiSpecialistPipeline] Specialist for {genre} loaded")

    def _get_pipeline(self, file_id: int) -> Pipeline:
        """Get the appropriate Pipeline for a file_id, or raise error if not loaded."""
        genre = self.router.classify(file_id)
        if genre not in self.specialists:
            raise RuntimeError(
                f"No pipeline loaded for genre {genre} (file {file_id}). "
                f"Load a baseline or specialist first via load_specialist()."
            )
        return self.specialists[genre]

    def process_text(self, file_id: int, text: str) -> List[dict]:
        """Process a single file, routing to appropriate specialist.

        Args:
            file_id: File number (used to classify genre)
            text: Input text to process

        Returns:
            List of entity dicts (as per Pipeline.process_text)
        """
        genre = self.router.classify(file_id)
        pipe = self._get_pipeline(file_id)
        entities = pipe.extract_entities(text)
        return [e.to_dict() for e in entities]

    def extract_entities(self, file_id: int, text: str) -> List[Entity]:
        """Extract entities for a single file (returns Entity objects, not dicts)."""
        pipe = self._get_pipeline(file_id)
        return pipe.extract_entities(text)

    def stats(self) -> dict:
        """Return genre distribution and loaded model status."""
        dist = self.router.stats()
        loaded = {genre: True for genre in self.specialists.keys()}
        return {
            "genre_distribution": dist,
            "loaded_specialists": loaded,
            "baseline_model": self.baseline_model,
        }

    def summary(self) -> str:
        """Human-readable status summary."""
        dist = self.router.stats()
        lines = [
            "[MultiSpecialistPipeline] Genre distribution (round-2 test set):",
            f"  Q&A_FULL:      {dist.get('Q&A_FULL', 0):3d} files",
            f"  HOSPITAL:      {dist.get('HOSPITAL', 0):3d} files",
            f"  HYBRID_QA:     {dist.get('HYBRID_QA', 0):3d} files",
            f"  FAQ:           {dist.get('FAQ', 0):3d} files",
            f"  HYBRID_CONSULT: {dist.get('HYBRID_CONSULT', 0):3d} files",
            f"  Total:         {sum(dist.values()):3d} files",
            "",
            "[MultiSpecialistPipeline] Loaded specialists:",
        ]
        for genre in ["Q&A_FULL", "HOSPITAL", "HYBRID_QA", "FAQ", "HYBRID_CONSULT"]:
            status = "✓" if genre in self.specialists else "✗"
            lines.append(f"  {status} {genre}")
        return "\n".join(lines)
