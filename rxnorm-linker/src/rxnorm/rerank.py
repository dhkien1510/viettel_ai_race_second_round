"""Cross-encoder reranking for tier-3 candidates using BAAI/bge-reranker-v2-m3.

Unlike EmbedIndex (a bi-encoder used to retrieve candidates from the full
RxNorm vocabulary via nearest-neighbor search), a cross-encoder scores each
(query, candidate) pair jointly and cannot build a static vector index. It is
therefore only used to *re-score* the small candidate pool that tier 3 already
retrieved -- never to search the full corpus.

Model is loaded lazily; first use downloads weights from Hugging Face Hub.
"""
from __future__ import annotations

import math

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Loaded lazily by RxNormLinker -- only touched once tier 3 has candidates to refine."""

    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def score(self, query: str, candidates: list[str]) -> list[float]:
        """Return relevance scores in [0, 1] (sigmoid of the raw logit), one per candidate."""
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        raw_scores = self.model.predict(pairs)
        return [1.0 / (1.0 + math.exp(-float(r))) for r in raw_scores]
