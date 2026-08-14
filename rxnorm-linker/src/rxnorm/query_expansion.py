"""Query expansion for embedding-based RxNorm linking.

Before encoding a query text with SapBERT, expand it with multiple views:
1. Raw query (original text)
2. Lowercase variant
3. Accent-stripped variant
4. Alias-expanded variant (using normalize.INGREDIENT_ALIASES)
5. Normalized token variant (stopwords removed, tokens joined)
6. Synonym-expanded variant (if any token matches a known RXCUI view)

This produces a "multi-view" corpus per query that captures different ways
the same concept can be expressed, improving embedding quality via average pooling.

All expansions are capped and deduped per config.
"""
from __future__ import annotations

import re
import unicodedata

from .config import CONFIG
from .normalize import INGREDIENT_ALIASES, PHRASE_ALIASES, _strip_accents


def _normalize_view(text: str) -> str:
    """Apply standard normalization to a query view."""
    return text.strip().lower()


def expand_query(
    text: str,
    alias_map: dict[str, str] | None = None,
    synonym_lookup: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    """Expand a raw query into multiple views with source tags.

    Args:
        text: The raw query string from NER span.
        alias_map: Optional custom alias map (defaults to INGREDIENT_ALIASES).
        synonym_lookup: Optional mapping from token -> list of known synonyms.

    Returns:
        List of dicts with keys "text" and "source", e.g.:
        [{"text": "atenolol", "source": "raw"}, ...]
    """
    if alias_map is None:
        alias_map = INGREDIENT_ALIASES
    if synonym_lookup is None:
        synonym_lookup = {}

    views: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_view(text: str, source: str):
        normalized = _normalize_view(text)
        if not normalized or len(normalized) < CONFIG.min_view_length:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        if len(views) >= CONFIG.max_query_expansions:
            return
        views.append({"text": normalized, "source": source})

    # 1. Raw query (highest priority)
    add_view(text, "raw")

    # 2. Lowercase variant
    add_view(text.lower(), "lowercase")

    # 3. Accent-stripped variant
    stripped = _strip_accents(text)
    if stripped.lower() != text.lower():
        add_view(stripped.lower(), "accent_stripped")

    # 4. Alias-expanded variant: replace colloquial names with RxNorm terms
    lowered = _strip_accents(text.lower())
    for phrase, replacement in PHRASE_ALIASES.items():
        lowered = re.sub(rf"\b{re.escape(phrase)}\b", replacement, lowered)
    aliased_parts = []
    for word in lowered.split():
        expanded = alias_map.get(word, word)
        aliased_parts.append(expanded)
    aliased_text = " ".join(aliased_parts)
    if aliased_text != lowered:
        add_view(aliased_text, "alias_expanded")

    # 5. Normalized token variant: remove stopwords, join remaining tokens
    from .normalize import _STOPWORDS, _WORD_RE
    words = _WORD_RE.findall(_strip_accents(lowered))
    filtered_tokens = [w for w in words if w not in _STOPWORDS]
    if filtered_tokens:
        normalized_token_text = " ".join(filtered_tokens)
        if normalized_token_text != lowered:
            add_view(normalized_token_text, "normalized_tokens")

    # 6. Synonym-expanded variant: look up each token in synonym map
    if synonym_lookup:
        synonym_candidates = []
        for token in filtered_tokens:
            syns = synonym_lookup.get(token, [])
            synonym_candidates.extend(syns)
        if synonym_candidates:
            synonym_text = " ".join(set(synonym_candidates))
            add_view(synonym_text, "synonym_expanded")

    # Cap to max expansions
    return views[:CONFIG.max_query_expansions]


def average_pool_query_views(
    model, texts: list[str], normalize_embeddings: bool = True
) -> np.ndarray:  # type: ignore[name-defined]
    """Encode multiple query views and average-pool them into a single vector.

    Args:
        model: A sentence-transformers model.
        texts: List of view texts to encode.
        normalize_embeddings: Whether to normalize individual view embeddings before pooling.

    Returns:
        A single normalized vector (1-D numpy array).
    """
    import numpy as np  # type: ignore[import-not-found]

    if not texts:
        raise ValueError("Cannot pool empty view list")

    vectors = model.encode(
        texts, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=normalize_embeddings,
    ).astype(np.float32)

    # Average pool across views
    pooled = np.mean(vectors, axis=0)

    # Re-normalize the pooled vector
    if normalize_embeddings:
        norm = np.linalg.norm(pooled)
        if norm > 1e-8:
            pooled = pooled / norm

    return pooled


# Backward-compatible function for legacy callers
def expand_and_pool(model, text: str, **kwargs) -> tuple[np.ndarray, list[str]]:  # type: ignore[name-defined]
    """Legacy wrapper: expand query and return pooled embedding + view texts.

    Args:
        model: A sentence-transformers model.
        text: Raw query string.

    Returns:
        (pooled_vector, view_texts) tuple.
    """
    views = expand_query(text, **kwargs)
    view_texts = [v["text"] for v in views]
    vector = average_pool_query_views(model, view_texts)
    return vector, view_texts
