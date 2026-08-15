"""Build a sentence-embedding index over RxNorm concepts using MULTI-VIEW average pooling.

Each RXCUI is now represented by MULTIPLE text views (STR, brand names, synonyms,
normalized variants) instead of a single representative string. Embeddings are created
by average pooling all view embeddings, then normalizing the result.

Run once (or whenever the RRF/index changes):
    python scripts/build_rxnorm_index.py --build-embeddings
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .build_index import CACHE_PATH as CONSO_CACHE_PATH, TTY_PRIORITY
from .config import CONFIG

MODEL_NAME = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
EMBED_CACHE_PATH = Path("data/rxnorm/cache/rxnorm_embed.npz")


def _pick_representatives(entries) -> list:
    """One row per RXCUI: the string from its highest-priority TTY (SCD > PSN > SBD > SY > ...).
    Kept for backward compatibility but NOT used for multi-view pooling anymore.
    """
    best: dict[str, object] = {}
    for e in entries:
        cur = best.get(e.rxcui)
        if cur is None or TTY_PRIORITY.get(e.tty, 99) < TTY_PRIORITY.get(cur.tty, 99):
            best[e.rxcui] = e
    return list(best.values())


def build(conso_cache_path: Path = CONSO_CACHE_PATH, out_path: Path = EMBED_CACHE_PATH) -> None:
    from sentence_transformers import SentenceTransformer

    with conso_cache_path.open("rb") as f:
        data = pickle.load(f)

    # Check cache version compatibility
    cached_version = data.get("version", "unknown")
    if cached_version != CONFIG.cache_version:
        raise ValueError(
            f"Cache version mismatch: expected {CONFIG.cache_version}, got {cached_version}. "
            f"Please rebuild the index with: python scripts/build_rxnorm_index.py"
        )

    entries = data["entries"]
    rxcui_views = data.get("rxcui_views", {})

    if not rxcui_views:
        # Fallback to old single-representative mode if no views available
        reps = _pick_representatives(entries)
        print(f"No multi-view data found, falling back to single-representative mode for {len(reps)} concepts.")
        strings_by_rxcui = {e.rxcui: [e.str_] for e in reps}
    else:
        # Collect unique views across all RXCUIs for batch encoding
        strings_by_rxcui = {}
        all_unique_views: set[str] = set()
        for rxcui, views in rxcui_views.items():
            view_texts = [v.text for v in views if len(v.text.strip()) >= CONFIG.min_view_length]
            if not view_texts:
                # If no valid views, use the entry STR as fallback
                rxcui_entries = [e for e in entries if e.rxcui == rxcui]
                if rxcui_entries:
                    view_texts = [rxcui_entries[0].str_]
                else:
                    view_texts = [""]
            strings_by_rxcui[rxcui] = view_texts
            all_unique_views.update(view_texts)

        print(f"Multi-view corpus: {len(rxcui_views)} RXCUIs, {len(all_unique_views)} unique views")

    model = SentenceTransformer(MODEL_NAME)

    # Encode all unique views in batches
    unique_view_list = sorted(all_unique_views)
    print(f"Encoding {len(unique_view_list)} unique views...")
    vectors = model.encode(
        unique_view_list, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    # Build view text -> vector index
    view_to_vec: dict[str, np.ndarray] = {}
    for i, v in enumerate(unique_view_list):
        view_to_vec[v] = vectors[i]

    # Average pool per RXCUI and compute representative strings
    rxcuis_list: list[str] = []
    pooled_vectors: list[np.ndarray] = []
    rep_strings: list[str] = []

    for rxcui in sorted(strings_by_rxcui.keys()):
        views = strings_by_rxcui[rxcui]
        vecs = [view_to_vec[v] for v in views if v in view_to_vec]

        if not vecs:
            continue

        # Average pooling (already normalized, so simple average works)
        avg_vec = np.mean(vecs, axis=0)
        # Re-normalize the averaged vector
        norm = np.linalg.norm(avg_vec)
        if norm > 1e-8:
            avg_vec = avg_vec / norm

        rxcuis_list.append(rxcui)
        pooled_vectors.append(avg_vec)

        # Representative string: use the highest-priority view text
        rep_strings.append(views[0])

    result = np.stack(pooled_vectors, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, vectors=result, rxcuis=np.array(rxcuis_list), strings=np.array(rep_strings)
    )
    print(f"Saved {result.shape} multi-view embeddings -> {out_path}")


class EmbedIndex:
    """Loaded lazily by RxNormLinker -- only touched once tiers 1-3 have already failed a span."""

    def __init__(
        self,
        path: Path = EMBED_CACHE_PATH,
        model_name: str = MODEL_NAME,
        preferred_entries: dict | None = None,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run `python scripts/build_rxnorm_index.py --build-embeddings` once first"
            )
        data = np.load(path, allow_pickle=False)
        # The compressed cache's fixed-width Unicode `strings` array expands
        # to >2 GB for this snapshot.  The linker already has preferred
        # strings in its lexical cache, so do not materialize that array.
        # float16 also halves the 703 MB vector matrix while preserving more
        # than enough precision for candidate retrieval.
        self.vectors = data["vectors"].astype(np.float16)
        self.rxcuis = data["rxcuis"]
        self.preferred_entries = preferred_entries or {}
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def search(self, text: str, top_k: int = 3) -> list[tuple[str, str, float]]:
        """Search by query text. Returns [(rxcui, str_, cosine_sim), ...]."""
        vec = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vectors @ vec.astype(np.float16)
        top_idx = np.argsort(-sims)[:top_k]
        return [
            (
                str(self.rxcuis[i]),
                getattr(self.preferred_entries.get(str(self.rxcuis[i])), "str_", str(self.rxcuis[i])),
                float(sims[i]),
            )
            for i in top_idx
        ]


if __name__ == "__main__":
    build()
