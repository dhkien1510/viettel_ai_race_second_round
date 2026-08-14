from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- Multi-view corpus (per RXCUI) ----------------------------------
    max_views_per_rxcui: int = 20
    min_view_length: int = 2
    max_view_length: int = 500

    # ---- Query expansion -------------------------------------------------
    max_query_expansions: int = 8

    # ---- Candidate pool & reranking --------------------------------------
    max_candidate_pool: int = 100
    lexical_weight: float = 0.6
    embedding_weight: float = 0.4
    exact_match_bonus: float = 0.15
    token_overlap_bonus: float = 0.05

    # Penalty cho unexplained INGREDIENT tokens (e.g. combo product has extra ingredient)
    # Should be heavy — a different ingredient is a real mismatch
    unexplained_ingredient_penalty: float = 0.4

    # Penalty cho unexplained FORM/SALT VARIANT tokens (e.g. "Chewable Tablet")
    # Should be light — just a variant, not a fundamentally different drug
    form_mismatch_penalty: float = 0.08
    salt_mismatch_penalty: float = 0.05

    strength_mismatch_penalty: float = 0.3
    min_lexical_score: float = 0.35
    min_embedding_similarity: float = 0.60
    collapse_threshold: float = 0.75
    top_k: int = 3

    # Penalty for RxNorm entries flagged SUPPRESS not in ("", "N") (editorially
    # suppressed/obsolete duplicates, e.g. RXCUI 866439 "metoprolol succinate
    # 50 MG Extended Release Oral Tablet" duplicates the live 866436 "24 HR
    # metoprolol succinate 50 MG..." -- BTC's own official example picks the
    # non-suppressed one even though it has an extra unmatched "24 HR" token).
    # Large enough to break near-ties in favor of the non-suppressed entry,
    # small enough that a suppressed entry can still win if it's the only match.
    suppressed_entry_penalty: float = 0.2

    # ---- Cross-encoder reranking (tier 3 refinement) ----------------------
    # bge-reranker-v2-m3 re-scores the top N hybrid candidates against the raw
    # query text. It is a cross-encoder (query+candidate pair -> relevance),
    # not a bi-encoder, so it cannot replace the SapBERT retrieval index --
    # it only refines the ordering of candidates SapBERT already retrieved.
    enable_cross_encoder_rerank: bool = True
    rerank_pool_size: int = 20
    rerank_weight: float = 0.5

    # If the best cross-encoder relevance score (sigmoid, 0..1) in the pool is
    # below this, none of the retrieved candidates are actually relevant --
    # return empty instead of forcing top_k guesses. This is the same insight
    # behind the conservative candidate policy
    # ("wrong candidates hurt more than an empty prediction"), applied as
    # a principled gate on the reranker's own confidence instead of a
    # hand-tuned side script.
    min_rerank_confidence: float = 0.5

    # If the top candidate's raw cross-encoder score clears this (higher bar
    # than min_rerank_confidence), trust it alone instead of returning the
    # whole top_k pool. Mirrors _finalize()'s lexical-tier collapse, but keyed
    # off the reranker's own sigmoid score rather than the blended hybrid
    # score, since that's a cleaner, better-calibrated confidence signal.
    # Evidenced by a real regression: without this, a case where the
    # reranked #1 candidate was already correct (Laxis -> furosemide 20mg)
    # still returned 2 extra wrong candidates (wrong dose form + an unrelated
    # drug), hurting Jaccard against a ground truth of size 1.
    tier3_collapse_confidence: float = 0.8

    # ---- Cache versioning ------------------------------------------------
    cache_version: str = "mv-hybrid-v2"


CONFIG = Config()
