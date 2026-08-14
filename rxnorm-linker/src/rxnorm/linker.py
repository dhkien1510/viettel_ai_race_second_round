from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .build_index import CACHE_PATH, Entry, build
from .config import CONFIG
from .normalize import (
    FORM_VARIANT_TOKENS,
    INGREDIENT_ALIASES,
    NEUTRAL_TOKENS,
    SALT_VARIANT_TOKENS,
    ParsedSpan,
    parse_span,
    NON_LINKABLE_CLASS_TERMS,
)
from .query_expansion import expand_query


@dataclass
class Candidate:
    rxcui: str
    tty: str
    str_: str
    score: float
    method: str = "lexical"  # "lexical" (tiers 1-2) | "embedding" (tier 3) | "hybrid" | "reranked" | "exact" | "nearest_dose"


class RxNormLinker:
    def __init__(self, cache_path: Path = CACHE_PATH):
        if not cache_path.exists():
            build(cache_path=cache_path)
        with cache_path.open("rb") as f:
            data = pickle.load(f)

        # Check cache version compatibility
        cached_version = data.get("version", None)
        if cached_version is None:
            print(f"WARNING: Cache at {cache_path} has no version field (old format). "
                  f"Consider rebuilding with: python scripts/build_rxnorm_index.py")
            data["rxcui_views"] = {}
        elif cached_version != CONFIG.cache_version:
            raise ValueError(
                f"Cache version mismatch: expected {CONFIG.cache_version}, got {cached_version}. "
                f"Please rebuild the index with: python scripts/build_rxnorm_index.py"
            )

        self.entries: list[Entry] = data["entries"]
        self.token_index: dict[str, list[int]] = data["token_index"]
        self.tty_priority: dict[str, int] = data["tty_priority"]
        self.rxcui_views: dict[str, list] = data.get("rxcui_views", {})
        self.generic_ingredients: set[str] = {
            tok for e in self.entries if e.tty in ("IN", "PIN", "MIN") for tok in e.tokens
        }
        self.brand_names: set[str] = {
            e.str_.lower().strip() for e in self.entries if e.tty == "BN"
        }
        from collections import defaultdict
        self.rxcui_to_entry_idxs = defaultdict(list)
        for i, e in enumerate(self.entries):
            self.rxcui_to_entry_idxs[e.rxcui].append(i)
        self._embed_index = None  # lazy: only load the model if tier 3 fires
        self._reranker = None  # lazy: only load the cross-encoder if tier 3 fires

    # -- scoring -----------------------------------------------------
    def _strength_score(self, query: ParsedSpan, entry: Entry) -> float | None:
        """None means 'query gave no strength, do not penalize'; else 0..1."""
        if not query.strengths:
            return None
        if entry.tty in _BARE_TTY:
            return None
        if not entry.strengths:
            return 0.0

        def canonical_strength(value: float, unit: str) -> tuple[float, str]:
            u = unit.upper()
            if u == "G":
                return value * 1000.0, "MG"
            if u == "MCG":
                return value / 1000.0, "MG"
            if u == "L":
                return value * 1000.0, "ML"
            return value, u

        q_can = [canonical_strength(v, u) for v, u in query.strengths]
        e_can = [canonical_strength(v, u) for v, u in entry.strengths]

        best = 0.0
        for qv, qu in q_can:
            for ev, eu in e_can:
                if qu == eu:
                    # Symmetric in the absolute gap only
                    closeness = 1 / (1 + abs(qv - ev))
                    best = max(best, closeness)
        return best if best > 0 else 0.0

    def _score_entry(self, query: ParsedSpan, idx: int) -> float:
        entry = self.entries[idx]
        
        # 0. Route hard constraints check
        raw_lower = query.raw.lower()
        is_iv = any(w in raw_lower for w in ["iv", "tiem", "tiêm", "truyen", "truyền", "tĩnh mạch", "tinh mach", "intravenous", "infusion", "injection", "injectable"])
        is_oral = any(w in raw_lower for w in ["po", "uong", "uống", "oral", "tablet", "capsule", "viên", "vien"])
        
        entry_str_lower = entry.str_.lower()
        has_oral = any(w in entry_str_lower for w in ["oral", "tablet", "capsule"])
        has_iv = any(w in entry_str_lower for w in ["injection", "intravenous", "infusion", "injectable"])
        
        if is_iv and has_oral:
            return 0.0
        if is_oral and has_iv:
            return 0.0

        exclude = (SALT_VARIANT_TOKENS | FORM_VARIANT_TOKENS | NEUTRAL_TOKENS) - {
            "sodium", "potassium", "calcium", "magnesium"
        }
        q_active = {tok for tok in set(query.ingredient_tokens) - exclude if len(tok) >= 2}
        e_active = {tok for tok in (set(entry.tokens) - exclude) & self.generic_ingredients if len(tok) >= 2}
        
        # 1. Mandatory active ingredient overlap check
        if q_active and e_active and not (q_active & e_active):
            return 0.0
            
        # 2. Partial ingredient match penalty (only for dosed entries)
        if entry.tty not in _BARE_TTY:
            if len(q_active) > 1 and len(e_active) < len(q_active):
                return 0.0
  
        q_tokens = set(query.all_tokens)
        if entry.tty in _BARE_TTY:
            exclude_trim = FORM_VARIANT_TOKENS | (SALT_VARIANT_TOKENS - {"sodium", "potassium", "calcium", "magnesium"})
            trimmed = q_tokens - exclude_trim
            if trimmed:
                q_tokens = trimmed
        e_tokens = set(entry.tokens)
        if not q_tokens or not e_tokens:
            containment = 0.0
        else:
            containment = len(q_tokens & e_tokens) / len(q_tokens)

        unexplained = e_tokens - q_tokens - NEUTRAL_TOKENS
        unexplained_variant = unexplained & (FORM_VARIANT_TOKENS | SALT_VARIANT_TOKENS)
        unexplained_ingredient = unexplained - unexplained_variant
        containment = max(0.0, containment
                           - CONFIG.unexplained_ingredient_penalty * len(unexplained_ingredient)
                           - CONFIG.form_mismatch_penalty * len(unexplained_variant))

        strength = self._strength_score(query, entry)
        
        # If the entry has unexplained generic ingredients, then any strength match
        # is a false positive cross-contamination from a different ingredient's strength.
        if strength is not None:
            unexplained_generic = unexplained_ingredient & self.generic_ingredients
            if unexplained_generic:
                strength = 0.0

        if strength is None:
            score = containment
        else:
            if strength == 0.0:
                return 0.0
            score = 0.5 * containment + 0.5 * strength

        # 4. Unexplained generic active ingredient penalty
        unexplained_generic = unexplained_ingredient & self.generic_ingredients
        if unexplained_generic:
            score -= 0.50

        # 3. Exact combination ingredient match bonus for MIN concepts
        if entry.tty == "MIN" and q_active == e_active and len(q_active) > 1:
            score += 0.20
        score = min(score, 1.0)

        # Determine TTY priority dynamically based on strategy and brand/generic query type
        strategy = getattr(self, "current_strategy", "most_specific")
        is_brand = getattr(self, "current_is_brand_query", False)
        
        if strategy == "ingredient_only":
            if entry.tty not in ("IN", "PIN", "MIN"):
                return 0.0
            tty_priority = {"MIN": 1, "IN": 2, "PIN": 3}
        elif strategy == "generic_only":
            if entry.tty in ("SBD", "SBDG", "SBDF", "BN"):
                return 0.0
            tty_priority = {"SCD": 1, "SCDG": 2, "SCDF": 3, "IN": 4, "MIN": 5, "PIN": 6}
        elif strategy == "surface_form":
            if is_brand:
                tty_priority = {"SBD": 1, "SBDG": 2, "SBDF": 3, "BN": 4, "SCD": 5, "SCDG": 6, "SCDF": 7, "MIN": 8, "PIN": 9, "IN": 10}
            else:
                tty_priority = {"SCD": 1, "SCDG": 2, "SCDF": 3, "IN": 4, "MIN": 5, "PIN": 6, "SBD": 7, "SBDG": 8, "SBDF": 9, "BN": 10}
        else:
            # most_specific or conservative
            tty_priority = self.tty_priority

        if strategy == "conservative" and entry.tty not in ("IN", "PIN", "MIN"):
            if score < 0.95:
                score -= 0.15

        # small nudge toward RxNorm's own "preferred" representations
        score += (20 - tty_priority.get(entry.tty, 20)) * 0.001

        # Editorially suppressed duplicate concepts penalty
        if entry.suppress not in ("", "N"):
            score -= CONFIG.suppressed_entry_penalty
        return score

    # -- token overlap for hybrid scoring ----------------------------
    def _token_overlap(self, query: ParsedSpan, entry: Entry) -> float:
        """Compute token overlap / containment between query and entry."""
        q_tokens = set(query.all_tokens)
        e_tokens = set(entry.tokens)
        if not q_tokens or not e_tokens:
            return 0.0
        return len(q_tokens & e_tokens) / len(q_tokens)

    def _exact_match_bonus(self, query: ParsedSpan, entry: Entry) -> float:
        """Check if query tokens exactly match entry tokens (with some tolerance)."""
        q_tokens = set(t.lower() for t in query.all_tokens)
        e_tokens = set(t.lower() for t in entry.tokens)
        if not q_tokens or not e_tokens:
            return 0.0
        if q_tokens.issubset(e_tokens) and len(q_tokens) >= len(e_tokens) * 0.7:
            return CONFIG.exact_match_bonus
        return 0.0

    # -- candidate generation -----------------------------------------
    def _candidate_idxs(self, query: ParsedSpan, ttys: set[str]) -> set[int]:
        strategy = getattr(self, "current_strategy", "most_specific")
        allowed_ttys = ttys
        if strategy == "ingredient_only":
            allowed_ttys = ttys & _INGREDIENT_TTY
            if not allowed_ttys:
                return set()
        elif strategy == "generic_only":
            allowed_ttys = ttys - {"SBD", "SBDG", "SBDF", "BN"}
            if not allowed_ttys:
                return set()
                
        idxs: set[int] = set()
        for tok in query.ingredient_tokens:
            for i in self.token_index.get(tok, ()):
                if self.entries[i].tty in allowed_ttys:
                    idxs.add(i)
        return idxs

    def _rank(self, query: ParsedSpan, idxs: set[int]) -> list[Candidate]:
        scored = [
            Candidate(self.entries[i].rxcui, self.entries[i].tty, self.entries[i].str_,
                      self._score_entry(query, i))
            for i in idxs
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        # de-dupe by rxcui, keep best-scoring representation of each concept
        seen: dict[str, Candidate] = {}
        for c in scored:
            if c.rxcui not in seen or c.score > seen[c.rxcui].score:
                seen[c.rxcui] = c
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    @staticmethod
    def _finalize(ranked: list[Candidate], top_k: int, allow_collapse: bool = True) -> list[Candidate]:
        """Above HIGH_CONF, a single top-scoring candidate is trusted alone;
        otherwise return the top-k spread. `allow_collapse=False` is only
        for a dose *range* ("325-650 mg"), which names two distinct RxNorm
        concepts on purpose -- collapsing to one would be wrong there even at high confidence.
        """
        threshold = CONFIG.collapse_threshold
        if allow_collapse and ranked and ranked[0].score >= threshold:
            return ranked[:1]
        return ranked[:top_k]

    def _has_exact_strength_match(self, query: ParsedSpan, idxs: set[int]) -> bool:
        q_strengths = set(query.strengths)
        return any(q_strengths & set(self.entries[i].strengths) for i in idxs)

    def _nearest_dose(
        self, query: ParsedSpan, ranked: list[Candidate], idxs: set[int]
    ) -> Candidate | None:
        """Among already-ranked candidates, pick the single nearest
        marketed strength by absolute difference -- ties go to the lower
        dose, then to the non-suppressed entry.
        """
        ranked_rxcuis = {c.rxcui for c in ranked}
        best_idx = None
        best_key = None
        for i in idxs:
            entry = self.entries[i]
            if entry.rxcui not in ranked_rxcuis or not entry.strengths:
                continue
            for qv, qu in query.strengths:
                for ev, eu in entry.strengths:
                    if eu != qu:
                        continue
                    suppressed = 0 if entry.suppress in ("", "N") else 1
                    key = (abs(ev - qv), ev, suppressed, self.tty_priority.get(entry.tty, 20))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
        if best_idx is None:
            return None
        e = self.entries[best_idx]
        return Candidate(e.rxcui, e.tty, e.str_, 1.0, method="nearest_dose")

    # -- public API -----------------------------------------------------
    def link(
        self,
        text: str,
        top_k: int | None = None,
        use_tier3: bool = True,
        dose_fallback: str = "nearest",
        context: str | None = None,
        strategy: str | None = None,
    ) -> list[Candidate]:
        if not text or not text.strip():
            return []
        if top_k is None:
            top_k = CONFIG.top_k

        old_strategy = getattr(self, "current_strategy", "most_specific")
        old_is_brand = getattr(self, "current_is_brand_query", False)

        if strategy is None:
            strategy = old_strategy
        self.current_strategy = strategy
        raw_clean = text.lower().strip()
        self.current_is_brand_query = raw_clean in self.brand_names
        if not self.current_is_brand_query:
            query_temp = parse_span(text)
            for tok in query_temp.ingredient_tokens:
                if tok in self.brand_names:
                    self.current_is_brand_query = True
                    break

        try:
            return self._link_impl(text, top_k, use_tier3, dose_fallback, context)
        finally:
            self.current_strategy = old_strategy
            self.current_is_brand_query = old_is_brand

    def _link_impl(
        self,
        text: str,
        top_k: int,
        use_tier3: bool,
        dose_fallback: str,
        context: str | None,
    ) -> list[Candidate]:
        # If context is provided, split definition into sub-queries and merge/resolve them
        if context:
            from .normalize import extract_context_info
            components = extract_context_info(text, context)
            if components:
                all_comps = [text] + components
                sub_candidates = []
                active_ingredients = set()
                exclude = (SALT_VARIANT_TOKENS | FORM_VARIANT_TOKENS | NEUTRAL_TOKENS) - {
                    "sodium", "potassium", "calcium", "magnesium"
                }
                
                # Link each sub-query component
                for comp in all_comps:
                    # Link without context recursively
                    cands = self.link(comp, top_k=5, use_tier3=use_tier3, dose_fallback=dose_fallback)
                    if cands and (cands[0].method != "embedding" or cands[0].score >= 0.75) and cands[0].score >= 0.70:
                        sub_candidates.append(cands)
                        top_rxcui = cands[0].rxcui
                        for entry_idx in self.rxcui_to_entry_idxs.get(top_rxcui, []):
                            entry = self.entries[entry_idx]
                            e_act = (set(entry.tokens) - exclude) & self.generic_ingredients
                            active_ingredients.update(e_act)
                            break
                
                # If multiple active ingredients are resolved, try to match a combination concept
                if len(active_ingredients) > 1:
                    combo_query = " ".join(sorted(active_ingredients))
                    combo_cands = self.link(combo_query, top_k=top_k, use_tier3=use_tier3, dose_fallback=dose_fallback)
                    
                    # Tier 1: Exact combination matching
                    exact_combo = []
                    for c in combo_cands:
                        for entry_idx in self.rxcui_to_entry_idxs.get(c.rxcui, []):
                            entry = self.entries[entry_idx]
                            e_act = (set(entry.tokens) - exclude) & self.generic_ingredients
                            if len(e_act) > 1 and active_ingredients == e_act:
                                exact_combo.append(c)
                                break
                    if exact_combo:
                        return exact_combo
                    
                    # Tier 2: Partial combination matching (only allowed if context indicates hidden components)
                    if "*****" in context:
                        partial_combo = []
                        for c in combo_cands:
                            for entry_idx in self.rxcui_to_entry_idxs.get(c.rxcui, []):
                                entry = self.entries[entry_idx]
                                e_act = (set(entry.tokens) - exclude) & self.generic_ingredients
                                if len(e_act) > 1 and active_ingredients.issubset(e_act):
                                    partial_combo.append(c)
                                    break
                        if partial_combo:
                            return partial_combo
                
                # If no valid combination candidate was found, merge sub-candidates
                seen: dict[str, Candidate] = {}
                for cands in sub_candidates:
                    for c in cands:
                        if c.rxcui not in seen or c.score > seen[c.rxcui].score:
                            seen[c.rxcui] = c
                merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)
                return merged[:top_k] if top_k else merged

        # Normal linking (without context or context matches)
        def tier3(t: str) -> list[Candidate]:
            return self._embedding_fallback(t, top_k) if use_tier3 else []

        query = parse_span(text)
        if query.ingredient_tokens and set(query.ingredient_tokens) <= NON_LINKABLE_CLASS_TERMS:
            return []
        if not query.ingredient_tokens:
            expanded_text = " ".join(query.all_tokens)
            return tier3(expanded_text if expanded_text else text)

        # No strength stated -> ingredient/brand level search first
        if not query.strengths:
            ranked = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
            ranked = [c for c in ranked if c.score >= CONFIG.min_lexical_score]
            if ranked:
                return self._finalize(ranked, top_k)
            expanded_text = " ".join(query.all_tokens)
            return tier3(expanded_text if expanded_text else text)

        # Tier 1+2: token-overlap + strength scoring over every dosed entry.
        dosed_idxs = self._candidate_idxs(query, _DOSED_TTY)
        ranked = self._rank(query, dosed_idxs)
        ranked = [c for c in ranked if c.score >= CONFIG.min_lexical_score]

        is_range = len({v for v, _u in query.strengths}) > 1
        if ranked:
            if not self._has_exact_strength_match(query, dosed_idxs):
                if dose_fallback == "nearest":
                    nearest = self._nearest_dose(query, ranked, dosed_idxs)
                    if nearest is not None:
                        return [nearest]
                elif dose_fallback == "bare":
                    ranked = []
            if ranked:
                return self._finalize(ranked, top_k, allow_collapse=not is_range)

        # Last resort: ingredient is real but exact strength not in RxNorm -> ingredient-level fallback.
        ranked = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
        ranked = [c for c in ranked if c.score >= CONFIG.min_lexical_score]
        if ranked:
            return self._finalize(ranked, top_k, allow_collapse=not is_range)

        # Tier 3 fallback
        expanded_text = " ".join(query.all_tokens)
        return tier3(expanded_text if expanded_text else text)

    def link_exact(self, text: str) -> list[Candidate]:
        """Only return a candidate when the match is exact -- no embedding
        fallback, no partial credit.
        """
        if not text or not text.strip():
            return []
        query = parse_span(text)
        if not query.ingredient_tokens:
            return []
        if set(query.ingredient_tokens) <= NON_LINKABLE_CLASS_TERMS:
            return []

        q_tokens = set(query.all_tokens) - NEUTRAL_TOKENS
        q_strengths = set(query.strengths)
        idxs = self._candidate_idxs(query, _BARE_TTY if not query.strengths else _DOSED_TTY)

        matches: dict[str, Candidate] = {}
        for i in idxs:
            entry = self.entries[i]
            e_tokens = set(entry.tokens) - NEUTRAL_TOKENS
            if e_tokens != q_tokens:
                continue
            if set(entry.strengths) != q_strengths:
                continue
            c = Candidate(entry.rxcui, entry.tty, entry.str_, 1.0, method="exact")
            best = matches.get(entry.rxcui)
            if best is None or self.tty_priority.get(entry.tty, 20) < self.tty_priority.get(best.tty, 20):
                matches[entry.rxcui] = c

        if not matches:
            return []
        best_priority = min(self.tty_priority.get(c.tty, 20) for c in matches.values())
        return [c for c in matches.values() if self.tty_priority.get(c.tty, 20) == best_priority]

    def link_exact_rxcuis(self, text: str) -> list[str]:
        return [c.rxcui for c in self.link_exact(text)]

    # -- tier 3: embedding fallback with hybrid scoring ---------------
    def _get_embed_index(self):
        if self._embed_index is None:
            from .embed_index import EmbedIndex
            self._embed_index = EmbedIndex()
        return self._embed_index

    def _get_reranker(self):
        if self._reranker is None:
            from .rerank import Reranker
            self._reranker = Reranker()
        return self._reranker

    def _embedding_fallback(self, text: str, top_k: int) -> list[Candidate]:
        try:
            index = self._get_embed_index()
        except FileNotFoundError:
            return []

        # Expand query into multiple views before embedding search
        views = expand_query(text)
        view_texts = [v["text"] for v in views] if views else [text.lower()]

        # Search using each view and aggregate scores per RXCUI
        rxcui_scores: dict[str, list[float]] = {}
        rxcui_strs: dict[str, str] = {}

        strategy = getattr(self, "current_strategy", "most_specific")

        for vt in view_texts:
            hits = index.search(vt, top_k=top_k * 5)
            for rxcui, str_, sim in hits:
                if sim >= CONFIG.min_embedding_similarity:
                    entry_idxs = self.rxcui_to_entry_idxs.get(rxcui, [])
                    if not entry_idxs:
                        continue
                    tty = self.entries[entry_idxs[0]].tty
                    
                    if strategy == "ingredient_only" and tty not in ("IN", "PIN", "MIN"):
                        continue
                    if strategy == "generic_only" and tty in ("SBD", "SBDG", "SBDF", "BN"):
                        continue
                        
                    rxcui_scores.setdefault(rxcui, []).append(sim)
                    rxcui_strs[rxcui] = str_

        # Average scores across views for each RXCUI, then pick top-k by avg similarity
        results: list[Candidate] = []
        for rxcui, sims in rxcui_scores.items():
            avg_sim = float(np.mean(sims))
            entry_idxs = self.rxcui_to_entry_idxs.get(rxcui, [])
            tty = self.entries[entry_idxs[0]].tty if entry_idxs else ""
            results.append(Candidate(rxcui=rxcui, tty=tty, str_=rxcui_strs[rxcui], score=avg_sim, method="embedding"))

        results.sort(key=lambda c: c.score, reverse=True)

        # Hybrid rerank: combine lexical signal (token overlap) with embedding similarity
        query = parse_span(text)
        if query.ingredient_tokens and results:
            results = self._hybrid_rerank(query, results[:CONFIG.max_candidate_pool])

        if CONFIG.enable_cross_encoder_rerank and results:
            results = self._cross_encoder_rerank(text, results)

        return results[:top_k]

    def _hybrid_rerank(self, query: ParsedSpan, candidates: list[Candidate]) -> list[Candidate]:
        """Rerank candidates using hybrid scoring: lexical + embedding."""
        scored: list[Candidate] = []

        for c in candidates:
            entry_idxs = self.rxcui_to_entry_idxs.get(c.rxcui, [])
            if entry_idxs:
                lex_score = max(self._score_entry(query, idx) for idx in entry_idxs)
                overlap = max(self._token_overlap(query, self.entries[idx]) for idx in entry_idxs)
                exact_bonus = max(self._exact_match_bonus(query, self.entries[idx]) for idx in entry_idxs) if c.method == "embedding" else 0.0

                embed_score = c.score

                hybrid = (CONFIG.lexical_weight * lex_score +
                          CONFIG.embedding_weight * embed_score +
                          CONFIG.token_overlap_bonus * overlap + exact_bonus)

                c.score = hybrid
                c.method = "hybrid"

            scored.append(c)

        scored.sort(key=lambda c: c.score, reverse=True)

        seen: dict[str, Candidate] = {}
        for c in scored:
            if c.rxcui not in seen or c.score > seen[c.rxcui].score:
                seen[c.rxcui] = c

        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    def _cross_encoder_rerank(self, text: str, candidates: list[Candidate]) -> list[Candidate]:
        """Refine ordering of the top candidates with bge-reranker-v2-m3."""
        pool = candidates[:CONFIG.rerank_pool_size]
        rest = candidates[CONFIG.rerank_pool_size:]

        try:
            reranker = self._get_reranker()
            ce_scores = reranker.score(text, [c.str_ for c in pool])
        except Exception as exc:
            print(f"WARNING: cross-encoder rerank skipped ({exc})")
            return candidates

        if not ce_scores or max(ce_scores) < CONFIG.min_rerank_confidence:
            return []

        paired = sorted(zip(pool, ce_scores), key=lambda p: p[1], reverse=True)

        w = CONFIG.rerank_weight
        for c, ce_score in paired:
            c.score = w * ce_score + (1 - w) * c.score
            c.method = "reranked"

        top_candidate, top_ce_score = paired[0]
        if top_ce_score >= CONFIG.tier3_collapse_confidence:
            return [top_candidate]

        pool = [c for c, _ce in paired]
        pool.sort(key=lambda c: c.score, reverse=True)
        return pool + rest

    def link_rxcuis(
        self,
        text: str,
        top_k: int = CONFIG.top_k,
        use_tier3: bool = True,
        dose_fallback: str = "nearest",
        context: str | None = None,
        strategy: str | None = None,
    ) -> list[str]:
        return [
            c.rxcui
            for c in self.link(
                text,
                top_k=top_k,
                use_tier3=use_tier3,
                dose_fallback=dose_fallback,
                context=context,
                strategy=strategy,
            )
        ]


@lru_cache(maxsize=1)
def get_linker() -> RxNormLinker:
    return RxNormLinker()


# Keep backward-compatible constants that may be imported elsewhere.
MIN_SCORE = CONFIG.min_lexical_score
HIGH_CONF = CONFIG.collapse_threshold
TOP_K = CONFIG.top_k
EMBED_MIN_SIM = CONFIG.min_embedding_similarity
_DOSED_TTY = {"SCD", "SBD", "SCDC", "SBDC", "SCDF", "SBDF", "PSN", "SY", "TMSY", "BN"}
_INGREDIENT_TTY = {"IN", "PIN", "MIN"}
_BARE_TTY = _INGREDIENT_TTY | {"BN"}
