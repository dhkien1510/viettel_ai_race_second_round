from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .build_index import CACHE_PATH, Entry, build, source_alias_key
from .config import CONFIG
from .normalize import (
    FORM_VARIANT_TOKENS,
    INGREDIENT_ALIASES,
    NEUTRAL_TOKENS,
    NON_LINKABLE_CLASS_TERMS,
    SALT_VARIANT_TOKENS,
    ParsedSpan,
    parse_span,
)
from .query_expansion import expand_query
from .medication_parser import RxNormVocabularyParser


@dataclass
class Candidate:
    rxcui: str
    tty: str
    str_: str
    score: float
    method: str = "lexical"  # "lexical" (tiers 1-2) | "embedding" (tier 3) | "hybrid"


class RxNormLinker:
    def __init__(self, cache_path: Path = CACHE_PATH):
        if not cache_path.exists():
            build(cache_path=cache_path)
        with cache_path.open("rb") as f:
            data = pickle.load(f)

        # Check cache version compatibility - soft fallback for old caches
        cached_version = data.get("version", None)
        if cached_version is None:
            # Old cache without version field - warn but continue loading
            print(f"WARNING: Cache at {cache_path} has no version field (old format). "
                  f"Consider rebuilding with: python scripts/build_rxnorm_index.py")
            # Add empty views dict for backward compatibility
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
        self.source_alias_index: dict[tuple, tuple[str, tuple[str, ...]]] = data.get(
            "source_alias_index", {}
        )
        self._preferred_entry_by_rxcui: dict[str, Entry] = {}
        for entry in self.entries:
            current = self._preferred_entry_by_rxcui.get(entry.rxcui)
            if current is None or self.tty_priority.get(entry.tty, 20) < self.tty_priority.get(current.tty, 20):
                self._preferred_entry_by_rxcui[entry.rxcui] = entry
        self._embed_index = None  # lazy: only load the model if tier 3 fires
        self._reranker = None  # lazy: only load the cross-encoder if tier 3 fires
        self._medication_parser = RxNormVocabularyParser(self.token_index, len(self.entries))

    # -- scoring -----------------------------------------------------
    def _strength_score(self, query: ParsedSpan, entry: Entry) -> float | None:
        """None means 'query gave no strength, do not penalize'; else 0..1."""
        if not query.strengths:
            return None
        if not entry.strengths:
            return 0.0
        best = 0.0
        for qv, qu in query.strengths:
            for ev, eu in entry.strengths:
                if qu != eu:
                    continue
                if qv == ev:
                    best = max(best, 1.0)
                else:
                    # Symmetric in the absolute gap only -- NOT relative to
                    # (qv+ev)/2, which used to score "1.5 vs 2" higher than
                    # "1.5 vs 1" (same 0.5 gap either way) just because 2 is
                    # the bigger number. That bias picked the wrong neighbor
                    # for BTC's own "clonazepam 1.5 mg" example (gold wants
                    # the 1 MG tablet, not 2 MG).
                    closeness = 1 / (1 + abs(qv - ev))
                    best = max(best, closeness * 0.6)
        return best

    def _score_entry(self, query: ParsedSpan, idx: int) -> float:
        entry = self.entries[idx]
        q_tokens = set(query.all_tokens)
        # IN/PIN/MIN/BN entries structurally never carry dose-form words --
        # "nystatin" the ingredient concept has no "suspension" in it. Do not
        # penalize them for a query that happens to mention form/route (e.g.
        # "nystatin oral suspension" with no strength stated) the way we
        # would for a dosed SCD/SBD entry, where the same missing word is a
        # real, meaningful difference between candidates.
        if entry.tty in _BARE_TTY:
            trimmed = q_tokens - FORM_VARIANT_TOKENS
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
        if strength is None:
            score = containment
        else:
            score = 0.5 * containment + 0.5 * strength

        # small nudge toward RxNorm's own "preferred" representations so
        # ties resolve the same way the competition's example output does
        score += (20 - self.tty_priority.get(entry.tty, 20)) * 0.001

        # Editorially suppressed/obsolete duplicate concepts (SUPPRESS not in
        # ("", "N")) shouldn't outrank the live one just because their STR
        # happens to tokenize a little closer to the query -- BTC's own
        # example picks the non-suppressed "24 HR metoprolol succinate..."
        # (866436) over the suppressed near-duplicate "metoprolol succinate
        # 50 MG Extended Release Oral Tablet" (866439) despite the latter
        # having no unmatched "24 HR" tokens.
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
        idxs: set[int] = set()
        for tok in query.ingredient_tokens:
            for i in self.token_index.get(tok, ()):
                if self.entries[i].tty in ttys:
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
        otherwise return the top-k spread.
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
        """Among already-ranked candidates (right ingredient/form, just no
        entry carries the literally-stated strength), pick the single nearest
        marketed strength by absolute difference -- ties go to the lower
        dose, then to the non-suppressed entry. Mirrors BTC's own example:
        "clonazepam 1.5 mg" (no such tablet exists) -> candidates ["197528"],
        the 1 MG tablet alone, not a 0.5/1/2 MG shortlist.
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

    @staticmethod
    def _mentions_injection(text: str) -> bool:
        text_l = text.lower()
        return bool(
            re.search(r"\b(?:iv|intravenous|injection|injectable)\b", text_l)
            or "truyền" in text_l
            or "truyen" in text_l
            or "tĩnh mạch" in text_l
            or "tinh mach" in text_l
        )

    @staticmethod
    def _mentioned_volumes_ml(text: str) -> set[float]:
        volumes: set[float] = set()
        for value, unit in re.findall(
            r"(\d+(?:[.,]\d+)?)\s*(ml|milliliter|milliliters|l|liter|liters)\b",
            text.lower(),
        ):
            amount = float(value.replace(",", "."))
            if unit.startswith("l") and unit not in {"l"}:
                amount *= 1000
            elif unit == "l":
                amount *= 1000
            volumes.add(amount)
        return volumes

    def _score_guided_postprocess(
        self, text: str, ranked: list[Candidate], top_k: int
    ) -> list[Candidate]:
        """Conservative leaderboard-oriented policy without per-span maps.

        The public scorer has repeatedly rewarded a single precise concept.
        Keep top-1, but when RxNorm's component tier (SCDC/TMSY) narrowly
        beats a full clinical drug because the query omitted a default form,
        prefer a full injectable product if the span itself hints IV/injection
        or the full injectable is essentially tied.  Also use stated bag
        volume to break ties among IV-fluid products.
        """
        if not ranked:
            return []

        volumes = self._mentioned_volumes_ml(text)
        if volumes:
            for c in ranked:
                for volume in volumes:
                    volume_text = f"{int(volume)}" if volume.is_integer() else f"{volume:g}"
                    if re.search(rf"\b{re.escape(volume_text)}\s*ML\b", c.str_, re.I):
                        c.score += 0.08
                        break
            ranked = sorted(ranked, key=lambda c: c.score, reverse=True)

        top = ranked[0]
        injection_hint = self._mentions_injection(text)
        if top.tty in {"SCDC", "SBDC", "TMSY"}:
            injectables = [
                c for c in ranked
                if c.tty in {"SCD", "SBD", "PSN", "SY"} and "injection" in c.str_.lower()
            ]
            if injectables:
                best = max(injectables, key=lambda c: c.score)
                margin = 0.40 if injection_hint else 0.05
                if top.score - best.score <= margin:
                    best.method = "score_guided"
                    return [best]

        return ranked[:1 if top_k <= 1 else min(top_k, 1)]

    @staticmethod
    def _surface_words(text: str) -> set[str]:
        """ASCII-ish words actually written by the author.

        This deliberately runs before ingredient aliases.  It lets the
        conservative policy distinguish a literal generic/brand name from a
        normalized typo or local alias without maintaining a mention-to-code
        lookup table.
        """
        import unicodedata

        folded = "".join(
            c for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
            if unicodedata.category(c) != "Mn"
        )
        return set(re.findall(r"[a-z]+\d*", folded))

    def _select_conservative_bare(
        self, query: ParsedSpan, ranked: list[Candidate]
    ) -> Candidate | None:
        """Select a bare concept only when the lexical evidence is unique.

        Exact brand tokens outrank a partial ingredient from parenthetical
        explanatory text (for example ``Brand (ingredient/ingredient)``).
        Tied refinements of an umbrella term, such as unspecified insulin,
        are rejected instead of choosing whichever RXCUI happens to sort
        first.
        """
        if not ranked:
            return None
        q_tokens = set(query.ingredient_tokens)
        query_surfaces = self._surface_words(query.raw)
        exact_brands = [
            c for c in ranked
            if c.tty == "BN"
            and self._surface_words(c.str_) <= query_surfaces
            and (not re.search(r"\d", c.str_) or bool(re.search(r"\d", query.raw)))
            and c.score >= 0.20
        ]
        # Override a partial ingredient only for the common explicit pattern
        # ``Brand (ingredient...)``.  A short generic phrase such as
        # "vitamin K" must not be reinterpreted as a similarly named brand.
        if exact_brands:
            return max(exact_brands, key=lambda c: c.score)

        top = ranked[0]
        top_tokens = set(parse_span(top.str_).ingredient_tokens) - NEUTRAL_TOKENS
        exact_containment = bool(top_tokens) and top_tokens <= q_tokens
        if top.score < CONFIG.min_lexical_score and not exact_containment:
            return None
        if len(ranked) > 1 and abs(top.score - ranked[1].score) < 0.01:
            return None
        return top

    # -- public API -----------------------------------------------------
    def _source_alias_match(self, text: str) -> list[Candidate]:
        match = self.source_alias_index.get(source_alias_key(text))
        if match is None:
            return []
        rxcui, sources = match
        entry = self._preferred_entry_by_rxcui.get(rxcui)
        if entry is None:
            return []
        return [Candidate(
            rxcui=entry.rxcui,
            tty=entry.tty,
            str_=entry.str_,
            score=1.0,
            method="source_alias:" + ",".join(sources),
        )]

    def link(
        self,
        text: str,
        top_k: int | None = None,
        use_tier3: bool = True,
        dose_fallback: str = "nearest",
        adaptive_parser: bool = False,
    ) -> list[Candidate]:
        """`use_tier3=False` and `dose_fallback="bare"` are experiment knobs
        (see label_rxnorm_candidates.py --no-tier3 / --dose-fallback) to A/B
        test open questions about BTC's own gold linking policy: does a
        misspelled/OOV drug name get a fuzzy-guessed candidate or stay empty
        (use_tier3), and when the stated dose isn't a real product, does gold
        want the nearest marketed strength or the bare ingredient with no
        dose at all (dose_fallback).
        """
        if not text or not text.strip():
            return []
        if top_k is None:
            top_k = CONFIG.top_k

        def tier3(t: str) -> list[Candidate]:
            return self._embedding_fallback(t, top_k) if use_tier3 else []

        query = self._medication_parser.parse(text) if adaptive_parser else parse_span(text)
        if query.ingredient_tokens and set(query.ingredient_tokens) <= NON_LINKABLE_CLASS_TERMS:
            return []
        if not query.ingredient_tokens:
            return self._source_alias_match(text) or tier3(text)

        # No strength stated -> ingredient/brand level search first; fall through to embedding if nothing matches.
        if not query.strengths:
            ranked = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
            ranked = [c for c in ranked if c.score >= CONFIG.min_lexical_score]
            if adaptive_parser and len(query.ingredient_tokens) == 1:
                # With a single generic KB token, do not invent a more
                # specific concept that the mention never stated. Example:
                # "vaccine phong dai" only anchors on "vaccine" and must not
                # become RxNorm's "pertussis vaccine" merely because both
                # contain that generic word.
                query_tokens = set(query.ingredient_tokens)
                raw_word_count = len(re.findall(r"[A-Za-zÀ-ỹ]+\d*", text))
                if raw_word_count > 1:
                    ranked = [
                        c for c in ranked
                        if (
                            set(parse_span(c.str_).ingredient_tokens)
                            - NEUTRAL_TOKENS - FORM_VARIANT_TOKENS
                        ) <= query_tokens
                    ]
            if ranked:
                return self._finalize(ranked, top_k)
            # ingredient token is not in RxNorm vocabulary at all (typo, informal name, glued-word error) -> tier 3.
            return tier3(text)

        # Tier 1+2: token-overlap + strength scoring over every dosed entry.
        dosed_idxs = self._candidate_idxs(query, _DOSED_TTY)
        ranked = self._rank(query, dosed_idxs)
        ranked = [c for c in ranked if c.score >= CONFIG.min_lexical_score]

        if ranked:
            if not self._has_exact_strength_match(query, dosed_idxs):
                if dose_fallback == "nearest":
                    nearest = self._nearest_dose(query, ranked, dosed_idxs)
                    if nearest is not None:
                        return [nearest]
                elif dose_fallback == "bare":
                    # Ignore the imperfectly-dosed matches entirely and drop
                    # straight to the ingredient-level tier below, as if the
                    # dosed search had found nothing.
                    ranked = []
            if ranked:
                return self._finalize(ranked, top_k)

        # Last resort: ingredient is real but exact strength not in RxNorm -> ingredient-level fallback.
        ranked = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
        ranked = [c for c in ranked if c.score >= CONFIG.min_lexical_score]
        if ranked:
            return self._finalize(ranked, top_k)

        # Tier 3: ingredient token itself is not in RxNorm vocabulary at all.
        return self._source_alias_match(text) or tier3(text)

    def link_exact(self, text: str) -> list[Candidate]:
        """Only return a candidate when the match is exact -- no embedding
        fallback, no partial credit. "Exact" means: after stripping pure
        filler words (NEUTRAL_TOKENS), the query's token set and the entry's
        token set are IDENTICAL (nothing extra on either side -- a form/salt
        variant the query didn't ask for is disqualifying, same as a
        candidate missing a token the query stated), and any strength(s) the
        query states match the entry's strength(s) exactly as a set (a bare
        query with no dose only matches entries that also carry no dose).
        A wrong candidate scores worse than no candidate under this
        competition's Jaccard metric, so anything short of exact is treated
        the same as no match at all.
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
        # Mirror link()'s own tiering: a bare (no-dose) mention should only
        # ever match bare ingredient/brand entries, never a dosed SCD/SCDF/etc
        # -- otherwise "omeprazole" alone would tie with "omeprazole Oral
        # Tablet" (SCDF), which states a form the query never did.
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
        # A dosed ingredient+strength query with the default oral-tablet form
        # ties SCD ("furosemide 40 MG Oral Tablet") against SCDC ("furosemide
        # 40 MG") -- same tokens+strength, different RXCUI, because "oral"/
        # "tablet" are neutral filler on both sides. These aren't a genuine
        # ambiguity between two different real-world drugs, just two RxNorm
        # modeling granularities of the one concept the query names -- keep
        # only the higher-priority tier (SCD/PSN/SBD over SCDC/SBDC/BN/...),
        # same tie-break link() already applies via the score nudge.
        best_priority = min(self.tty_priority.get(c.tty, 20) for c in matches.values())
        return [c for c in matches.values() if self.tty_priority.get(c.tty, 20) == best_priority]

    def link_exact_rxcuis(self, text: str) -> list[str]:
        return [c.rxcui for c in self.link_exact(text)]

    # -- tier 3: embedding fallback with hybrid scoring ---------------
    def _get_embed_index(self):
        if self._embed_index is None:
            from .embed_index import EmbedIndex
            self._embed_index = EmbedIndex(preferred_entries=self._preferred_entry_by_rxcui)
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

        # If no expansion produced valid views, fall back to raw text search
        view_texts = [v["text"] for v in views] if views else [text.lower()]

        # Search using each view and aggregate scores per RXCUI
        rxcui_scores: dict[str, list[float]] = {}
        rxcui_strs: dict[str, str] = {}

        for vt in view_texts:
            hits = index.search(vt, top_k=top_k * 5)  # get more candidates to merge from multiple views
            for rxcui, str_, sim in hits:
                if sim >= CONFIG.min_embedding_similarity:
                    rxcui_scores.setdefault(rxcui, []).append(sim)
                    rxcui_strs[rxcui] = str_

        # Average scores across views for each RXCUI, then pick top-k by avg similarity
        results: list[Candidate] = []
        for rxcui, sims in rxcui_scores.items():
            avg_sim = float(np.mean(sims))
            results.append(Candidate(rxcui=rxcui, tty="", str_=rxcui_strs[rxcui], score=avg_sim, method="embedding"))

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
            # Find the entry index for this RXCUI to compute lexical features
            entry_idx = None
            for i, e in enumerate(self.entries):
                if e.rxcui == c.rxcui:
                    entry_idx = i
                    break

            if entry_idx is not None:
                entry = self.entries[entry_idx]

                # Lexical component (reuse existing scorer)
                lex_score = self._score_entry(query, entry_idx)

                # Token overlap bonus
                overlap = self._token_overlap(query, entry)

                # Exact match bonus (if applicable) - use current embedding score as proxy for exact match check
                exact_bonus = self._exact_match_bonus(query, entry) if c.method == "embedding" else 0.0

                # Hybrid score: weighted combination of lexical and embedding signals
                embed_score = c.score  # already normalized cosine similarity from embedding search

                hybrid = (CONFIG.lexical_weight * lex_score +
                          CONFIG.embedding_weight * embed_score +
                          CONFIG.token_overlap_bonus * overlap + exact_bonus)

                c.score = hybrid
                c.method = "hybrid"

            scored.append(c)

        scored.sort(key=lambda c: c.score, reverse=True)

        # De-duplicate by rxcui again after reranking (shouldn't happen but be safe)
        seen: dict[str, Candidate] = {}
        for c in scored:
            if c.rxcui not in seen or c.score > seen[c.rxcui].score:
                seen[c.rxcui] = c

        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    def _cross_encoder_rerank(self, text: str, candidates: list[Candidate]) -> list[Candidate]:
        """Refine ordering of the top candidates with bge-reranker-v2-m3.

        Cross-encoder scores are blended with the existing hybrid score rather
        than replacing it outright, since the cross-encoder only sees the
        candidate's RxNorm string (not strength/form structure the lexical
        scorer already accounts for). If nothing in the pool clears
        `min_rerank_confidence`, the pool is discarded entirely (empty tier-3
        result) rather than forcing a top_k guess -- a wrong candidate scores
        worse than no candidate under the competition's Jaccard metric.

        If the single best candidate clears the higher `tier3_collapse_confidence`
        bar, it is returned alone -- same collapse behavior tiers 1-2 already
        get via `_finalize()`, now extended to tier 3 so a confident top-1
        embedding match isn't diluted by weaker candidates riding along in top_k.
        """
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

        # Pair before blending so the collapse decision below uses the
        # reranker's own raw confidence, not the post-blend score.
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
        adaptive_parser: bool = False,
    ) -> list[str]:
        return [
            c.rxcui
            for c in self.link(
                text, top_k=top_k, use_tier3=use_tier3,
                dose_fallback=dose_fallback, adaptive_parser=adaptive_parser,
            )
        ]

    def link_score_guided_rxcuis(self, text: str) -> list[str]:
        query = parse_span(text)
        if query.ingredient_tokens and set(query.ingredient_tokens) <= NON_LINKABLE_CLASS_TERMS:
            return []

        if query.ingredient_tokens and not query.strengths:
            ranked = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
            selected = self._select_conservative_bare(query, ranked)
            if selected:
                return [selected.rxcui]
            return [c.rxcui for c in self._source_alias_match(text)]

        dosed = self._rank(query, self._candidate_idxs(query, _DOSED_TTY))
        dosed = [c for c in dosed if c.score >= CONFIG.min_lexical_score]
        surfaces = self._surface_words(text)

        if dosed and self._has_exact_strength_match(
            query, self._candidate_idxs(query, _DOSED_TTY)
        ):
            top = dosed[0]
            top_tokens = set(parse_span(top.str_).ingredient_tokens)
            literal_generic = bool(top_tokens & surfaces & set(query.ingredient_tokens))

            # An exact component is safe when it is the best representation;
            # a full default oral tablet is safe only when its generic name
            # was literally written (not reached through a typo/local alias).
            volumes = self._mentioned_volumes_ml(text)
            injection_hint = self._mentions_injection(text)

            # A bare administered volume does not identify a product unless
            # that same volume exists in the RxNorm string.  Otherwise retain
            # the ingredient/concentration concept.
            product_volume_match = not volumes or any(
                re.search(rf"\b{re.escape(f'{volume:g}')}\s*ML\b", c.str_, re.I)
                for volume in volumes for c in dosed
            )

            if top.score >= 1.0 and product_volume_match and (
                (top.tty in {"SCDC", "SBDC"}
                 and not any(
                     c.tty in {"SCD", "SBD", "PSN", "SY"}
                     and "injection" in c.str_.lower()
                     and top.score - c.score <= 0.05
                     for c in dosed
                 ))
                or (top.tty == "SCD" and literal_generic and not injection_hint)
            ):
                return [top.rxcui]

            # Explicit oral shorthand can disambiguate a salt-specific tablet
            # even when the salt qualifier causes a small lexical penalty.
            if (top.tty == "SCD" and literal_generic and not injection_hint
                    and top.score >= 0.95
                    and re.search(r"\bpo\b", text, re.I)):
                return [top.rxcui]

            # A literal branded product with the stated strength is stronger
            # evidence than its dose-less BN concept.
            if (top.tty in {"PSN", "SBD", "SY"} and top.score >= 0.75
                    and not injection_hint):
                entry_words = self._surface_words(top.str_)
                bare_ranked = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
                literal_bn = any(
                    c.tty == "BN" and self._surface_words(c.str_) & surfaces
                    for c in bare_ranked[:5]
                )
                if entry_words & surfaces and (
                    top.tty in {"PSN", "SBD"} or (literal_bn and top.score >= 0.90)
                ):
                    return [top.rxcui]

            injectables = [
                c for c in dosed
                if c.tty in {"SCD", "SBD", "PSN", "SY"}
                and "injection" in c.str_.lower()
                and set(parse_span(c.str_).strengths) & set(query.strengths)
            ]
            oral_products = [
                c for c in dosed
                if c.score >= 0.90 and "oral" in c.str_.lower()
            ]
            if injectables:
                injectable = max(injectables, key=lambda c: c.score)
                if injection_hint and not oral_products and injectable.score >= 0.90:
                    return [injectable.rxcui]
                # A stated bag volume uniquely identifies an IV-fluid product.
                for volume in volumes:
                    volume_text = f"{volume:g}"
                    if re.search(rf"\b{re.escape(volume_text)}\s*ML\b", injectable.str_, re.I):
                        return [injectable.rxcui]
                # If RxNorm has a strength component and exactly one nearly
                # tied full product, the product supplies the missing default
                # dose form without guessing among alternatives.
                if top.tty in {"SCDC", "SBDC"} and len(injectables) == 1:
                    if top.score - injectable.score <= 0.05:
                        return [injectable.rxcui]

        bare = self._rank(query, self._candidate_idxs(query, _BARE_TTY))
        selected = self._select_conservative_bare(query, bare)
        if selected:
            return [selected.rxcui]
        return [c.rxcui for c in self._source_alias_match(text)]


@lru_cache(maxsize=1)
def get_linker() -> RxNormLinker:
    return RxNormLinker()


# Keep backward-compatible constants that may be imported elsewhere. These are now read from config.
MIN_SCORE = CONFIG.min_lexical_score   # type: ignore[name-defined] # noqa: F821 
HIGH_CONF = CONFIG.collapse_threshold   # type: ignore[name-defined] # noqa: F821 
TOP_K = CONFIG.top_k                   # type: ignore[name-defined] # noqa: F821 
EMBED_MIN_SIM = CONFIG.min_embedding_similarity   # type: ignore[name-defined] # noqa: F821 
_DOSED_TTY = {"SCD", "SBD", "SCDC", "SBDC", "SCDF", "SBDF", "PSN", "SY", "TMSY", "BN"}   # type: ignore[name-defined] # noqa: F821 
_INGREDIENT_TTY = {"IN", "PIN", "MIN"}   # type: ignore[name-defined] # noqa: F821 
_BARE_TTY = _INGREDIENT_TTY | {"BN"}   # type: ignore[name-defined] # noqa: F821 
