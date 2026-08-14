"""Deterministic ICD-10 TT06 matcher cascade."""

from __future__ import annotations

import re
from typing import Any

from .fuzzy import top_fuzzy_candidates
from .index import ICDIndex
from .normalize import normalize_aggressive, normalize_conservative
from .qualifier import parse_qualifiers


_CODE_RE = re.compile(r"\b([A-Z][0-9]{2}(?:\.[0-9A-Z])?)\b", re.IGNORECASE)
_NEGATED_ASSERTIONS = {"negated", "absent", "denied", "no", "false"}


class ICDMatcher:
    def __init__(
        self,
        index_dir: str = "data/processed",
        use_embedding: bool = False,
        embedding_mode: str = "disabled",
        embedding_model: str | None = None,
        embedding_index: Any | None = None,
        index: ICDIndex | None = None,
        embedding_top_k: int = 5,
        embedding_min_score: float = 0.80,
        embedding_min_margin: float = 0.03,
    ):
        self.index = index if index is not None else ICDIndex(index_dir)
        self.use_embedding = use_embedding or embedding_mode != "disabled"
        if use_embedding and embedding_mode == "disabled":
            embedding_mode = "fallback"
        if embedding_mode not in {"disabled", "fallback", "rerank_fuzzy", "replace_fuzzy"}:
            raise ValueError(f"Unsupported embedding_mode: {embedding_mode}")
        self.embedding_mode = embedding_mode
        self.embedding_model = embedding_model
        self.embedding_index = embedding_index
        self.embedding_top_k = embedding_top_k
        self.embedding_min_score = embedding_min_score
        self.embedding_min_margin = embedding_min_margin

    def match(self, span: str, assertion: str | None = None) -> dict[str, Any]:
        normalized = normalize_aggressive(span)
        parsed = parse_qualifiers(span)
        result = self._empty_result(span, normalized, parsed)

        if assertion and assertion.lower() in _NEGATED_ASSERTIONS:
            result.update({"code": None, "confidence": 0.0, "method": "skipped_negated"})
            return result

        # Clean text for linking: strip assertion prefixes/suffixes
        from .assertion import separate_assertion
        linking_text, _ = separate_assertion(span, known_assertion=assertion)
        # Use linking_text for all downstream steps instead of raw span
        # Only override for code/alias/name/fuzzy/embedding (not qualifier parsing)
        linking_normalized = normalize_aggressive(linking_text)

        code_match = _CODE_RE.search(linking_text or "")
        if code_match:
            code = code_match.group(1).upper()
            node = self.index.get_code(code)
            if node:
                return self._final(result, node, 1.0, "code_exact")

        alias = self.index.lookup_alias(linking_text)
        if alias and (node := self.index.get_code(alias.get("code"))):
            return self._final(result, node, float(alias.get("confidence", 1.0)), "dictionary_exact")

        if node := self.index.lookup_name(linking_text):
            return self._final(result, node, 1.0, "dictionary_exact")

        backoff = self._qualifier_backoff(parsed)
        if backoff:
            return self._final(
                result,
                backoff["node"],
                backoff["confidence"],
                backoff["method"],
                candidates=backoff["candidates"],
                debug=backoff["debug"],
            )

        if self.embedding_mode == "replace_fuzzy":
            return self._embedding_match(result, linking_text, method_high="embedding_high_confidence")

        if self.embedding_mode == "rerank_fuzzy":
            fuzzy = self._fuzzy_match(linking_normalized)
            candidate_codes = [candidate["code"] for candidate in fuzzy["candidates"] if candidate.get("code")]
            if candidate_codes:
                return self._embedding_match(
                    result,
                    span,
                    candidate_codes=candidate_codes,
                    method_high="fuzzy+embedding_rerank",
                    extra_debug=fuzzy["debug"],
                )
            result["candidates"] = fuzzy["candidates"]
            result["debug"].update(fuzzy["debug"])
            result["method"] = "no_match"
            return result

        fuzzy = self._fuzzy_match(linking_normalized)
        if fuzzy["node"]:
            return self._final(
                result,
                fuzzy["node"],
                fuzzy["confidence"],
                fuzzy["method"],
                candidates=fuzzy["candidates"],
                debug=fuzzy["debug"],
            )

        if self.embedding_mode == "fallback":
            return self._embedding_match(
                result,
                linking_text,
                method_high="embedding_high_confidence",
                extra_debug=fuzzy["debug"],
            )

        result["candidates"] = fuzzy["candidates"]
        result["debug"]["fuzzy_topk"] = fuzzy["debug"].get("fuzzy_topk", [])
        result["method"] = "no_match"
        return result

    def _empty_result(self, span: str, normalized: str, parsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": span,
            "normalized": normalized,
            "base_text": parsed["base_text"],
            "qualifiers": parsed["qualifiers"],
            "code": None,
            "name_vi": None,
            "confidence": 0.0,
            "method": "no_match",
            "candidates": [],
            "debug": {"removed_phrases": parsed["removed_phrases"]},
        }

    def _final(
        self,
        result: dict[str, Any],
        node: dict[str, Any],
        confidence: float,
        method: str,
        candidates: list[dict[str, Any]] | None = None,
        debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result.update(
            {
                "code": node.get("code"),
                "name_vi": node.get("name_vi"),
                "confidence": confidence,
                "method": method,
                "candidates": candidates or [],
            }
        )
        if debug:
            result["debug"].update(debug)
        return result

    def _qualifier_backoff(self, parsed: dict[str, Any]) -> dict[str, Any] | None:
        base = parsed["base_text"]
        base_alias = self.index.lookup_alias(base)
        base_node = None
        if base_alias:
            base_node = self.index.get_code(base_alias.get("code"))
        if base_node is None:
            base_node = self.index.lookup_name(base)
        if base_node is None:
            return None

        children = self.index.child_nodes(base_node.get("code", ""))
        candidates = [
            {
                "code": child.get("code"),
                "name_vi": child.get("name_vi"),
                "score": 0.0,
                "method": "qualifier_backoff",
            }
            for child in children
        ]
        qualifiers = parsed["qualifiers"]
        if not children:
            return {
                "node": base_node,
                "confidence": 0.75,
                "method": "category_with_candidates",
                "candidates": candidates,
                "debug": {"qualifier_backoff": "no_children"},
            }

        if qualifiers.get("unspecified"):
            child = self._best_unspecified_child(children)
            if child:
                return {
                    "node": child["node"],
                    "confidence": child["confidence"],
                    "method": "dictionary+qualifier_backoff",
                    "candidates": candidates,
                    "debug": {"qualifier_backoff": child["reason"], "base_code": base_node.get("code")},
                }

        return {
            "node": base_node,
            "confidence": 0.75,
            "method": "category_with_candidates",
            "candidates": candidates,
            "debug": {"qualifier_backoff": "no_safe_child"},
        }

    def _best_unspecified_child(self, children: list[dict[str, Any]]) -> dict[str, Any] | None:
        for child in children:
            normalized = child.get("normalized_name", "")
            if "khong xac dinh" in normalized or "khong dac hieu" in normalized:
                return {"node": child, "confidence": 0.95, "reason": "unspecified_phrase"}
        for child in children:
            if str(child.get("code", "")).endswith(".9"):
                return {"node": child, "confidence": 0.85, "reason": "dot_9_fallback"}
        return None

    def _fuzzy_match(self, normalized: str) -> dict[str, Any]:
        top = top_fuzzy_candidates(normalized, self.index.fuzzy_choices(), top_k=5)
        candidates = [
            {
                "code": item.code,
                "name_vi": item.name_vi,
                "score": item.score / 100,
                "method": "fuzzy",
                "source": item.source,
            }
            for item in top
        ]
        debug = {"fuzzy_topk": candidates}
        if not top:
            return {"node": None, "confidence": 0.0, "method": "no_match", "candidates": [], "debug": debug}

        first = top[0]
        second_score = 0.0
        for item in top[1:]:
            if item.code != first.code:
                second_score = item.score
                break
        margin = first.score - second_score
        node = self.index.get_code(first.code)
        if node and first.score >= 85 and margin >= 5:
            confidence = min(0.9, max(0.7, first.score / 100))
            return {
                "node": node,
                "confidence": confidence,
                "method": "fuzzy_high_confidence",
                "candidates": candidates,
                "debug": {**debug, "fuzzy_margin": margin},
            }
        return {
            "node": None,
            "confidence": 0.0,
            "method": "fuzzy_candidates",
            "candidates": candidates,
            "debug": {**debug, "fuzzy_margin": margin},
        }

    def _get_embedding_index(self) -> Any:
        if self.embedding_index is None:
            from .embedding import ICDEmbeddingIndex

            self.embedding_index = ICDEmbeddingIndex(
                index_dir=self.index.index_dir,
                model_name=self.embedding_model,
            )
        return self.embedding_index

    def _embedding_match(
        self,
        result: dict[str, Any],
        query: str,
        candidate_codes: list[str] | None = None,
        method_high: str = "embedding_high_confidence",
        extra_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if extra_debug:
            result["debug"].update(extra_debug)

        embedding = self._get_embedding_index()

        # Multi-view: raw + normalized
        from .normalize import normalize_aggressive
        q_raw = query
        q_norm = normalize_aggressive(query)

        # Encode both views
        raw_candidates_raw = embedding.retrieve(
            q_raw,
            top_k=self.embedding_top_k,
            candidate_codes=candidate_codes,
        )
        raw_candidates_norm = embedding.retrieve(
            q_norm,
            top_k=self.embedding_top_k,
            candidate_codes=candidate_codes,
        )

        # Merge by code: take max score
        merged: dict[str, dict] = {}
        for entry in raw_candidates_raw:
            code = entry["code"]
            merged[code] = {"code": code, "name_vi": entry["name_vi"],
                           "score": entry["score"], "source": "raw"}
        for entry in raw_candidates_norm:
            code = entry["code"]
            if code in merged:
                # Boost: 0.95 * normalized score if it's higher
                merged[code]["score"] = max(merged[code]["score"], 0.95 * entry["score"])
                merged[code]["source"] = "hybrid"
            else:
                merged[code] = {"code": code, "name_vi": entry["name_vi"],
                               "score": 0.95 * entry["score"], "source": "normalized"}

        candidates = sorted(merged.values(),
                           key=lambda x: -x["score"])[:self.embedding_top_k]

        # Add method tag
        for c in candidates:
            c["method"] = "embedding"

        result["debug"]["embedding_topk"] = candidates
        if not candidates:
            result["candidates"] = []
            result["method"] = "embedding_topk"
            return result

        top_score = candidates[0]["score"]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = top_score - second_score
        result["debug"]["embedding_margin"] = margin

        top_node = self.index.get_code(candidates[0]["code"])
        if top_node and top_score >= self.embedding_min_score and margin >= self.embedding_min_margin:
            return self._final(
                result,
                top_node,
                top_score,
                method_high,
                candidates=candidates,
            )

        result["code"] = None
        result["name_vi"] = None
        result["confidence"] = top_score
        result["method"] = "embedding_topk"
        result["candidates"] = candidates
        return result
