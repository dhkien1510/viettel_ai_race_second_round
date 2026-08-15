"""Dataset-independent medication query parsing.

Instead of enumerating Vietnamese administration words, this parser asks the
RxNorm index which tokens can identify a concept.  The protocol is deliberately
small so a learned token classifier can replace this implementation later.
"""
from __future__ import annotations

import re
from typing import Mapping, Protocol, Sequence

from .normalize import (
    FORM_VARIANT_TOKENS, INGREDIENT_ALIASES, NEUTRAL_TOKENS, PHRASE_ALIASES,
    SALT_VARIANT_TOKENS, ParsedSpan, _FORM_HINTS, _STRENGTH_RE, _WORD_RE,
    _strip_accents, parse_span,
)


class MedicationParser(Protocol):
    def parse(self, text: str) -> ParsedSpan: ...


class RxNormVocabularyParser:
    """Select identity anchors using document frequency in the RxNorm index."""

    def __init__(self, token_index: Mapping[str, Sequence[int]], entry_count: int,
                 max_df_ratio: float = 1.0) -> None:
        self.token_index = token_index
        # The ratio remains configurable for experiments, but defaults to
        # full vocabulary coverage. Common ingredients (guaifenesin, acid,
        # insulin...) are legitimate anchors and must not be removed merely
        # because they have many RxNorm product descendants.
        self.max_document_frequency = max(1, int(entry_count * max_df_ratio))

    def _is_identity_anchor(self, token: str) -> bool:
        postings = self.token_index.get(token)
        return bool(postings) and len(postings) <= self.max_document_frequency

    def parse(self, text: str) -> ParsedSpan:
        # Keep the mature strength/unit canonicalization, but replace its
        # stopword-filtered identity extraction.
        legacy = parse_span(text)
        normalized = _strip_accents(_STRENGTH_RE.sub(" ", text.lower()))
        for phrase, replacement in PHRASE_ALIASES.items():
            normalized = re.sub(rf"\b{re.escape(phrase)}\b", replacement, normalized)

        identity: list[str] = []
        single_letter_anchors: list[str] = []
        form_hints: list[str] = []
        for word in _WORD_RE.findall(normalized):
            if word in _FORM_HINTS:
                form_hints.append(_FORM_HINTS[word])
                continue
            for token in INGREDIENT_ALIASES.get(word, word).split():
                # A bare letter is not a safe concept anchor ("B-histamin"
                # previously retrieved the unrelated brand "B-3-50").
                # Meaningful compact names such as B12 retain their digits.
                if len(token) == 1:
                    if token in self.token_index:
                        single_letter_anchors.append(token)
                    continue
                if token in FORM_VARIANT_TOKENS:
                    form_hints.append(token)
                elif token in NEUTRAL_TOKENS:
                    # Keep KB-known neutral words as low-weight metadata; they
                    # must not become identity retrieval anchors.
                    if token in self.token_index:
                        form_hints.append(token)
                    continue
                elif token in SALT_VARIANT_TOKENS or self._is_identity_anchor(token):
                    identity.append(token)

        # A single letter is meaningful only when qualified by another drug
        # anchor (Vitamin K), never as the sole evidence (B-histamin -> brand
        # B-3-50). This is evidence-based rather than language-specific.
        if identity:
            identity.extend(single_letter_anchors)

        return ParsedSpan(
            raw=text,
            ingredient_tokens=tuple(dict.fromkeys(identity)),
            form_hints=tuple(dict.fromkeys(form_hints)),
            strengths=legacy.strengths,
        )
