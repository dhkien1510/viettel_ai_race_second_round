"""Content and section router for the text/assertion pipeline.

The router supplies context to a downstream model. It never creates, drops or
retypes entities, because genre is a useful prior rather than an annotation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..sections import SectionCues, annotate

DocumentGenre = Literal["FAQ", "Q&A", "EHR", "HYBRID", "FREE_TEXT"]
SegmentKind = Literal[
    "FAQ_EDUCATIONAL",
    "QA_USER",
    "QA_CLINICIAN",
    "EHR_HISTORY",
    "EHR_CURRENT",
    "EHR_LAB",
    "EHR_DIAGNOSIS",
    "EHR_OTHER",
    "FREE_TEXT",
]


_QA_USER = re.compile(
    r"(?im)^\s*(?:hỏi|câu\s+hỏi(?:\s+từ\s+người\s+dùng)?)\s*:"
)
_QA_CLINICIAN = re.compile(
    r"(?im)^\s*(?:trả\s+lời|câu\s+trả\s+lời(?:\s+của\s+bác\s+sĩ)?|"
    r"bác\s+sĩ\s+trả\s+lời)\s*:"
)
_FAQ = re.compile(
    r"(?im)^\s*(?:\d+\s*[.)]\s*)?(?:.+\s+là\s+gì\?|tổng\s+quan|nguyên\s+nhân|"
    r"dấu\s+hiệu|phòng\s+ngừa|khi\s+nào\s+cần)"
)

_HISTORY_CATEGORIES = {"HISTORY", "MED_HISTORY", "FAMILY_HISTORY"}
_CURRENT_CATEGORIES = {
    "HPI", "SYMPTOM_CURRENT", "PRESENT_ILLNESS", "EXAM", "TREATMENT",
}
_LAB_CATEGORIES = {"LAB", "IMAGING", "PROCEDURE", "VITAL"}
_DIAGNOSIS_CATEGORIES = {"DIAGNOSIS", "ASSESSMENT"}


@dataclass(frozen=True)
class ContextSegment:
    start: int
    end: int
    kind: SegmentKind
    section_category: str
    text: str

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "section_category": self.section_category,
            "text": self.text,
        }


@dataclass(frozen=True)
class RoutedDocument:
    genre: DocumentGenre
    segments: tuple[ContextSegment, ...]


class ContextRouter:
    """Classify a document and split it at role/clinical-section boundaries."""

    def __init__(self, cues: SectionCues | None = None):
        self.cues = cues or SectionCues.load()

    def classify(self, text: str) -> DocumentGenre:
        qa_positions = [
            match.start()
            for pattern in (_QA_USER, _QA_CLINICIAN)
            for match in pattern.finditer(text)
        ]
        smap = annotate(text, self.cues)
        clinical = [s for s in smap.sections if s.category != "ROOT"]
        has_ehr = len(clinical) >= 2

        # Q&A answers frequently contain headings that look like EHR sections.
        # A role marker at the beginning therefore wins over section counts.
        # A later role marker means a substantial pre-dialogue region exists.
        if qa_positions:
            return "Q&A" if min(qa_positions) == 0 else "HYBRID"
        # FAQ headings are only a document-level signal near the beginning.
        # EHR notes may contain an unrelated educational fragment later on.
        if _FAQ.search(text[:300]):
            return "FAQ"
        if has_ehr:
            return "EHR"
        return "FREE_TEXT"

    def route(self, text: str) -> RoutedDocument:
        genre = self.classify(text)
        smap = annotate(text, self.cues)

        # Both section starts and speaker-role markers change the policy that
        # the downstream prompt/assertion pass should use.
        boundaries = {0, len(text)}
        boundaries.update(s.start for s in smap.sections)
        for match in _QA_USER.finditer(text):
            boundaries.add(match.start())
        for match in _QA_CLINICIAN.finditer(text):
            boundaries.add(match.start())

        points = sorted(boundaries)
        segments = []
        active_role: SegmentKind | None = None
        for start, end in zip(points, points[1:]):
            if start == end or not text[start:end].strip():
                continue

            prefix = text[start:min(end, start + 120)]
            if _QA_USER.match(prefix):
                active_role = "QA_USER"
            elif _QA_CLINICIAN.match(prefix):
                active_role = "QA_CLINICIAN"

            section = smap.section_at(start)
            category = section.category if section else "ROOT"
            kind = active_role or self._kind_for_section(category, genre)
            segments.append(ContextSegment(start, end, kind, category, text[start:end]))

        if not segments and text:
            segments.append(ContextSegment(0, len(text), "FREE_TEXT", "ROOT", text))
        return RoutedDocument(genre, tuple(segments))

    @staticmethod
    def _kind_for_section(category: str, genre: DocumentGenre) -> SegmentKind:
        # FAQ articles often contain headings such as "Đánh giá" or "Triệu
        # chứng" that overlap EHR section cues. Once the document-level FAQ
        # signal is established, those headings remain educational content.
        if genre == "FAQ":
            return "FAQ_EDUCATIONAL"
        if category in _HISTORY_CATEGORIES:
            return "EHR_HISTORY"
        if category in _CURRENT_CATEGORIES:
            return "EHR_CURRENT"
        if category in _LAB_CATEGORIES:
            return "EHR_LAB"
        if category in _DIAGNOSIS_CATEGORIES:
            return "EHR_DIAGNOSIS"
        if category != "ROOT":
            return "EHR_OTHER"
        return "FREE_TEXT"
