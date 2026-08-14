"""Section annotator.

Scans the note line-by-line, detects heading lines, and produces a mapping
from every character position to the clinical section it belongs to. This is
the single strongest signal in this dataset: it drives both type resolution
and the isHistorical assertion.

Key correctness point (the trap in earlier plans): "Tiền sử bệnh hiện tại" /
"Bệnh sử hiện tại" is the HISTORY OF PRESENT ILLNESS — it is CURRENT, not
historical. Only genuine past-history / pre-admission-medication sections are
`historical_prone`. We guarantee this by preferring the LONGEST matching cue,
so "tiền sử bệnh hiện tại" (HPI) always beats "tiền sử bệnh" (HISTORY).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from .schema import DATA_DIR
from .offsets import line_spans

_LEADING = re.compile(
    r"^\s*(?:\*{1,2}\s*)?(?:(?:\d+|[ivxlcdm]+)\s*[.\)]\s*)?(?:[-*•]\s*)?", re.IGNORECASE
)

HISTORICAL_PRONE = {"HISTORY", "MED_HISTORY"}
FAMILY_PRONE = {"FAMILY_HISTORY"}


@dataclass
class Section:
    name: str            # matched cue (or "ROOT")
    category: str        # HPI / SYMPTOM_CURRENT / MED_HISTORY / HISTORY / ...
    start: int
    end: int

    @property
    def historical_prone(self) -> bool:
        return self.category in HISTORICAL_PRONE

    @property
    def family_prone(self) -> bool:
        return self.category in FAMILY_PRONE


class SectionCues:
    """Loaded heading cues, flattened to (cue, category) sorted longest-first."""

    def __init__(self, cues_by_cat: Dict[str, List[str]]):
        pairs = []
        for cat, cues in cues_by_cat.items():
            for c in cues:
                pairs.append((c.casefold(), cat))
        # longest cue first so the most specific heading wins
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
        self.pairs = pairs

    @classmethod
    def load(cls, path=None) -> "SectionCues":
        path = path or (DATA_DIR / "section_cues.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def match_heading(self, line_core: str) -> Optional[tuple]:
        """Return (cue, category) if the (numbering-stripped) line starts with a
        known heading cue, else None."""
        stripped = _LEADING.sub("", line_core).casefold().strip()
        if not stripped:
            return None
        # every cue that is a prefix of the line, most specific (longest) first
        hits = [(cue, cat) for cue, cat in self.pairs if stripped.startswith(cue)]
        for cue, cat in hits:
            # accept if the heading is the whole line, or is followed by a
            # separator / space (inline "Lý do nhập viện: ...") — avoids
            # matching a longer unrelated word that merely starts with cue.
            # "*" is accepted too (closing markdown bold, "**Bệnh sử hiện tại**").
            nxt = stripped[len(cue):len(cue) + 1]
            if nxt in ("", " ", ":", "-", "/", ",", ".", "(", "*"):
                return cue, cat
            # The cue is GLUED to the following text ("Tiền sử bệnh hiện
            # tạiBệnh nhân nhập viện vì…" — input routinely loses spaces). The
            # boundary guard just rejected it; without this branch we fall
            # through to a SHORTER cue that happens to end on a space and is a
            # prefix of this one — e.g. HPI "tiền sử bệnh hiện tại" losing to
            # HISTORY "tiền sử bệnh", which then marks the whole present-illness
            # narrative isHistorical. A cue that merely prefixes this one is by
            # construction less specific, so it must never win here.
            if any(other != cue and cue.startswith(other) for other, _ in hits):
                return cue, cat
        return None


class SectionMap:
    """Answers `category_at(pos)` and `historical_at(pos)`."""

    def __init__(self, sections: List[Section], text_len: int):
        self.sections = sections
        self.text_len = text_len

    def _find(self, pos: int) -> Optional[Section]:
        # sections are contiguous & ordered; linear scan is fine for note sizes
        for s in self.sections:
            if s.start <= pos < s.end:
                return s
        return self.sections[-1] if self.sections else None

    def category_at(self, pos: int) -> str:
        s = self._find(pos)
        return s.category if s else "ROOT"

    def historical_at(self, pos: int) -> bool:
        s = self._find(pos)
        return bool(s and s.historical_prone)

    def family_at(self, pos: int) -> bool:
        s = self._find(pos)
        return bool(s and s.family_prone)

    def section_at(self, pos: int) -> Optional[Section]:
        return self._find(pos)


def annotate(text: str, cues: Optional[SectionCues] = None) -> SectionMap:
    cues = cues or SectionCues.load()
    lines = line_spans(text)

    # find heading boundaries
    boundaries: List[tuple] = []  # (start_offset, cue, category)
    for (ls, le, core) in lines:
        m = cues.match_heading(core)
        if m:
            boundaries.append((ls, m[0], m[1]))

    sections: List[Section] = []
    if not boundaries or boundaries[0][0] > 0:
        # leading region before the first heading
        first = boundaries[0][0] if boundaries else len(text)
        sections.append(Section("ROOT", "ROOT", 0, first))

    for i, (start, cue, cat) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        sections.append(Section(cue, cat, start, end))

    if not sections:
        sections.append(Section("ROOT", "ROOT", 0, len(text)))

    return SectionMap(sections, len(text))
