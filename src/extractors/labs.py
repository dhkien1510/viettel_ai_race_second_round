"""Lab / test extractor.

Emits TÊN_XÉT_NGHIỆM for test names and, separately, KẾT_QUẢ_XÉT_NGHIỆM for
the value/result — the two are ALWAYS distinct spans (the spec's most
emphasized scoring rule). Handles numeric values (with , or . decimals,
ranges, arrows) and, in high-recall mode, textual results such as
"không ghi nhận gì bất thường".
"""

from __future__ import annotations

import re
from typing import List

import yaml

from ..schema import DATA_DIR, Entity, TYPE_TEST_NAME, TYPE_TEST_RESULT
from ..offsets import find_phrase_spans

# Known unit tokens only (NOT a generic [a-zA-Z]+ scan): a bare ASCII-letter
# run would greedily eat the leading ASCII letter(s) of the next Vietnamese
# word (e.g. "6.3 mẫu" -> unit group grabs "m" from "mẫu", since "ẫ" falls
# outside [a-zA-Z] and stops the run mid-word, producing "6.3 m"). Requiring
# an exact, whole unit from this list avoids that class of bug entirely.
_UNIT = (
    r"(?:mmol\s*/\s*l|mg\s*/\s*dl|mg\s*/\s*l|meq\s*/\s*l|g\s*/\s*l|"
    r"k\s*/\s*ul|/\s*ul|/\s*mm3|u\s*/\s*l|ui\s*/\s*l|iu\s*/\s*l|"
    r"mmhg|bpm|mcg|mg|ml|g|iu|%)"
)
# a numeric result, optionally a range / arrow / trailing unit
_VALUE = re.compile(
    r"""(?P<val>
        \d+(?:[.,]\d+)?                               # 12.0 / 14,43 / 26
        (?:\s*(?:-->|->|→|–|-|đến|lên|tới)\s*\d+(?:[.,]\d+)?)*  # ranges/arrows
        (?:\s*""" + _UNIT + r"""(?!\w))?                # unit, but not mid-word
    )""",
    re.VERBOSE | re.IGNORECASE,
)
# connector tokens allowed between a lab name and its value
_CONNECTOR = re.compile(r"^[\s:]*(?:là|đo được|ở mức|khoảng|tăng lên|tăng từ|giảm xuống|giảm còn|nâng cao lên|:)?\s*")

_LOOKAHEAD = 60  # chars to scan after a lab name for its value


class LabExtractor:
    def __init__(self, cfg: dict):
        self.blood = sorted(cfg.get("blood_labs", []), key=len, reverse=True)
        self.tests = sorted(cfg.get("tests", []), key=len, reverse=True)
        self.result_phrases = sorted(cfg.get("result_phrases", []), key=len, reverse=True)

    @classmethod
    def load(cls, path=None) -> "LabExtractor":
        path = path or (DATA_DIR / "lab_alias.yaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def extract(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        # name spans -> so we can bound value lookahead and avoid crossing names
        name_spans: List[tuple] = []

        for name in self.blood:
            for s, e in find_phrase_spans(text, name):
                ents.append(Entity(text[s:e], s, e, TYPE_TEST_NAME, source="lab:blood"))
                name_spans.append((s, e, True))   # numeric-bearing
        for name in self.tests:
            for s, e in find_phrase_spans(text, name):
                ents.append(Entity(text[s:e], s, e, TYPE_TEST_NAME, source="lab:test"))
                name_spans.append((s, e, False))

        name_spans.sort()

        # attach a value to each name (numeric for blood labs, else numeric or
        # textual phrase for both)
        starts = [ns[0] for ns in name_spans]
        for i, (s, e, numeric) in enumerate(name_spans):
            # bound: until next name start, or newline / ';', or +LOOKAHEAD
            hard = e + _LOOKAHEAD
            nl = text.find("\n", e)
            sc = text.find(";", e)
            nxt = name_spans[i + 1][0] if i + 1 < len(name_spans) else len(text)
            bound = min(x for x in [hard, nxt, len(text)] + ([nl] if nl != -1 else []) + ([sc] if sc != -1 else []))
            window = text[e:bound]

            val = self._first_value(text, e, window)
            if val is not None:
                ents.append(val)
                continue
            # textual result phrase right after the (non-numeric) test name
            res = self._result_phrase(text, e, window)
            if res is not None:
                ents.append(res)
        return ents

    def _first_value(self, text: str, name_end: int, window: str):
        m_conn = _CONNECTOR.match(window)
        off = m_conn.end() if m_conn else 0
        m = _VALUE.match(window, off) or _VALUE.search(window[:max(off + 30, 30)])
        if not m:
            return None
        vs = name_end + m.start("val")
        ve = name_end + m.end("val")
        # trim trailing spaces from unit capture
        while ve > vs and text[ve - 1].isspace():
            ve -= 1
        val_text = text[vs:ve]
        if not any(ch.isdigit() for ch in val_text):
            return None
        return Entity(val_text, vs, ve, TYPE_TEST_RESULT, source="lab:value")

    def _result_phrase(self, text: str, name_end: int, window: str):
        low = window.casefold()
        best = None
        for ph in self.result_phrases:
            idx = low.find(ph.casefold())
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, ph)
        if best is None:
            return None
        idx, ph = best
        s = name_end + idx
        e = s + len(ph)
        return Entity(text[s:e], s, e, TYPE_TEST_RESULT, source="lab:result_phrase")
