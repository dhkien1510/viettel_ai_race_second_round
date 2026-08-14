"""Drug extractor.

Dictionary of ingredient/brand names matched at word starts, then the span is
widened to include an adjacent dose/route/frequency ("metoprolol 25mg po bid").
Glued names produced by translation errors ("doxycyclinebactrim",
"ciproflagyl", "atenololtrong") are handled by CHAINING: after a matched drug
that is immediately followed by letters, we try to match another drug exactly
at that boundary. This splits two glued drugs without opening the door to
matching drug substrings inside arbitrary Vietnamese words.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import yaml

from ..schema import DATA_DIR, Entity, TYPE_DRUG
from ..offsets import is_word_char


class DrugExtractor:
    def __init__(self, cfg: dict):
        self.alias_to_canon: Dict[str, str] = {}
        for d in cfg.get("drugs", []):
            canon = d["canonical"]
            for a in d.get("aliases", [canon]):
                self.alias_to_canon[a.casefold()] = canon
        # aliases sorted longest-first for greedy matching
        self.aliases = sorted(self.alias_to_canon.keys(), key=len, reverse=True)

        units = cfg.get("dose_units", [])
        routes = cfg.get("routes", [])
        freqs = cfg.get("frequencies", [])
        toks = sorted(set(units + routes + freqs), key=len, reverse=True)
        tok_alt = "|".join(re.escape(t) for t in toks)
        # a tail chunk: number+unit, or a route/freq token, or "x 1", "/ngày".
        # NOTE: no leading ^ — this is used with .match(text, pos) which already
        # anchors at pos; a ^ would only match at string start and fail.
        self._tail = re.compile(
            r"\s*(?:"
            r"\d+(?:[.,]\d+)?\s*%?\s*(?:" + tok_alt + r")?"  # 25 / 25mg / 0.9 % / 2.5 mg
            r"|(?:" + tok_alt + r")"                          # po / bid / daily
            r"|x\s*\d+"                                       # x 1
            r"|/\s*(?:ngày|ml)"                               # /ngày /ml
            r"|\d+\s*/\s*\d+"                                 # 130/72
            r")",
            re.IGNORECASE,
        )

    @classmethod
    def load(cls, path=None) -> "DrugExtractor":
        path = path or (DATA_DIR / "drug_alias_rxnorm.yaml")
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def _match_drug_at(self, text_cf: str, pos: int) -> Optional[Tuple[str, str]]:
        """Return (alias, canonical) if a known drug alias starts exactly at pos."""
        for a in self.aliases:
            if text_cf.startswith(a, pos):
                return a, self.alias_to_canon[a]
        return None

    def _widen_tail(self, text: str, end: int) -> int:
        """Greedily consume adjacent dose/route/freq chunks after `end`."""
        cur = end
        while True:
            m = self._tail.match(text, cur)
            if not m or m.end() == cur:
                break
            cur = m.end()
        # do not leave a trailing space in the span
        while cur > end and text[cur - 1].isspace():
            cur -= 1
        return cur

    def extract(self, text: str) -> List[Entity]:
        text_cf = text.casefold()
        ents: List[Entity] = []
        i = 0
        n = len(text)
        while i < n:
            # only start a match at a word start
            if is_word_char(text[i]) and (i == 0 or not is_word_char(text[i - 1])):
                hit = self._match_drug_at(text_cf, i)
                if hit:
                    alias, canon = hit
                    core_end = i + len(alias)
                    span_end = self._widen_tail(text, core_end)
                    ents.append(Entity(text[i:span_end], i, span_end, TYPE_DRUG,
                                       source="drug", canonical=canon))
                    # chain glued drug names: another drug immediately after core
                    j = core_end
                    while j < n and is_word_char(text[j]):
                        nxt = self._match_drug_at(text_cf, j)
                        if not nxt:
                            break
                        a2, c2 = nxt
                        e2_core = j + len(a2)
                        e2 = self._widen_tail(text, e2_core)
                        ents.append(Entity(text[j:e2], j, e2, TYPE_DRUG,
                                           source="drug:glued", canonical=c2))
                        j = e2_core
                    # advance past the whole run
                    i = max(span_end, j)
                    continue
            i += 1
        return ents
