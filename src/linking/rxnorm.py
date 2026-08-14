"""RxNorm linker for THUỐC.

Primary path: rxnorm-linker/ (teammate's token-overlap + strength scoring
cascade over the real RxNorm release, with a sentence-embedding fallback for
typos/synonyms; see rxnorm-linker/README.md). Self-contained under
rxnorm-linker/src — not pip-installed, so its src/ is added to sys.path lazily.

That package needs data/rxnorm/rrf/RXNCONSO.RRF (the official RxNorm RRF
release, UMLS-licensed — not shipped in this repo) to build its index. Until
that file is supplied, this falls back to the small manual alias seed list in
data/drug_alias_rxnorm.yaml so --emit true never hard-crashes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..schema import DATA_DIR, Entity, TYPE_DRUG
from .base import Linker

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RXNORM_LINKER_SRC = _REPO_ROOT / "rxnorm-linker" / "src"
_RXNORM_CACHE = _REPO_ROOT / "rxnorm-linker" / "data" / "rxnorm" / "cache" / "rxnorm_index.pkl"


def _build_real_linker():
    """Import + construct rxnorm-linker's RxNormLinker. Raises (missing RRF
    source / cache, or missing optional embedding deps) so callers fall back."""
    if str(_RXNORM_LINKER_SRC) not in sys.path:
        sys.path.insert(0, str(_RXNORM_LINKER_SRC))
    from rxnorm.linker import RxNormLinker as _RealRxNormLinker  # type: ignore  # noqa: E402

    return _RealRxNormLinker(cache_path=_RXNORM_CACHE)


class RxNormLinker:
    def __init__(self, linker=None, fallback: Optional["_SeedRxLinker"] = None):
        self.linker = linker
        self.fallback = fallback

    @classmethod
    def load(cls, path=None) -> "RxNormLinker":
        fallback = _SeedRxLinker.load(path)
        try:
            real = _build_real_linker()
        except Exception as exc:
            print(f"[RxNormLinker] rxnorm-linker không sẵn sàng ({exc}) -> cần "
                  f"data/rxnorm/rrf/RXNCONSO.RRF (UMLS), dùng seed alias YAML "
                  f"({DATA_DIR / 'drug_alias_rxnorm.yaml'}) tạm thời.", file=sys.stderr)
            real = None
        return cls(linker=real, fallback=fallback)

    def link(self, e: Entity) -> List[str]:
        if e.type != TYPE_DRUG:
            return []
        if self.linker is None:
            return self.fallback.link(e)
        try:
            return self.linker.link_rxcuis(e.text, top_k=3)
        except Exception as exc:
            print(f"[RxNormLinker] lỗi link '{e.text}': {exc}", file=sys.stderr)
            return self.fallback.link(e)


class _SeedRxLinker:
    """Original manual-alias fast-path — used only when rxnorm-linker's real
    RxNorm data (RXNCONSO.RRF) isn't available yet."""

    def __init__(self, canon_to_codes: Dict[str, List[str]]):
        self.linker = Linker(canon_to_codes)

    @classmethod
    def load(cls, path=None) -> "_SeedRxLinker":
        path = path or (DATA_DIR / "drug_alias_rxnorm.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        m: Dict[str, List[str]] = {}
        for d in data.get("drugs", []):
            m[d["canonical"]] = list(d.get("rxcui", []))
        return cls(m)

    def link(self, e: Entity) -> List[str]:
        return self.linker.lookup(e.text, e.canonical)
