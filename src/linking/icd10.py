"""ICD-10 linker for CHẨN_ĐOÁN.

Primary path: icd-linker/ (teammate's ICD-10 TT06 knowledge base + deterministic
cascade — dictionary exact -> qualifier backoff -> fuzzy -> BGE-M3 embedding
fallback; see icd-linker/README.md). That package is self-contained under
icd-linker/src, with its data under icd-linker/data/processed — it is not a
pip-installed dependency, so we add its src/ to sys.path lazily on first load.

Falls back to the small manual alias seed list in data/diagnoses_vi_icd.yaml
if icd-linker's index is missing or its optional deps (torch, FlagEmbedding,
sentence-transformers) aren't installed, so --emit true never hard-crashes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..schema import DATA_DIR, Entity, TYPE_DIAGNOSIS
from .base import Linker

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICD_LINKER_SRC = _REPO_ROOT / "icd-linker" / "src"
_ICD_LINKER_INDEX = _REPO_ROOT / "icd-linker" / "data" / "processed"

# icd-linker's ICDMatcher.match(span, assertion=...) short-circuits to
# "no candidates" for these labels; our schema only produces "isNegated".
_NEGATED_ASSERTIONS = {"isNegated"}


def _build_real_matcher(embedding_mode: str):
    """Import + construct icd-linker's ICDMatcher. Raises on any problem
    (missing index files, missing optional deps) so callers can fall back."""
    if str(_ICD_LINKER_SRC) not in sys.path:
        sys.path.insert(0, str(_ICD_LINKER_SRC))
    from icd.matcher import ICDMatcher  # type: ignore  # noqa: E402

    matcher = ICDMatcher(index_dir=str(_ICD_LINKER_INDEX), embedding_mode=embedding_mode)
    if not matcher.index.by_code:
        raise FileNotFoundError(
            f"icd-linker index rỗng tại {_ICD_LINKER_INDEX} (thiếu data/processed/*.json?)"
        )
    return matcher


class ICD10Linker:
    def __init__(self, matcher=None, fallback: Optional["_SeedICDLinker"] = None):
        self.matcher = matcher
        self.fallback = fallback

    @classmethod
    def load(cls, path=None, embedding_mode: str = "fallback") -> "ICD10Linker":
        """embedding_mode: disabled | fallback (default) | rerank_fuzzy | replace_fuzzy
        — see icd-linker/README.md. "fallback" only touches BGE-M3 (needs
        torch+FlagEmbedding) for spans dictionary/fuzzy can't resolve."""
        fallback = _SeedICDLinker.load(path)
        try:
            matcher = _build_real_matcher(embedding_mode)
        except Exception as exc:
            print(f"[ICD10Linker] icd-linker không sẵn sàng ({exc}) -> dùng seed "
                  f"alias YAML ({DATA_DIR / 'diagnoses_vi_icd.yaml'}).", file=sys.stderr)
            matcher = None
        return cls(matcher=matcher, fallback=fallback)

    def link(self, e: Entity) -> List[str]:
        if e.type != TYPE_DIAGNOSIS:
            return []
        if self.matcher is None:
            return self.fallback.link(e)
        assertion = "negated" if set(e.assertions) & _NEGATED_ASSERTIONS else None
        try:
            result = self.matcher.match(e.text, assertion=assertion)
        except Exception as exc:
            print(f"[ICD10Linker] lỗi match '{e.text}': {exc}", file=sys.stderr)
            return self.fallback.link(e)
        if result.get("code"):
            return [result["code"]]
        # uncertain: surface the top candidate codes instead of nothing
        return [c["code"] for c in result.get("candidates", []) if c.get("code")][:5]


class _SeedICDLinker:
    """Original manual-alias fast-path — used only when icd-linker isn't
    available (missing index/deps)."""

    def __init__(self, surface_to_codes: Dict[str, List[str]]):
        self.linker = Linker(surface_to_codes)

    @classmethod
    def load(cls, path=None) -> "_SeedICDLinker":
        path = path or (DATA_DIR / "diagnoses_vi_icd.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        m = {it["surface"]: list(it.get("icd", [])) for it in data.get("diagnoses", [])}
        return cls(m)

    def link(self, e: Entity) -> List[str]:
        if e.candidates:  # seed codes already stashed by the extractor
            return list(e.candidates)
        return self.linker.lookup(e.text, e.canonical)
