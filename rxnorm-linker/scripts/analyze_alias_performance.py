"""Measure manual-alias and branded-candidate agreement with a GT directory."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.linker import get_linker  # noqa: E402
from rxnorm.normalize import (  # noqa: E402
    INGREDIENT_ALIASES, PHRASE_ALIASES, _WORD_RE, _strip_accents,
)

# Strict branded concept types. PSN/SY/TMSY are representations/synonyms and
# can describe generic products, so counting them as trade names inflates the
# branded group.
BRANDED_TTYS = {"BN", "SBD", "SBDC", "SBDF", "SBDG", "SBDGP", "SBDFP"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ground_truth", type=Path)
    args = ap.parse_args()
    linker = get_linker()
    rows = []
    for path in args.ground_truth.glob("*.json"):
        for entity in json.loads(path.read_text(encoding="utf-8")):
            if not str(entity.get("type", "")).startswith("THU"):
                continue
            text = entity.get("text", "")
            normalized = _strip_accents(text.lower())
            words = _WORD_RE.findall(normalized)
            manual = any(re.search(rf"\b{re.escape(p)}\b", normalized) for p in PHRASE_ALIASES)
            manual = manual or any(word in INGREDIENT_ALIASES for word in words)
            candidates = linker.link(text, top_k=1, use_tier3=False)
            predicted = [candidate.rxcui for candidate in candidates]
            gold = list(map(str, entity.get("candidates") or []))
            branded = bool(candidates and candidates[0].tty in BRANDED_TTYS)
            rows.append({"manual": manual, "branded": branded,
                         "same": set(gold) == set(predicted), "text": text,
                         "gold": gold, "predicted": predicted,
                         "tty": candidates[0].tty if candidates else ""})

    result = {}
    for name, predicate in {
        "manual_alias": lambda row: row["manual"],
        "branded_prediction": lambda row: row["branded"],
        "alias_or_branded": lambda row: row["manual"] or row["branded"],
    }.items():
        group = [row for row in rows if predicate(row)]
        mismatches = [row for row in group if not row["same"]]
        result[name] = {
            "entities": len(group),
            "same": len(group) - len(mismatches),
            "same_percent": 100 * (len(group) - len(mismatches)) / len(group) if group else 0,
            "different": len(mismatches),
            "different_percent": 100 * len(mismatches) / len(group) if group else 0,
            "mismatches": mismatches,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
