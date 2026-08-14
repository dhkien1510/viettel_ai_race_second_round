"""Analyze only ground-truth drug entities that have an assigned candidate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.linker import get_linker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.ground_truth.glob("*.json")):
        for entity in json.loads(path.read_text(encoding="utf-8")):
            if entity.get("type") == "THUỐC" and entity.get("candidates"):
                rows.append((path.name, entity))

    linker = get_linker()
    tiers = Counter()
    matches = Counter()
    cardinality = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)

    for filename, entity in rows:
        text = entity["text"]
        gold = [str(code).strip() for code in entity["candidates"] if str(code).strip()]
        exact = linker.link_exact_rxcuis(text)
        lexical_candidates = linker.link(text, top_k=3, use_tier3=False)
        lexical = [candidate.rxcui for candidate in lexical_candidates]

        if exact:
            tier = "T1_EXACT"
            ours = exact
        elif lexical_candidates and lexical_candidates[0].method == "nearest_dose":
            tier = "T2_NEAREST_DOSE"
            ours = lexical
        elif lexical:
            tier = "T2_LEXICAL_PARTIAL"
            ours = lexical
        else:
            tier = "T3_REQUIRED_OR_UNRESOLVED"
            ours = []

        tiers[tier] += 1
        cardinality[len(entity["candidates"])] += 1
        gold_set, ours_set = set(gold), set(ours)
        if not gold_set:
            status = "INVALID_GOLD"
        elif gold_set == ours_set:
            status = "EXACT_SET_MATCH"
        elif gold_set & ours_set:
            status = "GOLD_PRESENT_IN_OURS"
        else:
            status = "NO_MATCH_WITH_T1_T2"
        matches[(tier, status)] += 1

        if len(examples[tier]) < args.examples:
            examples[tier].append(
                {
                    "file": filename,
                    "text": text,
                    "gold": gold,
                    "ours_t1_t2": ours,
                    "status": status,
                }
            )

    result = {
        "assigned_gold_drugs": len(rows),
        "gold_candidate_cardinality": dict(sorted(cardinality.items())),
        "tier_distribution": dict(tiers),
        "match_by_tier": {
            tier: {
                status: count
                for (item_tier, status), count in matches.items()
                if item_tier == tier
            }
            for tier in tiers
        },
        "examples": dict(examples),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
