"""Compare RxNorm candidate-selection strategies against a JSON GT directory."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.linker import get_linker  # noqa: E402


def jaccard(gold: list[str], prediction: list[str]) -> float:
    left, right = set(map(str, gold)), set(map(str, prediction))
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--skip-tier3", action="store_true")
    args = parser.parse_args()

    entities = []
    for path in sorted(args.ground_truth.glob("*.json")):
        entities.extend(
            entity
            for entity in json.loads(path.read_text(encoding="utf-8"))
            if entity.get("type") == "THUỐC"
        )

    linker = get_linker()
    totals: Counter[str] = Counter()
    exact_hits: Counter[str] = Counter()
    cardinalities: dict[str, Counter[int]] = {}
    names = ["all_empty", "exact_only", "lexical_top1"]
    if not args.skip_tier3:
        names += ["default_top1", "default_top3"]
    for name in names:
        cardinalities[name] = Counter()

    for entity in entities:
        text = entity["text"]
        gold = entity.get("candidates", [])
        exact = linker.link_exact_rxcuis(text)
        lexical = linker.link_rxcuis(text, top_k=1, use_tier3=False)
        predictions = {
            "all_empty": [],
            "exact_only": exact[:1],
            "lexical_top1": lexical[:1],
        }
        if not args.skip_tier3:
            default = linker.link_rxcuis(text, top_k=3, use_tier3=True)
            predictions["default_top1"] = default[:1]
            predictions["default_top3"] = default
        for name, prediction in predictions.items():
            score = jaccard(gold, prediction)
            totals[name] += score
            cardinalities[name][len(prediction)] += 1
            if set(map(str, gold)) == set(map(str, prediction)):
                exact_hits[name] += 1

    result = {
        "drug_entities": len(entities),
        "strategies": {
            name: {
                "mean_jaccard": totals[name] / len(entities),
                "exact_sets": exact_hits[name],
                "candidate_cardinality": dict(sorted(cardinalities[name].items())),
            }
            for name in names
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
