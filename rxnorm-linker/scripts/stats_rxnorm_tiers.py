"""Classify THUOC candidate assignments by RxNorm linker confidence tier."""
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
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    entities: list[tuple[str, dict]] = []
    for path in sorted(args.directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        entities.extend(
            (path.name, entity)
            for entity in data
            if entity.get("type") == "THUỐC"
        )

    linker = get_linker()
    counts: Counter[str] = Counter()
    candidate_counts: dict[str, Counter[int]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)

    for filename, entity in entities:
        text = entity["text"]
        output = entity.get("candidates", [])
        exact = linker.link_exact(text)
        lexical = linker.link(text, use_tier3=False)

        if exact:
            tier = "T1_EXACT_RXNORM"
        elif lexical and lexical[0].method == "nearest_dose":
            tier = "T2_NEAREST_DOSE"
        elif lexical:
            tier = "T2_LEXICAL_PARTIAL"
        elif output:
            tier = "T3_EMBEDDING_RERANK"
        else:
            tier = "EMPTY"

        counts[tier] += 1
        candidate_counts[tier][len(output)] += 1
        if len(examples[tier]) < args.examples:
            examples[tier].append(
                {"file": filename, "text": text, "candidates": output}
            )

    result = {
        "total": len(entities),
        "counts": dict(counts),
        "candidate_count_by_tier": {
            tier: dict(sorted(values.items()))
            for tier, values in candidate_counts.items()
        },
        "examples": dict(examples),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
