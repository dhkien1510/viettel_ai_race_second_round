"""Compare RxNorm candidate sets in two submission-format directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def entity_key(entity: dict) -> tuple:
    return (tuple(entity.get("position", ())), entity.get("type"), entity.get("text"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()

    total = same = empty_to_filled = filled_to_empty = changed_code = 0
    jaccard_sum = 0.0
    baseline_covered = trial_covered = 0
    changes: list[dict] = []
    for baseline_path in sorted(args.baseline.glob("*.json")):
        trial_path = args.trial / baseline_path.name
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        trial = {entity_key(e): e for e in json.loads(trial_path.read_text(encoding="utf-8"))}
        for entity in baseline:
            if not str(entity.get("type", "")).startswith("THU"):
                continue
            other = trial[entity_key(entity)]
            left = set(map(str, entity.get("candidates") or []))
            right = set(map(str, other.get("candidates") or []))
            total += 1
            baseline_covered += bool(left)
            trial_covered += bool(right)
            union = left | right
            jaccard_sum += len(left & right) / len(union) if union else 1.0
            if left == right:
                same += 1
                continue
            if not left and right:
                empty_to_filled += 1
            elif left and not right:
                filled_to_empty += 1
            else:
                changed_code += 1
            changes.append({"file": baseline_path.name, "text": entity.get("text"),
                            "baseline": sorted(left), "trial": sorted(right)})

    result = {
        "drug_entities": total,
        "same": same,
        "same_percent": 100 * same / total,
        "different": total - same,
        "different_percent": 100 * (total - same) / total,
        "mean_jaccard_percent": 100 * jaccard_sum / total,
        "baseline_coverage_percent": 100 * baseline_covered / total,
        "trial_coverage_percent": 100 * trial_covered / total,
        "empty_to_filled": empty_to_filled,
        "filled_to_empty": filled_to_empty,
        "changed_code": changed_code,
        "changes": changes,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
