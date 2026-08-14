"""Evaluate text/span/type and assertion fit against a reference label directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ASSERTABLE_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}


def load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected list")
    return payload


def key(item: dict[str, Any]) -> tuple[tuple[int, int], str, str]:
    return (tuple(int(value) for value in item["position"]), item["type"], item["text"])


def f1(true_positive: int, predicted: int, reference: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "reference": reference,
        "precision": precision,
        "recall": recall,
        "f1": score,
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--expected-files", type=int, default=100)
    args = parser.parse_args()

    prediction_dir = Path(args.prediction)
    reference_dir = Path(args.reference)
    counts = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    assertion_scores: list[float] = []
    assertion_exact = 0

    for document_id in map(str, range(1, args.expected_files + 1)):
        pred_items = load(prediction_dir / f"{document_id}.json")
        ref_items = load(reference_dir / f"{document_id}.json")
        pred_map = {key(item): item for item in pred_items}
        ref_map = {key(item): item for item in ref_items}
        pred_keys = set(pred_map)
        ref_keys = set(ref_map)
        matches = pred_keys & ref_keys
        counts["predicted_entities"] += len(pred_keys)
        counts["reference_entities"] += len(ref_keys)
        counts["matched_entities"] += len(matches)
        for item_key in pred_keys:
            by_type[item_key[1]]["predicted"] += 1
        for item_key in ref_keys:
            by_type[item_key[1]]["reference"] += 1
        for item_key in matches:
            by_type[item_key[1]]["matched"] += 1
            if item_key[1] not in ASSERTABLE_TYPES:
                continue
            predicted_assertions = set(pred_map[item_key].get("assertions") or [])
            reference_assertions = set(ref_map[item_key].get("assertions") or [])
            assertion_scores.append(jaccard(predicted_assertions, reference_assertions))
            assertion_exact += predicted_assertions == reference_assertions

    span_type_text = f1(
        counts["matched_entities"],
        counts["predicted_entities"],
        counts["reference_entities"],
    )
    report = {
        "files": args.expected_files,
        "span_type_text": span_type_text,
        "by_type": {
            entity_type: f1(values["matched"], values["predicted"], values["reference"])
            for entity_type, values in sorted(by_type.items())
        },
        "assertion_on_matched_assertable_entities": {
            "rows": len(assertion_scores),
            "exact": assertion_exact / len(assertion_scores) if assertion_scores else 0.0,
            "jaccard": sum(assertion_scores) / len(assertion_scores) if assertion_scores else 0.0,
        },
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
