"""Evaluate candidate-code agreement on exact-matched entity spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def entity_map(path: Path, entity_type: str) -> dict[tuple[int, int, str], set[str]]:
    entities = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(item["position"][0]), int(item["position"][1]), item["type"]): {
            str(code) for code in item.get("candidates", [])
        }
        for item in entities
        if item.get("type") == entity_type
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--type", required=True, dest="entity_type")
    parser.add_argument("--output")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction)
    reference_dir = Path(args.reference)
    matched = exact = both_empty = 0
    jaccard_sum = 0.0
    predicted_codes: set[tuple[str, tuple[int, int, str], str]] = set()
    reference_codes: set[tuple[str, tuple[int, int, str], str]] = set()
    prediction_entities = reference_entities = 0

    for ref_path in sorted(reference_dir.glob("*.json"), key=lambda path: int(path.stem)):
        pred_path = prediction_dir / ref_path.name
        if not pred_path.exists():
            continue
        pred = entity_map(pred_path, args.entity_type)
        ref = entity_map(ref_path, args.entity_type)
        prediction_entities += len(pred)
        reference_entities += len(ref)
        for key in pred.keys() & ref.keys():
            matched += 1
            pred_set = pred[key]
            ref_set = ref[key]
            exact += pred_set == ref_set
            both_empty += not pred_set and not ref_set
            union = pred_set | ref_set
            jaccard_sum += len(pred_set & ref_set) / len(union) if union else 1.0
            predicted_codes.update((ref_path.stem, key, code) for code in pred_set)
            reference_codes.update((ref_path.stem, key, code) for code in ref_set)

    true_positive = len(predicted_codes & reference_codes)
    precision = true_positive / len(predicted_codes) if predicted_codes else 0.0
    recall = true_positive / len(reference_codes) if reference_codes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "type": args.entity_type,
        "prediction_entities": prediction_entities,
        "reference_entities": reference_entities,
        "exact_matched_entities": matched,
        "exact_candidate_set": exact,
        "exact_candidate_set_accuracy": exact / matched if matched else 0.0,
        "both_candidate_sets_empty": both_empty,
        "macro_jaccard": jaccard_sum / matched if matched else 0.0,
        "micro_candidates": {
            "true_positive": true_positive,
            "predicted": len(predicted_codes),
            "reference": len(reference_codes),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
