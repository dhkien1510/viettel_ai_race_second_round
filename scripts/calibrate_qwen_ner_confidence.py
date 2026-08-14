"""Fit type-specific NER confidence guards on a held-out document split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("_confidence_mean", "_confidence_min", "_margin_mean")


def load_ids(path: Path, subset: str) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(value) for value in data[subset]]


def f1_at_threshold(predictions: list[dict], references: set[tuple], metric: str, threshold: float) -> tuple:
    kept = [item for item in predictions if float(item[metric]) >= threshold]
    predicted = {(item["document_id"], tuple(item["position"]), item["type"]) for item in kept}
    true_positive = len(predicted & references)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(references) if references else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall, len(predicted), true_positive


def candidate_thresholds(items: list[dict], metric: str, limit: int = 200) -> list[float]:
    values = sorted({float(item[metric]) for item in items})
    if len(values) <= limit:
        return [float("-inf"), *values]
    indexes = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
    return [float("-inf"), *(values[index] for index in sorted(indexes))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset", default="validation", choices=("train", "validation"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    reference_dir = Path(args.reference_dir)
    ids = load_ids(Path(args.split), args.subset)
    predictions: list[dict] = []
    references_by_type: dict[str, set[tuple]] = {}

    for document_id in ids:
        for item in json.loads((debug_dir / f"{document_id}.json").read_text(encoding="utf-8")):
            predictions.append({"document_id": document_id, **item})
        for item in json.loads((reference_dir / f"{document_id}.json").read_text(encoding="utf-8")):
            references_by_type.setdefault(item["type"], set()).add(
                (document_id, tuple(item["position"]), item["type"])
            )

    report = {"split": str(args.split), "subset": args.subset, "types": {}}
    for entity_type in sorted(references_by_type):
        type_predictions = [item for item in predictions if item["type"] == entity_type]
        references = references_by_type[entity_type]
        best = None
        for metric in METRICS:
            for threshold in candidate_thresholds(type_predictions, metric):
                result = f1_at_threshold(type_predictions, references, metric, threshold)
                candidate = (result[0], result[1], result[2], -result[3], metric, threshold, result)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
        assert best is not None
        result = best[-1]
        report["types"][entity_type] = {
            "metric": best[4],
            "threshold": best[5],
            "f1": result[0],
            "precision": result[1],
            "recall": result[2],
            "predicted": result[3],
            "true_positive": result[4],
            "reference": len(references),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
