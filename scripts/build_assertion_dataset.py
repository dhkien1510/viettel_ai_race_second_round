"""Build multi-label assertion classifier rows from entity labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_qwen_assertion_classifier import LABELS  # noqa: E402
from src.model.qwen_verifier_dataset import build_inference_row  # noqa: E402
from src.schema import ASSERTABLE_TYPES  # noqa: E402


def load_split(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "folds" in data:
        fold = data["folds"]["0"]
        validation = set(str(value) for value in fold["validation"])
        train = set(str(value) for value in fold["train"])
        return train, validation
    return set(str(value) for value in data.get("train", [])), set(str(value) for value in data.get("validation", []))


def build_rows(input_dir: Path, labels_dir: Path) -> list[dict]:
    rows = []
    for label_path in sorted(labels_dir.glob("*.json"), key=lambda item: int(item.stem)):
        document_id = label_path.stem
        text = (input_dir / f"{document_id}.txt").read_text(encoding="utf-8")
        items = json.loads(label_path.read_text(encoding="utf-8"))
        for item in items:
            if item.get("type") not in ASSERTABLE_TYPES:
                continue
            row = build_inference_row(document_id, text, item, "phase1-assertion-gold")
            values = set(item.get("assertions") or [])
            row.update({
                "labels": [1 if label in values else 0 for label in LABELS],
                "loss_weight": 1.0,
                "source": "phase1_best",
            })
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--context-chars", type=int, default=500)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    args = parser.parse_args()
    rows = build_rows(Path(args.input), Path(args.labels))
    train_ids, validation_ids = load_split(Path(args.split_manifest) if args.split_manifest else None)
    if not train_ids and not validation_ids:
        validation_ids = {row["file_id"] for index, row in enumerate(rows) if index % 5 == 0}
        train_ids = {row["file_id"] for row in rows} - validation_ids
    train = [row for row in rows if row["file_id"] in train_ids]
    validation = [row for row in rows if row["file_id"] in validation_ids]
    for path, values in ((Path(args.train_output), train), (Path(args.validation_output), validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "train": len(train), "validation": len(validation)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
