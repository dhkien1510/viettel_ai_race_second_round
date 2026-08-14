"""Validate a 100-file submission directory against raw input."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schema import VALID_ASSERTIONS, VALID_TYPES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--expected-files", type=int, default=100)
    parser.add_argument("--fail-on-text-mismatch", action="store_true")
    args = parser.parse_args()
    input_dir = Path(args.input)
    prediction_dir = Path(args.prediction)
    report = Counter()
    errors = []
    files = sorted(prediction_dir.glob("*.json"), key=lambda item: int(item.stem))
    report["json_files"] = len(files)
    for path in files:
        document_id = path.stem
        raw_path = input_dir / f"{document_id}.txt"
        if not raw_path.exists():
            report["missing_raw"] += 1
            errors.append({"file": document_id, "error": "missing_raw"})
            continue
        raw = raw_path.read_text(encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            report["schema_errors"] += 1
            errors.append({"file": document_id, "error": "json_not_list"})
            continue
        seen = set()
        for index, item in enumerate(payload):
            report["entities"] += 1
            position = item.get("position")
            if not isinstance(position, list) or len(position) != 2:
                report["schema_errors"] += 1
                errors.append({"file": document_id, "index": index, "error": "bad_position"})
                continue
            start, end = position
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(raw)):
                report["invalid_positions"] += 1
                errors.append({"file": document_id, "index": index, "error": "invalid_position"})
                continue
            if raw[start:end] != item.get("text"):
                report["text_mismatches"] += 1
                errors.append({"file": document_id, "index": index, "error": "text_mismatch"})
            if item.get("type") not in VALID_TYPES:
                report["invalid_types"] += 1
                errors.append({"file": document_id, "index": index, "error": "invalid_type"})
            if not isinstance(item.get("assertions", []), list) or any(value not in VALID_ASSERTIONS for value in item.get("assertions", [])):
                report["invalid_assertions"] += 1
                errors.append({"file": document_id, "index": index, "error": "invalid_assertions"})
            if not isinstance(item.get("candidates", []), list):
                report["schema_errors"] += 1
                errors.append({"file": document_id, "index": index, "error": "invalid_candidates"})
            key = (start, end, item.get("type"), item.get("text"))
            if key in seen:
                report["duplicate_exact_entities"] += 1
                errors.append({"file": document_id, "index": index, "error": "duplicate_exact_entity"})
            seen.add(key)
    print(json.dumps({**dict(report), "errors": errors[:20]}, ensure_ascii=False, indent=2))
    fatal = report["json_files"] != args.expected_files or report["schema_errors"] or report["invalid_positions"] or report["invalid_types"]
    if args.fail_on_text_mismatch:
        fatal = fatal or report["text_mismatches"]
    if fatal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
