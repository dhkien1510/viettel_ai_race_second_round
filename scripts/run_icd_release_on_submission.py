"""Apply icd-linker-release to diagnosis entities in a submission directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_TYPES = {"CHẨN_ĐOÁN", "CHAN_DOAN"}


def context_window(note: str, start: int, end: int, window: int) -> tuple[str, str]:
    before_start = max(0, start - window)
    after_end = min(len(note), end + window)
    return note[before_start:start], note[end:after_end]


def load_linker(release_dir: Path, model_dir: Path, data_dir: Path, device: str | None):
    sys.path.insert(0, str(release_dir))
    from predict import ICDLinker  # noqa: WPS433

    return ICDLinker(data_dir=data_dir, model_dir=model_dir, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="JSON directory after Qwen NER + assertion.")
    parser.add_argument("--input-dir", required=True, help="Directory containing <id>.txt notes.")
    parser.add_argument("--output-dir", required=True, help="Directory for ICD-enriched JSON files.")
    parser.add_argument("--release-dir", default="icd-linker-release")
    parser.add_argument("--model-dir", default="icd-linker-release/models/reranker")
    parser.add_argument("--data-dir", default="icd-linker-release/data/processed")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto.")
    parser.add_argument("--context-window", type=int, default=1200)
    parser.add_argument("--expected-files", type=int, default=100)
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    release_dir = Path(args.release_dir)
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    if not (model_dir / "best.pt").exists():
        raise FileNotFoundError(
            f"Missing ICD reranker checkpoint: {model_dir / 'best.pt'}. "
            "Run icd-linker-release/train.py first."
        )

    linker = load_linker(release_dir, model_dir, data_dir, args.device)
    stats: dict[str, Any] = {
        "files": 0,
        "entities": 0,
        "diagnoses": 0,
        "diagnoses_with_icd": 0,
        "none_predictions": 0,
    }

    for document_id in map(str, range(1, args.expected_files + 1)):
        source_path = source_dir / f"{document_id}.json"
        text_path = input_dir / f"{document_id}.txt"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not text_path.exists():
            raise FileNotFoundError(text_path)

        note = text_path.read_text(encoding="utf-8")
        entities = json.loads(source_path.read_text(encoding="utf-8"))
        for entity in entities:
            stats["entities"] += 1
            entity.setdefault("candidates", [])
            if entity.get("type") not in DIAGNOSIS_TYPES:
                continue
            stats["diagnoses"] += 1
            position = entity.get("position") or []
            if len(position) != 2:
                entity["candidates"] = []
                stats["none_predictions"] += 1
                continue
            start, end = int(position[0]), int(position[1])
            before, after = context_window(note, start, end, args.context_window)
            code = linker.predict(text=str(entity.get("text", "")), before=before, after=after)
            if code and code != "NONE":
                entity["candidates"] = [code]
                stats["diagnoses_with_icd"] += 1
            else:
                entity["candidates"] = []
                stats["none_predictions"] += 1

        (output_dir / f"{document_id}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stats["files"] += 1

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
