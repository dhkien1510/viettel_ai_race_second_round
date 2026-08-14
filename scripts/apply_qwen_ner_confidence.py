"""Apply calibrated NER confidence guards and emit schema-clean entities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["types"]

    for path in sorted(debug_dir.glob("*.json"), key=lambda item: int(item.stem)):
        kept = []
        for item in json.loads(path.read_text(encoding="utf-8")):
            guard = thresholds.get(item["type"])
            if guard and float(item[guard["metric"]]) < float(guard["threshold"]):
                continue
            kept.append({key: value for key, value in item.items() if not key.startswith("_")})
        (output_dir / path.name).write_text(
            json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
