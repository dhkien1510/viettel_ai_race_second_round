"""Validate a JSON directory shape and create a deterministic submission ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-files", type=int, default=100)
    parser.add_argument("--report")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    paths = [source / f"{index}.json" for index in range(1, args.expected_files + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} expected files: {missing[:5]}")

    unexpected = sorted(path.name for path in source.glob("*.json") if path not in paths)
    if unexpected:
        raise ValueError(f"Unexpected JSON files in source: {unexpected[:5]}")

    stats: Counter[str] = Counter()
    for path in paths:
        entities = json.loads(path.read_text(encoding="utf-8"))
        for entity in entities:
            entity_type = entity["type"]
            stats["entities"] += 1
            stats[f"type:{entity_type}"] += 1
            candidates = entity.get("candidates") or []
            if candidates:
                stats["entities_with_candidates"] += 1
                stats[f"with_candidates:{entity_type}"] += 1
                stats["candidate_codes"] += len(candidates)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    expected_names = [path.name for path in paths]
    if names != expected_names:
        raise ValueError("ZIP member order or root paths do not match the expected submission layout")

    report = {
        "source": str(source),
        "archive": str(output),
        "archive_size_bytes": output.stat().st_size,
        "json_files": len(paths),
        "stats": dict(sorted(stats.items())),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
