"""Fill in the `candidates` field of THUỐC entities in NER output JSON.

Consumes/produces the exact list-of-dict format from vong_1_so_loai_quy_dinh.md:
  [{"text": ..., "type": "THUỐC", "candidates": [...], "assertions": [...], "position": [...]}, ...]

Usage:
    python scripts/label_rxnorm_candidates.py path/to/1.json [path/to/2.json ...]
    python scripts/label_rxnorm_candidates.py --dir path/to/ner_output --out path/to/output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.linker import get_linker


def extract_context_line(note_text: str, start: int, end: int) -> str:
    # Extract the line containing the entity, plus the line immediately before and after it.
    line_start = note_text.rfind('\n', 0, start) + 1
    line_end = note_text.find('\n', end)
    if line_end == -1:
        line_end = len(note_text)
    
    prev_line_start = note_text.rfind('\n', 0, max(0, line_start - 2)) + 1
    next_line_end = note_text.find('\n', line_end + 1)
    if next_line_end == -1:
        next_line_end = len(note_text)
        
    return note_text[prev_line_start:next_line_end]


def fill_candidates(entities: list[dict], note_text: str | None = None, top_k: int = 1, strategy: str = "most_specific") -> list[dict]:
    linker = get_linker()
    for ent in entities:
        if ent.get("type") != "THUỐC":
            continue
        pos = ent.get("position")
        local_context = None
        if note_text and pos and len(pos) == 2:
            start, end = pos
            local_context = extract_context_line(note_text, start, end)
        ent["candidates"] = linker.link_rxcuis(ent["text"], top_k=top_k, context=local_context, strategy=strategy)
    return entities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="NER output .json files to update in place")
    ap.add_argument("--dir", help="process every .json file in this directory instead")
    ap.add_argument("--out", help="write results here instead of overwriting inputs "
                                   "(mirrors input filenames)")
    ap.add_argument("--input-dir", help="directory containing <id>.txt notes for context-aware linking")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--strategy", choices=["most_specific", "ingredient_only", "generic_only", "surface_form", "conservative"], default="most_specific",
                    help="priority selection strategy for TTYs")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    if args.dir:
        files += sorted(Path(args.dir).glob("*.json"))
    if not files:
        ap.error("no input files given (pass paths or --dir)")

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        entities = json.loads(path.read_text(encoding="utf-8"))
        
        # Search for corresponding text note
        note_name = path.stem + ".txt"
        note_text = None
        candidate_dirs = [Path(args.input_dir)] if args.input_dir else []
        candidate_dirs.extend([
            Path("data/input-part2-real/input"),
            Path("../data/input-part2-real/input"),
            Path("../../data/input-part2-real/input"),
            Path("data/input"),
            Path("../data/input"),
            Path("../../data/input"),
            ROOT / "data" / "input-part2-real" / "input",
            ROOT / "data" / "input",
        ])
        for candidate_dir in candidate_dirs:
            note_path = candidate_dir / note_name
            if note_path.exists():
                note_text = note_path.read_text(encoding="utf-8")
                break

        fill_candidates(entities, note_text=note_text, top_k=args.top_k, strategy=args.strategy)
        target = out_dir / path.name if out_dir else path
        target.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {target} (context loaded: {note_text is not None})")


if __name__ == "__main__":
    main()
