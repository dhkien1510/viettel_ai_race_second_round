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


def fill_candidates(
    entities: list[dict],
    top_k: int = 3,
    exact_only: bool = False,
    use_tier3: bool = True,
    dose_fallback: str = "nearest",
    score_guided_manual: bool = False,
    adaptive_parser: bool = False,
) -> list[dict]:
    linker = get_linker()
    if score_guided_manual:
        for ent in entities:
            if ent.get("type") != "THUỐC":
                continue
            ent["candidates"] = linker.link_score_guided_rxcuis(ent.get("text", ""))
        return entities

    for ent in entities:
        if ent.get("type") != "THUỐC":
            continue
        if exact_only:
            ent["candidates"] = linker.link_exact_rxcuis(ent["text"])
        else:
            ent["candidates"] = linker.link_rxcuis(
                ent["text"], top_k=top_k, use_tier3=use_tier3,
                dose_fallback=dose_fallback, adaptive_parser=adaptive_parser,
            )
    return entities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="NER output .json files to update in place")
    ap.add_argument("--dir", help="process every .json file in this directory instead")
    ap.add_argument("--out", help="write results here instead of overwriting inputs "
                                   "(mirrors input filenames)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--exact-only", action="store_true",
                     help="skip embedding/rerank tier entirely; only keep a candidate when "
                          "the RxNorm entry's tokens+strength match the query exactly, "
                          "otherwise leave candidates empty")
    ap.add_argument("--no-tier3", action="store_true",
                     help="disable the SapBERT embedding + reranker fallback tier entirely -- "
                          "a term with no lexical dictionary match stays empty instead of "
                          "getting a fuzzy guess (typos, OOV/informal names, glued names)")
    ap.add_argument("--dose-fallback", choices=["nearest", "bare"], default="nearest",
                     help="when the stated dose isn't a real marketed strength: 'nearest' picks "
                          "the closest existing strength (default); 'bare' drops the dose and "
                          "falls back to the bare ingredient/brand candidate instead")
    ap.add_argument("--score-guided-manual", action="store_true",
                    help="use the score-guided conservative one-candidate heuristic; "
                         "no per-submission text map is used")
    ap.add_argument("--adaptive-parser", action="store_true",
                    help="derive identity tokens from the RxNorm vocabulary instead of "
                         "the dataset-specific stopword list")
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
        fill_candidates(
            entities,
            top_k=args.top_k,
            exact_only=args.exact_only,
            use_tier3=not args.no_tier3,
            dose_fallback=args.dose_fallback,
            score_guided_manual=args.score_guided_manual,
            adaptive_parser=args.adaptive_parser,
        )
        target = out_dir / path.name if out_dir else path
        target.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {target}")


if __name__ == "__main__":
    main()
