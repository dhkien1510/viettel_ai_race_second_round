from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.linker import RxNormLinker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tra cứu mã RxNorm từ tên thuốc (± liều/đường/tần suất)")
    parser.add_argument("--span", action="append", help="Tên thuốc cần tra cứu")
    parser.add_argument("--input-file", help="File text, mỗi dòng một tên thuốc")
    parser.add_argument("--cache-path", default="data/rxnorm/cache/rxnorm_index.pkl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-jsonl", default=None)
    return parser.parse_args()


def collect_spans(args: argparse.Namespace) -> list[str]:
    spans = list(args.span or [])
    if args.input_file:
        path = Path(args.input_file)
        spans.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not spans:
        raise ValueError("Cần cung cấp --span hoặc --input-file")
    return spans


def _needs_review(candidates: list[dict]) -> bool:
    return not candidates or candidates[0]["method"] == "embedding" or candidates[0]["score"] < 0.75


def main() -> int:
    args = parse_args()
    linker = RxNormLinker(cache_path=Path(args.cache_path))
    rows = []
    for span in collect_spans(args):
        candidates = [
            {"rxcui": c.rxcui, "tty": c.tty, "str": c.str_, "score": c.score, "method": c.method}
            for c in linker.link(span, top_k=args.top_k)
        ]
        row = {
            "input": span,
            "candidates": candidates,
            "needs_review": _needs_review(candidates),
        }
        if not args.debug:
            row = {
                "input": span,
                "rxcuis": [c["rxcui"] for c in candidates],
                "needs_review": row["needs_review"],
            }
        rows.append(row)

    text = json.dumps(rows[0] if len(rows) == 1 else rows, ensure_ascii=False, indent=2)
    print(text)

    if args.debug_jsonl:
        out = Path(args.debug_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
