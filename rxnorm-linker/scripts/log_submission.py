"""Append a row to submission/SCORES.md after each run of run_rxnorm.ps1.

Captures the git commit and the CONFIG knobs that matter for RxNorm tier-3
behavior automatically, so the log stays accurate without hand-copying
numbers. The "Điểm BTC" (score) column is always left as TBD -- there is no
local score, it comes back from the organizers after submission, and must be
filled in by hand once known.

Usage:
    python scripts/log_submission.py submission/0708/Cuong02 --note "tang min_rerank_confidence len 0.6"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.config import CONFIG

SCORES_PATH = ROOT.parent / "submission" / "SCORES.md"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _strategy_summary() -> str:
    parts = ["lexical (tier 1-2) -> SapBERT embedding fallback (tier 3)"]
    if CONFIG.enable_cross_encoder_rerank:
        parts.append(
            f"-> rerank bge-reranker-v2-m3 (rerank_weight={CONFIG.rerank_weight}, "
            f"min_rerank_confidence={CONFIG.min_rerank_confidence})"
        )
    else:
        parts.append("-> cross-encoder rerank DISABLED")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="thư mục nộp, vd: submission/0708/Cuong02")
    ap.add_argument("--note", default="", help="ghi chú ngắn về thay đổi so với lần trước")
    args = ap.parse_args()

    row = (
        f"| {date.today().isoformat()} | {args.folder} | {_git_commit()} | "
        f"{_strategy_summary()} | TBD | {args.note} |\n"
    )

    if not SCORES_PATH.exists():
        SCORES_PATH.write_text(
            "# Submission score log\n\n"
            "| Ngày | Thư mục nộp | Commit | Chiến lược RxNorm | Điểm BTC | Ghi chú |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )

    with SCORES_PATH.open("a", encoding="utf-8") as f:
        f.write(row)

    print(f"Logged -> {SCORES_PATH}")
    print(row.strip())


if __name__ == "__main__":
    main()
