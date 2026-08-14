from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxnorm.build_index import CACHE_PATH, RRF_PATH
from rxnorm.build_index import build as build_conso_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RxNorm lookup index (+ optional embedding cache).")
    parser.add_argument("--rrf", default=str(RRF_PATH))
    parser.add_argument("--out", default=str(CACHE_PATH))
    parser.add_argument("--build-embeddings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rrf_path = Path(args.rrf)
    out_path = Path(args.out)
    if not rrf_path.exists():
        raise FileNotFoundError(f"RXNCONSO.RRF not found: {rrf_path}")
    build_conso_index(rrf_path=rrf_path, cache_path=out_path)
    if args.build_embeddings:
        from rxnorm.embed_index import build as build_embed_cache

        build_embed_cache(conso_cache_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
