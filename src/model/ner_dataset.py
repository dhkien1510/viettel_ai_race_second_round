"""Build a character-level BIO dataset from gold JSON annotations.

Gold format: for each input file `<id>.txt` a `<id>.json` with the same list of
{text, position:[s,e], type, ...}. We convert to per-character BIO tags so the
offsets are preserved exactly (no tokenizer-induced offset drift). The training
script maps these onto subword tokens using the fast tokenizer's offset
mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from ..schema import VALID_TYPES
from ..offsets import read_text


def char_bio(text: str, items: List[dict]) -> List[str]:
    """Return a per-character BIO tag list aligned to `text`."""
    tags = ["O"] * len(text)
    # longer spans first so nested/duplicate shorter spans don't overwrite
    for it in sorted(items, key=lambda x: x["position"][1] - x["position"][0], reverse=True):
        s, e = it["position"]
        typ = it["type"]
        if typ not in VALID_TYPES:
            continue
        if any(tags[i] != "O" for i in range(s, e)):
            continue
        tags[s] = f"B-{typ}"
        for i in range(s + 1, e):
            tags[i] = f"I-{typ}"
    return tags


def load_gold_pairs(input_dir: Path, gold_dir: Path) -> List[Tuple[str, List[dict], List[str]]]:
    out = []
    for gjson in sorted(gold_dir.glob("*.json")):
        txt = input_dir / (gjson.stem + ".txt")
        if not txt.exists():
            continue
        text = read_text(txt)
        items = json.loads(gjson.read_text(encoding="utf-8"))
        max_end = max((it["position"][1] for it in items), default=0)
        if max_end > len(text):
            raise ValueError(
                f"{gjson.name}: nhãn có offset {max_end} > độ dài text {len(text)} "
                f"({txt.name}). Thường do line-ending CRLF↔LF lệch giữa notes và labels "
                f"(vd clone lại trên OS khác). Sinh lại bằng: "
                f"python scripts/parse_synthetic.py --augment-labs"
            )
        out.append((text, items, char_bio(text, items)))
    return out


def label_list() -> List[str]:
    labels = ["O"]
    for t in sorted(VALID_TYPES):
        labels += [f"B-{t}", f"I-{t}"]
    return labels
