from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG
from .normalize import parse_span

RRF_PATH = Path("data/rxnorm/rrf/RXNCONSO.RRF")
CACHE_PATH = Path("data/rxnorm/cache/rxnorm_index.pkl")

# RXNCONSO.RRF columns (pipe-delimited, no header):
# 0 RXCUI 1 LAT 2 TS 3 LUI 4 STT 5 SUI 6 ISPREF 7 RXAUI 8 SAUI 9 SCUI
# 10 SDUI 11 SAB 12 TTY 13 CODE 14 STR 15 SRL 16 SUPPRESS 17 CVF

KEEP_TTY = {
    "SCD", "SBD", "SCDC", "SBDC", "SCDF", "SBDF", "SCDG", "SBDG",
    "SCDGP", "SCDFP", "SBDFP", "IN", "PIN", "MIN", "BN", "PSN", "SY", "TMSY",
}

TTY_PRIORITY = {
    "SCD": 0, "PSN": 1, "SBD": 2, "SY": 3, "TMSY": 4,
    "SCDC": 5, "SBDC": 6, "BN": 7, "SCDF": 8, "SBDF": 9,
    "IN": 10, "PIN": 11, "MIN": 12,
    "SCDG": 13, "SBDG": 14, "SCDGP": 15, "SBDFP": 16, "SCDFP": 17,
}


@dataclass
class Entry:
    rxcui: str
    tty: str
    str_: str
    tokens: tuple[str, ...]
    strengths: tuple[tuple[float, str], ...]
    suppress: str = ""


@dataclass
class View:
    text: str
    source: str  # e.g. "STR", "BN", "SY", "NORMALIZED"


def _dedupe_views(views: list[View], max_count: int) -> list[View]:
    seen: set[str] = set()
    result: list[View] = []
    for v in views:
        key = v.text.lower()
        if key not in seen and len(v.text.strip()) >= CONFIG.min_view_length:
            seen.add(key)
            result.append(v)
        if len(result) >= max_count:
            break
    return result


def _build_views_for_rxcui(entries: list[Entry]) -> list[View]:
    by_source: dict[str, list[str]] = {"STR": [], "BN": [], "SY": [], "NORMALIZED": []}
    for e in entries:
        if e.tty == "BN":
            by_source["BN"].append(e.str_)
        elif e.tty == "SY" or e.tty == "TMSY":
            by_source["SY"].append(e.str_)
        elif e.tty in ("SCD", "SBD", "SCDC", "SBDC"):
            by_source["STR"].append(e.str_)
        else:
            by_source["NORMALIZED"].append(e.str_)

    all_views: list[View] = []
    for src, strs in by_source.items():
        for s in strs:
            if len(s.strip()) <= CONFIG.max_view_length:
                all_views.append(View(text=s.strip(), source=src))

    return _dedupe_views(all_views, CONFIG.max_views_per_rxcui)


def build(rrf_path: Path = RRF_PATH, cache_path: Path = CACHE_PATH) -> None:
    entries: list[Entry] = []
    rxcui_groups: dict[str, list[Entry]] = {}

    with rrf_path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            rxcui, sab, tty, str_ = fields[0], fields[11], fields[12], fields[14]
            if sab != "RXNORM" or tty not in KEEP_TTY:
                continue
            suppress = fields[16] if len(fields) > 16 else ""
            parsed = parse_span(str_)
            entry = Entry(rxcui, tty, str_, parsed.all_tokens, parsed.strengths, suppress)
            entries.append(entry)
            rxcui_groups.setdefault(rxcui, []).append(entry)

    # Build multi-view corpus per RXCUI
    rxcui_views: dict[str, list[View]] = {}
    for rxcui, group in rxcui_groups.items():
        rxcui_views[rxcui] = _build_views_for_rxcui(group)

    # Build token index from all entry tokens
    token_index: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        for tok in set(e.tokens):
            token_index.setdefault(tok, []).append(i)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                "version": CONFIG.cache_version,
                "entries": entries,
                "token_index": token_index,
                "tty_priority": TTY_PRIORITY,
                "rxcui_views": rxcui_views,
            },
            f,
        )
    print(f"Indexed {len(entries)} RXNORM entries with {len(rxcui_views)} RXCUI views -> {cache_path}")


if __name__ == "__main__":
    build()