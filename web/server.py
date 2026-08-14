from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pickle
import sys
import unicodedata
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_ROOT = Path(__file__).resolve().parent


def find_rxnorm_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("RXNORM_LINKER_ROOT"):
        candidates.append(Path(os.environ["RXNORM_LINKER_ROOT"]))
    candidates.extend((WEB_ROOT.parent / "rxnorm-linker", WEB_ROOT.parent))
    for parent in WEB_ROOT.parents:
        candidates.extend((parent / "rxnorm-linker", parent))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "rxnorm").is_dir():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy rxnorm-linker. Đặt web cạnh rxnorm-linker, khai báo "
        "RXNORM_LINKER_ROOT hoặc truyền --rxnorm-root."
    )


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return " ".join(value.split())


class DrugDatabase:
    def __init__(self, cache_path: Path):
        with cache_path.open("rb") as stream:
            data = pickle.load(stream)

        self.entries = data["entries"]
        self.token_index = data["token_index"]
        self.tty_priority = data["tty_priority"]
        self.source_alias_index = data.get("source_alias_index", {})
        self.by_rxcui: dict[str, list[int]] = defaultdict(list)
        self.exact_names: dict[str, set[int]] = defaultdict(set)

        for index, entry in enumerate(self.entries):
            self.by_rxcui[entry.rxcui].append(index)
            self.exact_names[normalize_text(entry.str_)].add(index)

    def _concept(self, rxcui: str, match: str, matched_name: str = "") -> dict:
        indexes = self.by_rxcui[rxcui]
        entries = sorted(
            (self.entries[index] for index in indexes),
            key=lambda entry: (
                entry.suppress not in ("", "N"),
                self.tty_priority.get(entry.tty, 99),
                len(entry.str_),
            ),
        )
        preferred = entries[0]
        names = []
        seen = set()
        for entry in entries:
            key = entry.str_.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append({"name": entry.str_, "tty": entry.tty})
            if len(names) == 8:
                break
        return {
            "rxcui": rxcui,
            "name": preferred.str_,
            "tty": preferred.tty,
            "suppressed": preferred.suppress not in ("", "N"),
            "match": match,
            "matched_name": matched_name,
            "names": names,
        }

    def search(self, raw_query: str, limit: int = 20) -> list[dict]:
        query = raw_query.strip()
        if not query:
            return []
        limit = max(1, min(limit, 50))

        if query.isdigit() and query in self.by_rxcui:
            return [self._concept(query, "RXCUI chính xác")]

        normalized = normalize_text(query)
        scored: dict[str, tuple[float, str, str]] = {}

        def add(index: int, score: float, match: str) -> None:
            entry = self.entries[index]
            current = scored.get(entry.rxcui)
            value = (score, match, entry.str_)
            if current is None or value[0] > current[0]:
                scored[entry.rxcui] = value

        for index in self.exact_names.get(normalized, ()):
            add(index, 100.0, "Tên chính xác")

        parsed = parse_span(query)
        token_sets = [set(self.token_index.get(token, ())) for token in parsed.ingredient_tokens]
        token_sets = [items for items in token_sets if items]
        candidate_indexes: set[int] = set()
        if token_sets:
            candidate_indexes = set.intersection(*token_sets)
            if not candidate_indexes:
                candidate_indexes = set.union(*token_sets)

        for index in candidate_indexes:
            entry = self.entries[index]
            name = normalize_text(entry.str_)
            if name == normalized:
                score, match = 100.0, "Tên chính xác"
            elif name.startswith(normalized):
                score, match = 88.0, "Khớp tiền tố"
            elif normalized in name:
                score, match = 78.0, "Tên có chứa từ khóa"
            else:
                query_tokens = set(parsed.ingredient_tokens)
                entry_tokens = set(entry.tokens)
                overlap = len(query_tokens & entry_tokens) / max(1, len(query_tokens))
                score, match = 50.0 + overlap * 25.0, "Khớp thành phần"
            if entry.suppress not in ("", "N"):
                score -= 15.0
            score -= self.tty_priority.get(entry.tty, 20) * 0.01
            add(index, score, match)

        alias = self.source_alias_index.get(source_alias_key(query))
        if alias:
            rxcui, sources = alias
            if rxcui in self.by_rxcui:
                index = self.by_rxcui[rxcui][0]
                add(index, 92.0, "Alias: " + ", ".join(sources))

        ranked = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
        return [
            self._concept(rxcui, metadata[1], metadata[2])
            for rxcui, metadata in ranked
        ]


class AppHandler(BaseHTTPRequestHandler):
    database: DrugDatabase

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        if request.path == "/api/search":
            query = parse_qs(request.query).get("q", [""])[0]
            try:
                limit = int(parse_qs(request.query).get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            results = self.database.search(query, limit)
            self.send_json({"query": query, "count": len(results), "results": results})
            return

        route = "/index.html" if request.path == "/" else request.path
        allowed = {
            "/index.html": WEB_ROOT / "index.html",
            "/styles.css": WEB_ROOT / "styles.css",
            "/app.js": WEB_ROOT / "app.js",
        }
        path = allowed.get(route)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RxNorm drug search web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rxnorm-root", type=Path, help="đường dẫn tới folder rxnorm-linker")
    parser.add_argument("--cache", type=Path, help="đường dẫn cache; mặc định lấy từ rxnorm-linker")
    args = parser.parse_args()

    try:
        rxnorm_root = find_rxnorm_root(args.rxnorm_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    sys.path.insert(0, str(rxnorm_root / "src"))

    # Imported after root discovery so the web folder can live anywhere.
    global source_alias_key, parse_span
    from rxnorm.build_index import source_alias_key
    from rxnorm.normalize import parse_span

    cache_path = (args.cache.expanduser().resolve() if args.cache else
                  rxnorm_root / "data" / "rxnorm" / "cache" / "rxnorm_index.pkl")
    if not cache_path.is_file():
        parser.error(f"RxNorm cache not found: {cache_path}")
    print(f"RxNorm root: {rxnorm_root}")
    print(f"Loading RxNorm cache: {cache_path}")
    AppHandler.database = DrugDatabase(cache_path)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"RxNorm search is ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
