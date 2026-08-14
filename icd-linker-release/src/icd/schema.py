"""Schema normalization and index construction for ICD-10 TT06 data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .normalize import normalize_aggressive, normalize_conservative


_CATEGORY_RE = re.compile(r"^[A-Z][0-9]{2}$")
_DISEASE_RE = re.compile(r"^[A-Z][0-9]{2}\.[0-9A-Z]$")
_SECTION_RE = re.compile(r"^[A-Z][0-9]{2}-[A-Z][0-9]{2}$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")


def extract_raw_fields(raw: dict[str, Any]) -> dict[str, Any]:
    wrapper = raw.get("raw") if isinstance(raw.get("raw"), dict) else raw
    data = wrapper.get("data") if isinstance(wrapper.get("data"), dict) else {}
    code = raw.get("code") or wrapper.get("code") or data.get("code") or wrapper.get("id") or data.get("id")
    name = raw.get("name") or wrapper.get("name") or data.get("name") or ""
    node_id = raw.get("id") or wrapper.get("id") or data.get("id") or code
    return {"id": str(node_id), "code": str(code).upper(), "name": str(name)}


def canonical_code(code: str) -> str:
    code = str(code).upper().strip()
    code = re.sub(r"[†*‡]+$", "", code).strip()
    if _DISEASE_RE.match(code):
        return code
    if re.match(r"^[A-Z][0-9]{3}$", code):
        return f"{code[:3]}.{code[3:]}"
    return code


def infer_level(code: str, model: str | None = None) -> str:
    if model in {"chapter", "section", "category", "disease"}:
        return model
    if _ROMAN_RE.match(code):
        return "chapter"
    if _SECTION_RE.match(code):
        return "section"
    if _CATEGORY_RE.match(code):
        return "category"
    if _DISEASE_RE.match(code):
        return "disease"
    return "unknown"


def parent_code_for(code: str, by_code: dict[str, dict[str, Any]]) -> str | None:
    if _DISEASE_RE.match(code):
        parent = code.split(".", 1)[0]
        return parent if parent in by_code else None
    return None


def normalize_raw_nodes(raw_nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    interim: dict[str, dict[str, Any]] = {}
    raw_by_code: dict[str, dict[str, Any]] = {}
    for raw in raw_nodes:
        fields = extract_raw_fields(raw)
        code = canonical_code(fields["code"])
        if not code or code in interim:
            continue
        node = {
            "id": fields["id"],
            "code": code,
            "name_vi": fields["name"],
            "level": infer_level(code, raw.get("model")),
            "parent_id": raw.get("parent_id"),
            "parent_code": raw.get("parent_code"),
            "children_ids": [],
            "children_codes": [],
            "path_ids": raw.get("path_ids", []),
            "path_codes": raw.get("path_codes", []),
            "path_names": raw.get("path_names", []),
            "normalized_name": normalize_aggressive(fields["name"]),
            "normalized_name_with_accents": normalize_conservative(fields["name"]),
            "raw": raw.get("raw", raw),
        }
        interim[code] = node
        raw_by_code[code] = raw

    for code, node in interim.items():
        parent = node.get("parent_code") or parent_code_for(code, interim)
        if parent:
            node["parent_code"] = parent
            node["parent_id"] = interim.get(parent, {}).get("id")
            parent_node = interim.get(parent)
            if parent_node:
                parent_node["children_codes"].append(code)
                parent_node["children_ids"].append(node["id"])

    return sorted(interim.values(), key=lambda item: item["code"])


def build_indexes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_code = {node["code"]: node for node in nodes if node.get("code")}
    by_name: dict[str, list[str]] = {}
    children = {
        node["code"]: node.get("children_codes", [])
        for node in nodes
        if node.get("children_codes")
    }
    aliases = build_aliases(nodes)

    for node in nodes:
        code = node.get("code")
        for key in {node.get("normalized_name"), node.get("normalized_name_with_accents")}:
            if key and code:
                by_name.setdefault(key, [])
                if code not in by_name[key]:
                    by_name[key].append(code)

    return {"by_code": by_code, "by_name": by_name, "children": children, "aliases": aliases}


def build_aliases(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    for node in nodes:
        code = node.get("code")
        name = node.get("name_vi")
        if not code or not name:
            continue
        for alias in {normalize_conservative(name), normalize_aggressive(name)}:
            if not alias or alias in aliases:
                continue
            aliases[alias] = {
                "code": code,
                "source": "auto_name",
                "confidence": 1.0,
                "notes": "",
            }
    return aliases


def load_raw_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_processed(nodes: list[dict[str, Any]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    indexes = build_indexes(nodes)
    (out / "icd_tt06_nodes.jsonl").write_text(
        "\n".join(json.dumps(node, ensure_ascii=False) for node in nodes) + "\n",
        encoding="utf-8",
    )
    (out / "icd_tt06_by_code.json").write_text(
        json.dumps(indexes["by_code"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "icd_tt06_by_name.json").write_text(
        json.dumps(indexes["by_name"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "icd_tt06_children.json").write_text(
        json.dumps(indexes["children"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "icd_tt06_aliases.yaml").write_text(
        yaml.safe_dump(indexes["aliases"], allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
