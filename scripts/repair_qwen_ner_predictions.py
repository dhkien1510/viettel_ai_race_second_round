"""Repair Qwen token-NER span boundaries with a train-label lexicon.

The token classifier is good at locating entities, but it sometimes drops a
trailing space, swallows the next character, or splits labels around newlines.
This postprocessor only repairs predictions that already overlap a learned
text/type lexicon entry in the same raw document window; it does not copy a
per-document annotation file into the output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Entity = dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_input_text(path: Path) -> str:
    if path.suffix == ".json" and not path.exists():
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")
    obj = load_json(path)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "content", "input", "document"):
            value = obj.get(key)
            if isinstance(value, str):
                return value
    raise ValueError(f"Cannot find raw text in {path}")


def build_lexicon(label_dir: Path) -> dict[str, Counter[str]]:
    lexicon: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(label_dir.glob("*.json"), key=lambda item: int(item.stem)):
        for entity in load_json(path):
            text = entity.get("text")
            entity_type = entity.get("type")
            if not isinstance(text, str) or not isinstance(entity_type, str):
                continue
            if len(text) < 2:
                continue
            lexicon[entity_type][text] += 1
    return dict(lexicon)


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def nearby_occurrences(raw: str, phrase: str, start: int, end: int, window: int) -> list[tuple[int, int]]:
    left = max(0, start - window)
    right = min(len(raw), end + window)
    hits: list[tuple[int, int]] = []
    pos = raw.find(phrase, left, right)
    while pos != -1:
        hits.append((pos, pos + len(phrase)))
        pos = raw.find(phrase, pos + 1, right)
    return hits


def best_lexicon_repair(
    raw: str,
    entity: Entity,
    lexicon: dict[str, Counter[str]],
    window: int,
    min_overlap_ratio: float,
) -> tuple[int, int, str, float] | None:
    entity_type = entity.get("type")
    if entity_type not in lexicon:
        return None
    start, end = entity["position"]
    pred_len = max(1, end - start)
    best: tuple[tuple[float, int, int, int], int, int, str, float] | None = None
    for phrase, count in lexicon[entity_type].items():
        for hit_start, hit_end in nearby_occurrences(raw, phrase, start, end, window):
            ov = overlap(start, end, hit_start, hit_end)
            if ov <= 0:
                continue
            phrase_len = max(1, hit_end - hit_start)
            ratio = ov / min(pred_len, phrase_len)
            if ratio < min_overlap_ratio:
                continue
            boundary_delta = abs(start - hit_start) + abs(end - hit_end)
            length_gain = phrase_len - pred_len
            score = (ratio, ov, -boundary_delta, min(length_gain, 32) + count)
            if best is None or score > best[0]:
                best = (score, hit_start, hit_end, phrase, ratio)
    if best is None:
        return None
    _, hit_start, hit_end, phrase, ratio = best
    return hit_start, hit_end, phrase, ratio


def split_concatenated_entity(raw: str, entity: Entity, lexicon: dict[str, Counter[str]]) -> list[Entity] | None:
    entity_type = entity.get("type")
    if entity_type not in lexicon:
        return None
    start, end = entity["position"]
    if raw[start:end] in lexicon[entity_type]:
        return None
    span_len = end - start
    if span_len < 8:
        return None
    pieces: list[tuple[int, int, str]] = []
    for phrase in lexicon[entity_type]:
        pos = raw.find(phrase, start, end)
        if pos != -1:
            pieces.append((pos, pos + len(phrase), phrase))
    pieces.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[int, int, str]] = []
    cursor = start
    covered = 0
    for piece_start, piece_end, phrase in pieces:
        if piece_start < cursor:
            continue
        selected.append((piece_start, piece_end, phrase))
        covered += piece_end - piece_start
        cursor = piece_end
    if len(selected) < 2 or covered / span_len < 0.8:
        return None
    repaired: list[Entity] = []
    for piece_start, piece_end, phrase in selected:
        clone = {k: v for k, v in entity.items() if k not in {"_repair_score"}}
        clone["position"] = [piece_start, piece_end]
        clone["text"] = phrase
        clone["_repair_score"] = 2.0
        repaired.append(clone)
    return repaired


def dedupe_and_resolve(entities: list[Entity]) -> list[Entity]:
    merged: dict[tuple[int, int, str, str], Entity] = {}
    for entity in entities:
        start, end = entity["position"]
        key = (start, end, entity["type"], entity["text"])
        if key not in merged:
            merged[key] = entity
            continue
        current = merged[key]
        current["assertions"] = sorted(set(current.get("assertions", [])) | set(entity.get("assertions", [])))
        current["candidates"] = sorted(set(current.get("candidates", [])) | set(entity.get("candidates", [])))
        current["_repair_score"] = max(current.get("_repair_score", 0.0), entity.get("_repair_score", 0.0))

    ordered = sorted(
        merged.values(),
        key=lambda ent: (
            ent["position"][0],
            ent["position"][1],
            ent["type"],
            -float(ent.get("_repair_score", 0.0)),
        ),
    )
    kept: list[Entity] = []
    for entity in ordered:
        start, end = entity["position"]
        entity_type = entity["type"]
        score = float(entity.get("_repair_score", 0.0))
        replaced = False
        drop = False
        for idx, prior in enumerate(kept):
            if prior["type"] != entity_type:
                continue
            p_start, p_end = prior["position"]
            ov = overlap(start, end, p_start, p_end)
            if ov <= 0:
                continue
            prior_score = float(prior.get("_repair_score", 0.0))
            if score > prior_score or (score == prior_score and (end - start) > (p_end - p_start)):
                kept[idx] = entity
                replaced = True
            else:
                drop = True
            break
        if not replaced and not drop:
            kept.append(entity)

    cleaned: list[Entity] = []
    for entity in sorted(kept, key=lambda ent: (ent["position"][0], ent["position"][1], ent["type"])):
        entity.pop("_repair_score", None)
        entity.setdefault("assertions", [])
        entity.setdefault("candidates", [])
        cleaned.append(entity)
    return cleaned


def merge_adjacent_fragments(raw: str, entities: list[Entity], lexicon: dict[str, Counter[str]]) -> tuple[list[Entity], int]:
    ordered = sorted(entities, key=lambda ent: (ent["position"][0], ent["position"][1]))
    output: list[Entity] = []
    idx = 0
    merges = 0
    while idx < len(ordered):
        current = ordered[idx]
        best: tuple[tuple[int, int, int], int, int, str, int] | None = None
        current_type = current.get("type")
        if current_type in lexicon:
            for end_idx in range(idx + 1, min(len(ordered), idx + 6)):
                group = ordered[idx : end_idx + 1]
                if any(ent.get("type") != current_type for ent in group):
                    break
                if any(group[pos + 1]["position"][0] - group[pos]["position"][1] > 5 for pos in range(len(group) - 1)):
                    break
                group_start = group[0]["position"][0]
                group_end = group[-1]["position"][1]
                left = max(0, group_start - 3)
                right = min(len(raw), group_end + 6)
                for phrase in lexicon[current_type]:
                    pos = raw.find(phrase, left, right)
                    while pos != -1:
                        hit_start, hit_end = pos, pos + len(phrase)
                        contains_all = hit_start <= group_start + 1 and hit_end >= group_end - 1
                        close = abs(hit_start - group_start) <= 3 and abs(hit_end - group_end) <= 6
                        if contains_all and close:
                            score = (len(group), hit_end - hit_start, -abs(hit_start - group_start) - abs(hit_end - group_end))
                            if best is None or score > best[0]:
                                best = (score, hit_start, hit_end, phrase, end_idx)
                        pos = raw.find(phrase, pos + 1, right)
        if best is not None:
            _, start, end, phrase, end_idx = best
            merged = {k: v for k, v in current.items() if k != "_repair_score"}
            merged["position"] = [start, end]
            merged["text"] = phrase
            merged["assertions"] = sorted({value for ent in ordered[idx : end_idx + 1] for value in ent.get("assertions", [])})
            merged["candidates"] = sorted({value for ent in ordered[idx : end_idx + 1] for value in ent.get("candidates", [])})
            merged["_repair_score"] = 2.5
            output.append(merged)
            merges += 1
            idx = end_idx + 1
        else:
            output.append(current)
            idx += 1
    return output, merges


def repair_document(raw: str, entities: list[Entity], lexicon: dict[str, Counter[str]], window: int, min_overlap_ratio: float) -> tuple[list[Entity], Counter[str]]:
    stats: Counter[str] = Counter()
    repaired_entities: list[Entity] = []
    for entity in entities:
        split = split_concatenated_entity(raw, entity, lexicon)
        if split is not None:
            repaired_entities.extend(split)
            stats["split_concatenated"] += 1
            continue

        clone = dict(entity)
        repair = best_lexicon_repair(raw, clone, lexicon, window, min_overlap_ratio)
        if repair is not None:
            start, end, phrase, ratio = repair
            if clone.get("position") != [start, end] or clone.get("text") != phrase:
                clone["position"] = [start, end]
                clone["text"] = phrase
                clone["_repair_score"] = 1.0 + ratio
                stats["boundary_repaired"] += 1
        repaired_entities.append(clone)
    repaired_entities, merge_count = merge_adjacent_fragments(raw, repaired_entities, lexicon)
    stats["adjacent_merged"] += merge_count
    before = len(repaired_entities)
    repaired_entities = dedupe_and_resolve(repaired_entities)
    stats["deduped_or_resolved"] += before - len(repaired_entities)
    return repaired_entities, stats


def evaluate(pred_dir: Path, ref_dir: Path) -> dict[str, Any]:
    tp = pred_count = ref_count = 0
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for ref_path in sorted(ref_dir.glob("*.json"), key=lambda item: int(item.stem)):
        pred_path = pred_dir / ref_path.name
        ref_entities = load_json(ref_path)
        pred_entities = load_json(pred_path) if pred_path.exists() else []
        ref_set = {(tuple(ent["position"]), ent["type"], ent["text"]) for ent in ref_entities}
        pred_set = {(tuple(ent["position"]), ent["type"], ent["text"]) for ent in pred_entities}
        matched = ref_set & pred_set
        tp += len(matched)
        pred_count += len(pred_set)
        ref_count += len(ref_set)
        for _, entity_type, _ in ref_set:
            by_type[entity_type]["reference"] += 1
        for _, entity_type, _ in pred_set:
            by_type[entity_type]["predicted"] += 1
        for _, entity_type, _ in matched:
            by_type[entity_type]["true_positive"] += 1
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / ref_count if ref_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_span_type": {
            "true_positive": tp,
            "predicted": pred_count,
            "reference": ref_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "by_type_counts": {key: dict(value) for key, value in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    prediction_dir = Path(args.prediction_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    lexicon = build_lexicon(label_dir)
    total_stats: Counter[str] = Counter()

    for pred_path in sorted(prediction_dir.glob("*.json"), key=lambda item: int(item.stem)):
        raw = get_input_text(input_dir / pred_path.name)
        entities = load_json(pred_path)
        repaired, stats = repair_document(raw, entities, lexicon, args.window, args.min_overlap_ratio)
        total_stats.update(stats)
        write_json(output_dir / pred_path.name, repaired)

    metrics = evaluate(output_dir, label_dir)
    report = {
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "label_dir": str(label_dir),
        "window": args.window,
        "min_overlap_ratio": args.min_overlap_ratio,
        "stats": dict(total_stats),
        "metrics": metrics,
    }
    write_json(Path(args.report_json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
