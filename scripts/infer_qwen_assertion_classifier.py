"""Apply a trained multi-label assertion classifier to predicted entities."""

from __future__ import annotations

import argparse
import json
import sys
sys.modules["tensorflow"] = None
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_qwen_assertion_classifier import LABELS, prompt, replace_quantized_score_head  # noqa: E402
from src.pipeline.context_router import ContextRouter  # noqa: E402
from src.schema import ASSERTABLE_TYPES  # noqa: E402


def is_reserved_reference_source(path: Path) -> bool:
    parts = [part.lower() for part in path.resolve().parts]
    reserved_markers = {"reference", "gold_reference", "reserved_reference"}
    return any(part in reserved_markers for part in parts)


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def inference_row(document_id: str, text: str, routed, item: dict) -> dict:
    segment = next(
        (value for value in routed.segments if value.start <= item["position"][0] < value.end),
        routed.segments[0],
    )
    start = max(segment.start, item["position"][0] - 220)
    end = min(segment.end, item["position"][1] + 220)
    return {
        "file_id": document_id,
        "genre": routed.genre,
        "segment_kind": segment.kind,
        "context": text[start:end],
        "context_start": start,
        "candidate": {
            "text": item["text"],
            "position": item["position"],
            "type": item["type"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input-part2-real/input")
    parser.add_argument(
        "--entities",
        required=True,
        help="Thu muc JSON span/type duoc sinh boi Qwen token-NER hoac pipeline hop le.",
    )
    parser.add_argument("--anchor", dest="entities", help=argparse.SUPPRESS)
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--validation-report",
        default="output/qwen25_3b_assertion_classifier_validation.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Thu muc output moi sau khi gan assertions cho entities dau vao.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument(
        "--entity-audit",
        default="",
        help=(
            "JSON audit chi tiet theo tung entity, gom probabilities va segment metadata. "
            "Dung cho precision bundle va offline review."
        ),
    )
    parser.add_argument(
        "--allow-prohibited-source",
        action="store_true",
        help=(
            "Chi dung cho audit offline voi thu muc tham chieu duoc chi dinh ro. "
            "Khong bat co nay khi tao submission."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entities_dir = Path(args.entities)
    if is_reserved_reference_source(entities_dir) and not args.allow_prohibited_source:
        raise SystemExit(
            "Refusing to run inference from a reserved reference folder. "
            "Use Qwen-generated entity predictions instead, or pass --allow-prohibited-source only for offline audit."
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    thresholds = load(Path(args.validation_report))["thresholds"]
    threshold_values = [thresholds[label] for label in LABELS]
    rows = []
    states = []
    documents: dict[str, list[dict]] = {}
    routed_rows: dict[str, list[dict]] = {}
    router = ContextRouter()
    for file_id in range(1, 101):
        document_id = str(file_id)
        text = Path(args.input, f"{document_id}.txt").read_text(encoding="utf-8")
        routed = router.route(text)
        items = load(entities_dir / f"{document_id}.json")
        documents[document_id] = items
        for index, item in enumerate(items):
            if item["type"] not in ASSERTABLE_TYPES:
                item["assertions"] = []
                continue
            row = inference_row(document_id, text, routed, item)
            rows.append(row)
            routed_rows.setdefault(document_id, []).append(row)
            states.append((document_id, index))

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if torch.cuda.is_available():
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base,
            num_labels=3,
            device_map="auto",
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
        )
        model = replace_quantized_score_head(model, 3, torch)
    else:
        print("WARNING: CUDA is not available. Loading model on CPU (this might be slow).")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base,
            num_labels=3,
            device_map=None,
            torch_dtype=torch.bfloat16,
        )
        model = replace_quantized_score_head(model, 3, torch)
    model.config.pad_token_id = tokenizer.pad_token_id
    if not hasattr(model, "prepare_inputs_for_generation"):
        model.prepare_inputs_for_generation = lambda *args, **kwargs: {}
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    probabilities = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        encoded = tokenizer(
            [prompt(row) for row in batch],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
        probabilities.extend(torch.sigmoid(logits).tolist())
        print(f"[{min(start + args.batch_size, len(rows))}/{len(rows)}]")

    assertion_counts = Counter()
    entity_audit: dict[str, list[dict]] = {}
    for (document_id, index), values in zip(states, probabilities):
        assertions = [
            label
            for label, probability, threshold in zip(LABELS, values, threshold_values)
            if probability >= threshold
        ]
        documents[document_id][index]["assertions"] = assertions
        assertion_counts[tuple(assertions)] += 1
        item = documents[document_id][index]
        entity_audit.setdefault(document_id, []).append({
            "index": index,
            "text": item["text"],
            "position": item["position"],
            "type": item["type"],
            "assertions": assertions,
            "probabilities": {
                label: probability for label, probability in zip(LABELS, values)
            },
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for document_id, items in documents.items():
        (output / f"{document_id}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for document_id in map(str, range(1, 101)):
            handle.write(output / f"{document_id}.json", f"{document_id}.json")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entities": sum(len(items) for items in documents.values()),
        "classified_entities": len(rows),
        "thresholds": thresholds,
        "assertion_counts": {
            "|".join(key) or "EMPTY": value for key, value in assertion_counts.items()
        },
        "archive": str(archive),
    }
    output.with_name(output.name + "_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.entity_audit:
        entity_audit_path = Path(args.entity_audit)
        entity_audit_path.parent.mkdir(parents=True, exist_ok=True)
        entity_audit_path.write_text(
            json.dumps({
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "entities_dir": str(entities_dir.resolve()),
                "input_dir": str(Path(args.input).resolve()),
                "thresholds": thresholds,
                "documents": entity_audit,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
