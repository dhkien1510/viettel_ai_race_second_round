"""Raw-text-only inference for the Qwen BIO adapter.

This command deliberately has no anchor/reference argument.  Evaluation is an
optional, separate pass performed only after all predictions are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.offsets import read_text  # noqa: E402
from src.schema import VALID_TYPES  # noqa: E402
from scripts.train_qwen_token_ner import normalized_token_span, replace_quantized_score_head  # noqa: E402


def decode_document(text: str, tokenizer, model, device, max_len: int, stride: int) -> list[dict]:
    import torch

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_len,
        stride=stride,
        padding=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)
    with torch.inference_mode():
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits.float().cpu()

    votes: dict[tuple[int, int], list] = defaultdict(list)
    for window in range(logits.shape[0]):
        for token in range(logits.shape[1]):
            start, end = (int(value) for value in offsets[window, token])
            start, end = normalized_token_span(text, start, end)
            if start != end:
                votes[(start, end)].append(logits[window, token])

    labels = model.config.id2label
    tagged = []
    for (start, end), values in sorted(votes.items()):
        mean_logits = torch.stack(values).mean(0)
        tagged.append((start, end, labels[int(mean_logits.argmax())]))

    entities = []
    current = None
    for start, end, label in tagged:
        if label == "O" or "-" not in label:
            if current is not None:
                entities.append(current)
                current = None
            continue
        bio, entity_type = label.split("-", 1)
        if entity_type not in VALID_TYPES:
            continue
        can_continue = (
            current is not None
            and bio == "I"
            and current["type"] == entity_type
            and "\n" not in text[current["position"][1]:start]
        )
        if not can_continue:
            if current is not None:
                entities.append(current)
            current = {
                "text": text[start:end],
                "position": [start, end],
                "type": entity_type,
                "assertions": [],
                "candidates": [],
            }
        else:
            current["position"][1] = end
            current["text"] = text[current["position"][0]:end]
    if current is not None:
        entities.append(current)

    unique = {(item["position"][0], item["position"][1], item["type"]): item for item in entities}
    return [unique[key] for key in sorted(unique)]


def evaluate(prediction_dir: Path, reference_dir: Path, ids: list[str]) -> dict:
    true_positive = predicted = reference = 0
    by_type = defaultdict(Counter)
    for document_id in ids:
        pred_path = prediction_dir / f"{document_id}.json"
        ref_path = reference_dir / f"{document_id}.json"
        if not pred_path.exists() or not ref_path.exists():
            continue
        pred_items = json.loads(pred_path.read_text(encoding="utf-8"))
        ref_items = json.loads(ref_path.read_text(encoding="utf-8"))
        pred_set = {(tuple(item["position"]), item["type"]) for item in pred_items}
        ref_set = {(tuple(item["position"]), item["type"]) for item in ref_items}
        matches = pred_set & ref_set
        predicted += len(pred_set)
        reference += len(ref_set)
        true_positive += len(matches)
        for _, entity_type in pred_set:
            by_type[entity_type]["predicted"] += 1
        for _, entity_type in ref_set:
            by_type[entity_type]["reference"] += 1
        for _, entity_type in matches:
            by_type[entity_type]["true_positive"] += 1
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_span_type": {
            "true_positive": true_positive,
            "predicted": predicted,
            "reference": reference,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "by_type_counts": {key: dict(value) for key, value in by_type.items()},
    }


def ids_from_split(path: Path, fold: int | None, subset: str) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "folds" in data:
        if fold is None:
            raise ValueError("--fold is required for grouped split manifests")
        fold_data = data["folds"][str(fold)] if isinstance(data["folds"], dict) else data["folds"][fold]
        return [str(value) for value in fold_data[subset]]
    return [str(value) for value in data[subset]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="data/input-part2-real/input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids-file", help="JSON split file; infer its validation ids only")
    parser.add_argument("--ids-from-split", help="Grouped split manifest; use with --fold and --subset")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--subset", choices=["train", "validation"], default="validation")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--evaluate-after-inference", help="Reference dir, read only after prediction completes")
    parser.add_argument("--metrics-out")
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig

    checkpoint = Path(args.checkpoint).resolve()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((checkpoint / "backend_meta.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    peft_config = PeftConfig.from_pretrained(checkpoint)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForTokenClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=len(meta["labels"]),
        id2label={index: label for index, label in enumerate(meta["labels"])},
        label2id={label: index for index, label in enumerate(meta["labels"])},
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    base = replace_quantized_score_head(base, len(meta["labels"]), torch)
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()

    if args.ids_from_split:
        ids = ids_from_split(Path(args.ids_from_split), args.fold, args.subset)
    elif args.ids_file:
        split_data = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
        ids = [str(value) for value in split_data[args.split]]
    else:
        ids = [path.stem for path in sorted(input_dir.glob("*.txt"), key=lambda p: int(p.stem))]

    # Prediction completes before an evaluator is allowed to open a reference.
    for index, document_id in enumerate(ids, 1):
        text = read_text(input_dir / f"{document_id}.txt")
        entities = decode_document(text, tokenizer, model, "cuda", meta["max_len"], meta["stride"])
        (output_dir / f"{document_id}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{index}/{len(ids)}] {document_id}: {len(entities)} entities", flush=True)

    if args.evaluate_after_inference:
        metrics = evaluate(output_dir, Path(args.evaluate_after_inference).resolve(), ids)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        metrics_path = Path(args.metrics_out) if args.metrics_out else output_dir.with_name(output_dir.name + "_metrics.json")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
