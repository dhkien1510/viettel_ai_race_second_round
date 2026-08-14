"""Train Qwen as a span-preserving BIO token classifier with QLoRA.

The reference directory is used only to construct training labels.  Inference
is implemented separately in ``infer_qwen_token_ner.py`` and never receives a
reference/anchor directory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.ner_dataset import char_bio, label_list  # noqa: E402
from src.offsets import read_text  # noqa: E402
from src.pipeline.genre_router import classify_text  # noqa: E402

BOUNDARY_DELIMITERS = ".:;/|-"


def normalized_token_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Remove tokenizer-owned separators without changing entity-internal text."""
    while start < end and (text[start].isspace() or text[start] in BOUNDARY_DELIMITERS):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in BOUNDARY_DELIMITERS):
        end -= 1
    return start, end


def replace_quantized_score_head(model, num_labels: int, torch, o_label_id: int | None = None):
    """Keep the newly initialized classification head trainable outside 4-bit."""
    head = getattr(model, "score", None)
    if head is None or head.__class__.__module__.startswith("torch.nn"):
        return model
    in_features = getattr(head, "in_features", None) or getattr(model.config, "hidden_size", None)
    if in_features is None:
        raise ValueError("Cannot infer score head input size")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.score = torch.nn.Linear(in_features, num_labels, bias=True).to(
        device=device,
        dtype=torch.bfloat16,
    )
    if o_label_id is not None and model.score.bias is not None:
        with torch.no_grad():
            model.score.bias.fill_(-2.0)
            model.score.bias[o_label_id] = 2.0
    return model


def load_documents(input_dir: Path, labels_dir: Path) -> list[dict]:
    documents = []
    for label_path in sorted(labels_dir.glob("*.json"), key=lambda p: int(p.stem)):
        text_path = input_dir / f"{label_path.stem}.txt"
        if not text_path.exists():
            continue
        text = read_text(text_path)
        entities = json.loads(label_path.read_text(encoding="utf-8"))
        documents.append({
            "id": label_path.stem,
            "text": text,
            "entities": entities,
            "tags": char_bio(text, entities),
            "genre": classify_text(text),
        })
    return documents


def stratified_split(documents: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if val_ratio <= 0:
        return documents, []
    groups: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        groups[document["genre"]].append(document)
    rng = random.Random(seed)
    train, validation = [], []
    for group in groups.values():
        rng.shuffle(group)
        count = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation


def split_from_manifest(
    documents: list[dict],
    manifest_path: Path | None,
    fold: int | None,
    train_ids_path: Path | None,
    validation_ids_path: Path | None,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    by_id = {document["id"]: document for document in documents}
    if train_ids_path or validation_ids_path:
        train_ids = json.loads(train_ids_path.read_text(encoding="utf-8")) if train_ids_path else []
        validation_ids = json.loads(validation_ids_path.read_text(encoding="utf-8")) if validation_ids_path else []
        train_docs = [by_id[str(value)] for value in train_ids if str(value) in by_id]
        val_docs = [by_id[str(value)] for value in validation_ids if str(value) in by_id]
        return train_docs, val_docs, {
            "source": "explicit-id-files",
            "train_ids": [doc["id"] for doc in train_docs],
            "validation_ids": [doc["id"] for doc in val_docs],
        }
    if manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if fold is None:
            raise ValueError("--fold is required when --split-manifest is provided")
        fold_data = manifest["folds"][str(fold)] if isinstance(manifest["folds"], dict) else manifest["folds"][fold]
        train_docs = [by_id[str(value)] for value in fold_data["train"] if str(value) in by_id]
        val_docs = [by_id[str(value)] for value in fold_data["validation"] if str(value) in by_id]
        return train_docs, val_docs, {
            "source": "split-manifest",
            "manifest": str(manifest_path),
            "fold": fold,
            "train_ids": [doc["id"] for doc in train_docs],
            "validation_ids": [doc["id"] for doc in val_docs],
        }
    train_docs, val_docs = stratified_split(documents, val_ratio, seed)
    return train_docs, val_docs, {
        "source": "stratified-random",
        "val_ratio": val_ratio,
        "train_ids": [doc["id"] for doc in train_docs],
        "validation_ids": [doc["id"] for doc in val_docs],
    }


def audit_boundaries(documents: list[dict], tokenizer) -> tuple[int, int]:
    total = representable = 0
    for document in documents:
        offsets = tokenizer(document["text"], add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        normalized = [normalized_token_span(document["text"], start, end) for start, end in offsets]
        starts = {start for start, end in normalized if start != end}
        ends = {end for start, end in normalized if start != end}
        for entity in document["entities"]:
            total += 1
            start, end = entity["position"]
            representable += start in starts and end in ends
    return representable, total


def encode_documents(documents: list[dict], tokenizer, labels: list[str], max_len: int, stride: int):
    from datasets import Dataset

    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for document in documents:
        encoded = tokenizer(
            document["text"],
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_len,
            stride=stride,
            padding="max_length",
        )
        for input_ids, attention_mask, offsets in zip(
            encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
        ):
            token_labels = []
            for start, end in offsets:
                start, end = normalized_token_span(document["text"], start, end)
                if start == end:
                    token_labels.append(-100)
                    continue
                tag = document["tags"][start]
                token_labels.append(label_to_id.get(tag, label_to_id["O"]))
            rows["input_ids"].append(input_ids)
            rows["attention_mask"].append(attention_mask)
            rows["labels"].append(token_labels)
    return Dataset.from_dict(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input", default="data/input-part2-real/input")
    parser.add_argument("--train", default="output/superbest_dataset")
    parser.add_argument("--out", default="models/superbest_qwen/ner")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--train-ids", default="")
    parser.add_argument("--validation-ids", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    input_dir = (ROOT / args.input).resolve()
    train_dir = (ROOT / args.train).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    documents = load_documents(input_dir, train_dir)
    train_docs, val_docs, split_info = split_from_manifest(
        documents,
        Path(args.split_manifest) if args.split_manifest else None,
        args.fold,
        Path(args.train_ids) if args.train_ids else None,
        Path(args.validation_ids) if args.validation_ids else None,
        args.val_ratio,
        args.seed,
    )
    split = {
        "seed": args.seed,
        **split_info,
        "train": [doc["id"] for doc in train_docs],
        "validation": [doc["id"] for doc in val_docs],
        "genres": {doc["id"]: doc["genre"] for doc in documents},
    }
    (out_dir / "document_split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    representable, total = audit_boundaries(documents, tokenizer)
    print(f"Documents: {len(train_docs)} train / {len(val_docs)} validation")
    print(f"Tokenizer exact-boundary ceiling: {representable}/{total} = {representable / total:.6f}")
    labels = label_list()
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_dataset = encode_documents(train_docs, tokenizer, labels, args.max_len, args.stride)
    val_dataset = encode_documents(val_docs, tokenizer, labels, args.max_len, args.stride) if val_docs else None
    print(f"Windows: {len(train_dataset)} train / {len(val_dataset) if val_dataset else 0} validation")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id=label_to_id,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model = replace_quantized_score_head(model, len(labels), torch, label_to_id.get("O"))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["score"],
        ),
    )
    model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    strategy = "epoch" if val_dataset is not None else "no"
    if val_dataset is not None:
        training_kwargs.update(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    try:
        training_args = TrainingArguments(eval_strategy=strategy, **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy=strategy, **training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)
    (out_dir / "backend_meta.json").write_text(
        json.dumps(
            {
                "backend": "qwen_token_ner",
                "base_model": args.model,
                "max_len": args.max_len,
                "stride": args.stride,
                "labels": labels,
                "labels_source_used_only_during_training": str(train_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved adapter: {out_dir}")


if __name__ == "__main__":
    main()
