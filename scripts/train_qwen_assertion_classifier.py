"""Train a three-label Qwen assertion occurrence classifier."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


LABELS = ("isHistorical", "isNegated", "isFamily")


def replace_quantized_score_head(model, num_labels: int, torch):
    """Keep the newly initialized sequence-classification head trainable outside 4-bit."""
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
    return model


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt(row: dict) -> str:
    item = row["candidate"]
    start = item["position"][0] - row["context_start"]
    end = item["position"][1] - row["context_start"]
    context = row["context"]
    marked = context[:start] + "<ENTITY>" + context[start:end] + "</ENTITY>" + context[end:]
    return (
        "Phan loai assertion cho dung occurrence y khoa. Moi nhan doc lap: "
        "isHistorical, isNegated, isFamily. Khong thay doi span hoac type.\n"
        f"Genre: {row['genre']}\nSegment: {row['segment_kind']}\n"
        f"Type: {item['type']}\nText: {item['text']}\nContext: {marked}"
    )


def jaccard(actual: set[int], predicted: set[int]) -> float:
    if not actual and not predicted:
        return 1.0
    return len(actual & predicted) / len(actual | predicted)


def threshold_report(rows: list[dict], probabilities: list[list[float]]) -> dict:
    candidates = [value / 20 for value in range(2, 19)]
    best = None
    for thresholds in itertools.product(candidates, repeat=3):
        by_source = {}
        for source in {row["source"] for row in rows}:
            indices = [index for index, row in enumerate(rows) if row["source"] == source]
            exact = total_jaccard = 0.0
            for index in indices:
                actual = {i for i, value in enumerate(rows[index]["labels"]) if value}
                predicted = {i for i, value in enumerate(probabilities[index]) if value >= thresholds[i]}
                exact += actual == predicted
                total_jaccard += jaccard(actual, predicted)
            by_source[source] = {
                "rows": len(indices),
                "exact": exact / len(indices),
                "jaccard": total_jaccard / len(indices),
            }
        primary = by_source.get("round2_pseudo", by_source.get("round1_gt"))
        if primary is None:
            total_rows = sum(item["rows"] for item in by_source.values())
            primary = {
                "rows": total_rows,
                "exact": sum(item["exact"] * item["rows"] for item in by_source.values()) / total_rows,
                "jaccard": sum(item["jaccard"] * item["rows"] for item in by_source.values()) / total_rows,
            }
            by_source["__all__"] = primary
        secondary = by_source.get("round1_gt", primary)
        score = (primary["jaccard"], primary["exact"], secondary["jaccard"])
        if best is None or score > best[0]:
            best = (score, thresholds, by_source)
    return {
        "thresholds": dict(zip(LABELS, best[1])),
        "validation_by_source": best[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/assertion_classifier_train.jsonl")
    parser.add_argument("--validation", default="data/assertion_classifier_validation.jsonl")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default="models/qwen25_3b_assertion_classifier")
    parser.add_argument("--report", default="output/qwen25_3b_assertion_classifier_validation.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--max-length", type=int, default=640)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    train_rows = load_rows(Path(args.train))
    validation_rows = load_rows(Path(args.validation))
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def materialize(rows):
        records = []
        for row in rows:
            encoded = tokenizer(prompt(row), truncation=True, max_length=args.max_length)
            encoded["labels"] = [float(value) for value in row["labels"]]
            encoded["sample_weight"] = float(row["loss_weight"])
            records.append(encoded)
        return Dataset.from_list(records)

    positives = np.asarray([row["labels"] for row in train_rows]).sum(axis=0)
    negatives = len(train_rows) - positives
    positive_weights = torch.tensor(np.minimum(negatives / np.maximum(positives, 1), 15.0), dtype=torch.float32)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=3, device_map="auto", quantization_config=quantization, torch_dtype=torch.bfloat16,
    )
    model = replace_quantized_score_head(model, 3, torch)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "multi_label_classification"
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {label: index for index, label in enumerate(LABELS)}
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"],
    ))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("sample_weight")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            losses = functional.binary_cross_entropy_with_logits(
                outputs.logits.float(), labels.float(), pos_weight=positive_weights.to(outputs.logits.device), reduction="none",
            ).mean(dim=1)
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            return (loss, outputs) if return_outputs else loss

    def metrics(result):
        logits, labels = result
        probabilities = 1 / (1 + np.exp(-logits))
        predicted = probabilities >= 0.5
        actual = labels >= 0.5
        exact = float(np.all(predicted == actual, axis=1).mean())
        scores = []
        for left, right in zip(actual, predicted):
            a = set(np.flatnonzero(left)); b = set(np.flatnonzero(right))
            scores.append(jaccard(a, b))
        return {"exact": exact, "jaccard": float(np.mean(scores))}

    output = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, bf16=True, logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="jaccard", greater_is_better=True,
        report_to="none", remove_unused_columns=False, seed=3407,
    )
    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=materialize(train_rows),
        eval_dataset=materialize(validation_rows), processing_class=tokenizer, compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    prediction = trainer.predict(materialize(validation_rows))
    probabilities = torch.sigmoid(torch.tensor(prediction.predictions).float()).tolist()
    report = threshold_report(validation_rows, probabilities)
    report.update({
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "positive_weights": dict(zip(LABELS, positive_weights.tolist())),
    })
    report_path = Path(args.report); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.joinpath(".complete").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
