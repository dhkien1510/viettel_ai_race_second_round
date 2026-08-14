"""Bước 4: Đánh giá pipeline đầy đủ (retrieval + reranker) trên tập gold
của bạn, theo đúng điều kiện suy luận thật (KHÔNG ép candidate đúng vào
pool) — số ra từ script này là số nên báo cáo, không phải train/val_acc lúc
train.py chạy (con số đó đo trên pool có trợ giúp).

Chạy:
    python evaluate.py \\
        --gold-dir path/to/your/gold_labels \\
        --input-dir path/to/your/raw_texts \\
        --data-dir data/processed \\
        --model-dir models/reranker \\
        --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from icd.index import ICDIndex          # noqa: E402
from icd.matcher import ICDMatcher      # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


def is_diagnosis_type(t: str) -> bool:
    return t.startswith("CH") and t.endswith("N") and "N_" in t and "O" in t


def extract_context(text: str, start: int, end: int, window: int) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return f"{text[lo:start]}[[{text[start:end]}]]{text[end:hi]}"


class Reranker(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.score_head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.score_head(cls).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", required=True, type=Path)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--data-dir", default=ROOT / "data" / "processed", type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--context-window", type=int, default=220)
    ap.add_argument("--cascade-topk", type=int, default=8)
    ap.add_argument("--dense-topk", type=int, default=30)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = args.device
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    val_docs_path = args.model_dir / "val_docs.json"
    val_docs = set(json.loads(val_docs_path.read_text(encoding="utf-8"))) if val_docs_path.exists() else set()

    index = ICDIndex(str(args.data_dir))
    matcher = ICDMatcher(index=index)
    cache: dict[str, dict] = {}

    def cached_match(span: str) -> dict:
        key = span.strip().lower()
        if key not in cache:
            cache[key] = matcher.match(span)
        return cache[key]

    print("Đang tải index vector + BGE-M3...", flush=True)
    dense_vectors = np.load(args.data_dir / "icd_tt06_vectors.npy")
    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense_vectors = (dense_vectors / norms).astype(np.float32)
    dense_entries = json.loads((args.data_dir / "icd_tt06_vector_meta.json").read_text(encoding="utf-8"))["entries"]
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-m3", device=device)

    print("Đang tải reranker đã train...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    reranker = Reranker(config["base_model"]).to(device)
    reranker.load_state_dict(torch.load(args.model_dir / "best.pt", map_location=device, weights_only=True))
    reranker.eval()

    entities = []
    for path in sorted(args.gold_dir.glob("*.json")):
        doc_id = path.stem
        text_path = args.input_dir / f"{doc_id}.txt"
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        for item in json.loads(path.read_text(encoding="utf-8")):
            if is_diagnosis_type(item.get("type", "")):
                entities.append((doc_id, item, text))

    unique_texts = sorted({e[1]["text"].strip() for e in entities})
    print(f"Encode {len(unique_texts)} cụm chẩn đoán duy nhất...", flush=True)
    qvecs = embed_model.encode(unique_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    text_to_vec = dict(zip(unique_texts, qvecs))

    def dense_retrieve(span: str, top_k: int):
        scores = dense_vectors @ text_to_vec[span.strip()]
        idx = np.argsort(-scores)[:top_k]
        return [{"code": dense_entries[i]["code"], "name_vi": dense_entries[i]["name_vi"]} for i in idx]

    def predict(context: str, span: str) -> str:
        cascade = cached_match(span)
        pool: dict[str, str] = {}
        for c in (cascade.get("candidates") or [])[: args.cascade_topk]:
            code = c.get("code")
            if code and code not in pool:
                pool[code] = c.get("name_vi", "")
        for c in dense_retrieve(span, args.dense_topk):
            if c["code"] not in pool:
                pool[c["code"]] = c["name_vi"]
        options = [("NONE", "(không gán mã ICD)")] + list(pool.items())[: args.cap - 1]
        contexts = [context] * len(options)
        cands = [f"{code}: {name}" for code, name in options]
        enc = tokenizer(contexts, cands, truncation=True, max_length=config["max_length"],
                         padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = reranker(**enc)
        return options[int(torch.argmax(logits).item())][0]

    results = {"all": {"correct": 0, "total": 0}, "val_only": {"correct": 0, "total": 0},
               "train_only": {"correct": 0, "total": 0}}
    mistakes = []

    for i, (doc_id, item, text) in enumerate(entities, 1):
        start, end = item["position"]
        context = extract_context(text, start, end, args.context_window)
        gold_list = item.get("candidates") or []
        gold_code = gold_list[0] if gold_list else "NONE"
        pred_code = predict(context, item["text"])
        is_correct = pred_code == gold_code

        bucket = "val_only" if doc_id in val_docs else "train_only"
        for b in ("all", bucket):
            results[b]["total"] += 1
            results[b]["correct"] += int(is_correct)
        if not is_correct:
            mistakes.append({"doc_id": doc_id, "text": item["text"], "gold": gold_code, "pred": pred_code})
        if i % 100 == 0:
            print(f"  ...{i}/{len(entities)}", flush=True)

    print("\n=== KẾT QUẢ (exact match: candidate dự đoán == candidate gold) ===")
    for name, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] else 0
        print(f"{name:12s}: {r['correct']}/{r['total']} = {acc*100:.2f}%")

    args.out.write_text(json.dumps({"results": results, "mistakes": mistakes}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\nBáo cáo đầy đủ ({len(mistakes)} case sai) -> {args.out}")


if __name__ == "__main__":
    main()
