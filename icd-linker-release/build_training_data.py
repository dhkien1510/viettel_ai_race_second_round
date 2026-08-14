"""Bước 1: Xây tập training data (context, candidate_code, label) cho
reranker, từ data gán nhãn của bạn.

Input cần có:
  --gold-dir   thư mục chứa <id>.json, mỗi file là list entity dạng:
               [{"text": "...", "position": [start, end], "type": "CHAN_DOAN",
                 "candidates": ["I10"]}, ...]   (candidates rỗng = không có mã)
  --input-dir  thư mục chứa <id>.txt (văn bản gốc, offset trong "position"
               phải trỏ đúng vào text này: text[start:end] == entity text)
  --data-dir   thư mục data/processed (ontology + alias + vector, đi kèm repo)

Với mỗi entity CHẨN_ĐOÁN, script sẽ:
  1. Lấy ngữ cảnh quanh span (mặc định ±220 ký tự, đánh dấu bằng [[ ]]).
  2. Sinh candidate pool = union(dict/fuzzy cascade top-K, dense BGE-M3 top-K).
  3. Ghi 1 dòng JSONL cho mỗi (entity, candidate) với label 1/0, cộng thêm
     1 dòng "NONE" (không gán mã) cho mỗi entity.

Chạy:
    python build_training_data.py \\
        --gold-dir path/to/your/gold_labels \\
        --input-dir path/to/your/raw_texts \\
        --data-dir data/processed \\
        --out data/training_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from icd.index import ICDIndex          # noqa: E402
from icd.matcher import ICDMatcher      # noqa: E402


def is_diagnosis_type(t: str) -> bool:
    """Nhận diện type CHẨN_ĐOÁN bất kể encoding dấu — đổi hàm này nếu bạn
    dùng type name khác."""
    return t.startswith("CH") and t.endswith("N") and "N_" in t and "O" in t


def extract_context(text: str, start: int, end: int, window: int) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return f"{text[lo:start]}[[{text[start:end]}]]{text[end:hi]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", required=True, type=Path)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--data-dir", default=ROOT / "data" / "processed", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--context-window", type=int, default=220, help="số ký tự lấy mỗi bên span")
    ap.add_argument("--cascade-topk", type=int, default=8)
    ap.add_argument("--dense-topk", type=int, default=30)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    index = ICDIndex(str(args.data_dir))
    matcher = ICDMatcher(index=index)
    cascade_cache: dict[str, dict] = {}

    def cached_match(span: str) -> dict:
        key = span.strip().lower()
        if key not in cascade_cache:
            cascade_cache[key] = matcher.match(span)
        return cascade_cache[key]

    print("Đang tải index vector + model BGE-M3...", flush=True)
    dense_vectors = np.load(args.data_dir / "icd_tt06_vectors.npy")
    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense_vectors = (dense_vectors / norms).astype(np.float32)
    dense_entries = json.loads((args.data_dir / "icd_tt06_vector_meta.json").read_text(encoding="utf-8"))["entries"]
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-m3", device=args.device)

    doc_ids = sorted(p.stem for p in args.gold_dir.glob("*.json"))
    all_entities = []
    text_cache: dict[str, str] = {}
    for doc_id in doc_ids:
        items = json.loads((args.gold_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        text_path = args.input_dir / f"{doc_id}.txt"
        if not text_path.exists():
            print(f"  cảnh báo: thiếu {text_path}, bỏ qua doc {doc_id}")
            continue
        text_cache[doc_id] = text_path.read_text(encoding="utf-8")
        for item in items:
            if is_diagnosis_type(item.get("type", "")):
                all_entities.append((doc_id, item))

    unique_texts = sorted({item["text"].strip() for _, item in all_entities})
    print(f"Encode {len(unique_texts)} cụm chẩn đoán duy nhất với BGE-M3...", flush=True)
    qvecs = embed_model.encode(unique_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    text_to_vec = dict(zip(unique_texts, qvecs))

    def dense_retrieve(span: str, top_k: int):
        scores = dense_vectors @ text_to_vec[span.strip()]
        idx = np.argsort(-scores)[:top_k]
        return [{"code": dense_entries[i]["code"], "name_vi": dense_entries[i]["name_vi"]} for i in idx]

    def candidate_name(code: str) -> str:
        node = index.get_code(code)
        return node.get("name_vi", "") if node else ""

    rows = []
    stats = {"entities": 0, "gold_present": 0, "gold_empty": 0, "gold_found_by_pool": 0, "gold_missing_from_pool": 0}

    for doc_id, item in all_entities:
        stats["entities"] += 1
        text = text_cache[doc_id]
        start, end = item["position"]
        assert text[start:end] == item["text"], f"offset không khớp ở doc {doc_id}: {item}"
        context = extract_context(text, start, end, args.context_window)
        gold_list = item.get("candidates") or []
        gold_code = gold_list[0] if gold_list else None

        cascade = cached_match(item["text"])
        dense = dense_retrieve(item["text"], args.dense_topk)

        pool: dict[str, str] = {}
        for c in (cascade.get("candidates") or [])[: args.cascade_topk]:
            code = c.get("code")
            if code and code not in pool:
                pool[code] = c.get("name_vi", "")
        for c in dense:
            if c["code"] not in pool:
                pool[c["code"]] = c["name_vi"]

        if gold_code:
            stats["gold_present"] += 1
            if gold_code in pool:
                stats["gold_found_by_pool"] += 1
            else:
                stats["gold_missing_from_pool"] += 1
                pool[gold_code] = candidate_name(gold_code)
        else:
            stats["gold_empty"] += 1

        for code, name in pool.items():
            rows.append({
                "doc_id": doc_id, "position": [start, end], "entity_text": item["text"],
                "context": context, "candidate_code": code, "candidate_name": name,
                "label": 1 if code == gold_code else 0, "gold_code": gold_code or "",
            })
        rows.append({
            "doc_id": doc_id, "position": [start, end], "entity_text": item["text"],
            "context": context, "candidate_code": "NONE", "candidate_name": "(không gán mã ICD)",
            "label": 1 if gold_code is None else 0, "gold_code": gold_code or "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    print(f"\nGhi {len(rows)} dòng training -> {args.out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["gold_present"]:
        cov = 100 * stats["gold_found_by_pool"] / stats["gold_present"]
        print(f"Recall của candidate pool trên mã gold: {cov:.2f}%")
        print("(Nếu số này thấp, cân nhắc bổ sung alias — xem generate_synonyms.py — trước khi train)")


if __name__ == "__main__":
    main()
