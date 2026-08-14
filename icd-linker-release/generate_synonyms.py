"""Bước 2 (tùy chọn, khuyến nghị): dùng LLM để mở rộng data train.

CHỈ dùng LLM ở bước sinh data này — không dùng LLM để gán nhãn lúc suy luận
thật. Script gọi LLM (qua OpenRouter) hỏi, với mỗi mã ICD xuất hiện trong
data gold của bạn:
  1. 6-10 từ đồng nghĩa/viết tắt/khẩu ngữ tiếng Việt hay dùng cho bệnh đó
     -> merge trực tiếp vào data/processed/icd_tt06_aliases.yaml (tăng
        recall của tầng retrieval).
  2. 4 đoạn văn bệnh án giả lập, mỗi đoạn nhắc tới bệnh đó 1 lần theo cách
     diễn đạt khác nhau, đánh dấu bằng [[ ]] -> dùng làm entity training bổ
     sung cho reranker (đa dạng cách diễn đạt hơn data thật vốn có).

Yêu cầu: biến môi trường OPEN_ROUTER_API (API key OpenRouter, https://openrouter.ai).

Chạy (2 bước):
    # 2a. Gọi LLM sinh alias + context giả lập cho từng mã trong gold-dir của bạn
    export OPEN_ROUTER_API=sk-or-...
    python generate_synonyms.py collect \\
        --gold-dir path/to/your/gold_labels \\
        --out-dir data/llm_generated

    # 2b. Merge kết quả vào alias dictionary + build thành training pairs
    #     (dùng CÙNG cascade/dense retrieval như build_training_data.py,
    #     để pool candidate cùng phân bố với data thật)
    python generate_synonyms.py build-pairs \\
        --llm-dir data/llm_generated \\
        --data-dir data/processed \\
        --out data/synthetic_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from icd.index import ICDIndex          # noqa: E402
from icd.matcher import ICDMatcher      # noqa: E402

MODEL = os.environ.get("LLM_DATAGEN_MODEL", "openai/gpt-4o-mini")

PROMPT_TEMPLATE = """Bạn là bác sĩ mã hoá ICD-10 (bản TT06 Việt Nam) giàu kinh nghiệm.

Mã: {code}
Tên chính thức: {name}

Nhiệm vụ:
1. Liệt kê 6-10 cách VIẾT KHÁC/từ đồng nghĩa/viết tắt/khẩu ngữ tiếng Việt mà bác sĩ
   hay dùng trong bệnh án để chỉ CHÍNH XÁC bệnh này (không phải bệnh liên quan/tương tự).
2. Viết 4 đoạn trích bệnh án NGẮN (60-120 từ), MỖI đoạn là một bối cảnh lâm sàng KHÁC NHAU,
   trong đó có đúng MỘT lần nhắc tới chẩn đoán này. Đánh dấu đúng cụm từ nhắc tới chẩn đoán
   bằng cặp ngoặc [[ ]], ví dụ: "...tiền sử [[tăng huyết áp]] và đái tháo đường...".
   Đa dạng cách diễn đạt cụm trong ngoặc (không lặp lại y hệt tên chính thức cả 4 lần).

Trả lời DUY NHẤT bằng JSON theo schema:
{{
  "aliases": ["...", "..."],
  "contexts": ["đoạn 1 có [[...]]", "đoạn 2 có [[...]]", "đoạn 3 có [[...]]", "đoạn 4 có [[...]]"]
}}
Không thêm giải thích ngoài JSON.
"""


def is_diagnosis_type(t: str) -> bool:
    return t.startswith("CH") and t.endswith("N") and "N_" in t and "O" in t


def call_llm(api_key: str, prompt: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_object"}, "temperature": 0.7},
                timeout=60,
            )
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {attempt+1}/{retries}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def cmd_collect(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPEN_ROUTER_API")
    if not api_key:
        raise SystemExit("Thiếu biến môi trường OPEN_ROUTER_API.")

    codes: set[str] = set()
    for path in Path(args.gold_dir).glob("*.json"):
        for item in json.loads(path.read_text(encoding="utf-8")):
            if is_diagnosis_type(item.get("type", "")):
                for code in item.get("candidates") or []:
                    codes.add(code)
    print(f"Tìm thấy {len(codes)} mã ICD duy nhất trong gold data.")

    index = ICDIndex(str(args.data_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alias_f = (out_dir / "aliases.jsonl").open("a", encoding="utf-8")
    ctx_f = (out_dir / "contexts.jsonl").open("a", encoding="utf-8")

    done = set()
    ctx_path = out_dir / "contexts.jsonl"
    if ctx_path.exists():
        for line in ctx_path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["code"])

    for i, code in enumerate(sorted(codes), 1):
        if code in done:
            print(f"[{i}/{len(codes)}] {code}: đã có, bỏ qua")
            continue
        node = index.get_code(code)
        name = node.get("name_vi", "") if node else ""
        if not name:
            print(f"[{i}/{len(codes)}] {code}: KHÔNG tìm thấy trong ontology, bỏ qua")
            continue
        result = call_llm(api_key, PROMPT_TEMPLATE.format(code=code, name=name))
        if result is None:
            print(f"[{i}/{len(codes)}] {code}: gọi LLM thất bại, bỏ qua")
            continue
        for alias in result.get("aliases") or []:
            alias_f.write(json.dumps({"alias": alias.strip().lower(), "code": code}, ensure_ascii=False) + "\n")
        alias_f.flush()
        n_ok = 0
        for raw in result.get("contexts") or []:
            if "[[" not in raw or "]]" not in raw:
                continue
            before, rest = raw.split("[[", 1)
            span, after = rest.split("]]", 1)
            if not span.strip():
                continue
            ctx_f.write(json.dumps({"code": code, "name": name, "before": before,
                                     "span": span.strip(), "after": after}, ensure_ascii=False) + "\n")
            n_ok += 1
        ctx_f.flush()
        print(f"[{i}/{len(codes)}] {code}: +{len(result.get('aliases') or [])} alias, +{n_ok} context")

    alias_f.close()
    ctx_f.close()


def cmd_build_pairs(args: argparse.Namespace) -> None:
    import numpy as np
    import yaml

    aliases_path = Path(args.data_dir) / "icd_tt06_aliases.yaml"
    data = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    added = 0
    for line in (Path(args.llm_dir) / "aliases.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        alias = row["alias"].strip()
        if alias and alias not in data:
            data[alias] = {"code": row["code"], "confidence": 0.9, "notes": "llm_synonym", "source": "llm_synonym"}
            added += 1
    aliases_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
    print(f"Alias: +{added} mới. Tổng: {len(data)}")

    index = ICDIndex(str(args.data_dir))
    matcher = ICDMatcher(index=index)  # nạp lại với alias vừa merge
    cache: dict[str, dict] = {}

    def cached_match(span: str) -> dict:
        key = span.strip().lower()
        if key not in cache:
            cache[key] = matcher.match(span)
        return cache[key]

    print("Đang tải index vector + BGE-M3...", flush=True)
    vectors = np.load(Path(args.data_dir) / "icd_tt06_vectors.npy")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype(np.float32)
    entries = json.loads((Path(args.data_dir) / "icd_tt06_vector_meta.json").read_text(encoding="utf-8"))["entries"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device=args.device)

    examples = [json.loads(l) for l in (Path(args.llm_dir) / "contexts.jsonl").read_text(encoding="utf-8").splitlines()]
    spans = sorted({e["span"] for e in examples})
    vecs = model.encode(spans, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    span_to_vec = dict(zip(spans, vecs))

    def dense_retrieve(span: str, top_k: int):
        scores = vectors @ span_to_vec[span]
        idx = np.argsort(-scores)[:top_k]
        return [{"code": entries[i]["code"], "name_vi": entries[i]["name_vi"]} for i in idx]

    rows = []
    for idx, ex in enumerate(examples):
        gold_code, span = ex["code"], ex["span"]
        context = f"{ex['before']}[[{span}]]{ex['after']}"
        cascade = cached_match(span)
        pool: dict[str, str] = {}
        for c in (cascade.get("candidates") or [])[: args.cascade_topk]:
            code = c.get("code")
            if code and code not in pool:
                pool[code] = c.get("name_vi", "")
        for c in dense_retrieve(span, args.dense_topk):
            if c["code"] not in pool:
                pool[c["code"]] = c["name_vi"]
        if gold_code not in pool:
            pool[gold_code] = ex["name"]

        doc_id = f"synth{idx}"
        for code, name in pool.items():
            rows.append({"doc_id": doc_id, "position": [0, len(span)], "entity_text": span,
                         "context": context, "candidate_code": code, "candidate_name": name,
                         "label": 1 if code == gold_code else 0, "gold_code": gold_code})
        rows.append({"doc_id": doc_id, "position": [0, len(span)], "entity_text": span,
                     "context": context, "candidate_code": "NONE", "candidate_name": "(không gán mã ICD)",
                     "label": 0, "gold_code": gold_code})

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"Ghi {len(rows)} dòng training tổng hợp ({len(examples)} entity) -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("collect", help="Gọi LLM sinh alias + context giả lập")
    p1.add_argument("--gold-dir", required=True, type=Path)
    p1.add_argument("--data-dir", default=ROOT / "data" / "processed", type=Path)
    p1.add_argument("--out-dir", required=True, type=Path)

    p2 = sub.add_parser("build-pairs", help="Merge alias + build training pairs từ context giả lập")
    p2.add_argument("--llm-dir", required=True, type=Path)
    p2.add_argument("--data-dir", default=ROOT / "data" / "processed", type=Path)
    p2.add_argument("--out", required=True, type=Path)
    p2.add_argument("--cascade-topk", type=int, default=8)
    p2.add_argument("--dense-topk", type=int, default=30)
    p2.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()
    if args.command == "collect":
        cmd_collect(args)
    else:
        cmd_build_pairs(args)


if __name__ == "__main__":
    main()
