# icd-linker-release

Gán mã ICD-10 (bản TT06 Việt Nam) cho một cụm chẩn đoán tiếng Việt, dựa trên
ngữ cảnh xung quanh nó trong bệnh án.

## Kiến trúc — 2 tầng

```
                    ┌─────────────────────────────────────────┐
 "béo phì" ────────▶│  Tầng 1: RETRIEVAL (tìm candidate)        │
 (chỉ text, chưa     │  - dict/alias exact match                │
  có ngữ cảnh)        │  - fuzzy string match (rapidfuzz)        │
                    │  - dense embedding (BGE-M3, cosine top-30)│
                    └─────────────────┬─────────────────────────┘
                                      │ pool ~10-40 mã ứng viên
                                      ▼
 "...Các bệnh mãn   ┌─────────────────────────────────────────┐
  tính\n- béo phì\n │  Tầng 2: RERANKING (chọn đúng mã)         │
  - THA..."  ───────▶│  XLM-RoBERTa cross-encoder, fine-tuned    │
 (context đầy đủ)    │  input: (context, tên_mã)                │
                    │  output: điểm phù hợp mỗi ứng viên       │
                    └─────────────────┬─────────────────────────┘
                                      ▼
                              "E66.9" (hoặc "NONE")
```

- **Tầng 1 (retrieval)**: dict/alias exact match + fuzzy string match +
  dense embedding (BGE-M3) — sinh ra một danh sách ~10-40 mã ICD ứng viên
  hợp lý cho cụm chẩn đoán, chưa xét ngữ cảnh.
- **Tầng 2 (reranking)**: model XLM-RoBERTa đọc **ngữ cảnh đầy đủ** (câu
  trước/sau trong bệnh án) cùng tên chính thức của từng mã ứng viên, chấm
  điểm mức độ phù hợp, chọn ra mã đúng nhất (hoặc quyết định không gán mã
  nào nếu không phù hợp). Model này cần được **train trên data gán nhãn của
  bạn** — repo này không đi kèm checkpoint sẵn, xem hướng dẫn train ở dưới.

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu GPU (khuyến nghị ≥8GB VRAM) để train/chạy nhanh — CPU vẫn chạy được
nhưng chậm hơn nhiều (BGE-M3 + XLM-R-base). Lần chạy đầu tiên sẽ tự tải model
`BAAI/bge-m3` (~2.2GB) từ HuggingFace Hub.

## Cấu trúc thư mục

```
icd-linker-release/
├── predict.py                    # API + CLI suy luận (dùng sau khi đã train)
├── build_training_data.py        # Bước 1: xây training data từ data gold của bạn
├── generate_synonyms.py          # Bước 2 (tùy chọn): mở rộng data bằng LLM
├── train.py                      # Bước 3: train reranker
├── evaluate.py                   # Bước 4: đánh giá pipeline đầy đủ
├── requirements.txt
├── src/icd/
│   ├── index.py                  # load ontology mã ICD-10 TT06 + alias dictionary
│   ├── matcher.py                # cascade dict/alias/qualifier/fuzzy (tầng 1, phần rule)
│   ├── fuzzy.py, normalize.py, qualifier.py, assertion.py, schema.py
│   └── embedding.py              # BGE-M3 dense retrieval (tầng 1, phần embedding)
└── data/processed/
    ├── icd_tt06_by_code.json     # danh mục mã ICD-10 TT06 (chính + đa cấp)
    ├── icd_tt06_aliases.yaml     # từ điển alias/từ đồng nghĩa
    ├── icd_tt06_vectors.npy      # embedding BGE-M3 sẵn có cho mọi mã
    └── icd_tt06_vector_meta.json
```

`data/processed/` (ontology + alias + vector) đã có sẵn, dùng chung cho mọi
lần train — bạn chỉ cần đem **data gán nhãn của bạn** (raw text + gold
label) vào để chạy các bước dưới.

## Định dạng data gán nhãn cần chuẩn bị

- Một thư mục text gốc: `<id>.txt` (nội dung bệnh án thô).
- Một thư mục nhãn gold: `<id>.json`, mỗi file là list entity:
  ```json
  [
    {"text": "tăng huyết áp", "position": [120, 133], "type": "CHAN_DOAN", "candidates": ["I10"]},
    {"text": "trầm cảm", "position": [200, 208], "type": "CHAN_DOAN", "candidates": []}
  ]
  ```
  `position` = [start, end] theo ký tự, PHẢI khớp chính xác với text gốc
  (`text[start:end] == entity["text"]`). `candidates` rỗng nghĩa là không
  nên gán mã ICD nào cho lần nhắc tới đó. Chỉ entity có `type` là chẩn đoán
  mới được dùng (script nhận diện qua hàm `is_diagnosis_type()` — sửa hàm
  này nếu tên type của bạn khác "CHAN_DOAN"/"CHẨN_ĐOÁN").

## Hướng dẫn chạy từ A-Z

### Bước 1 — Xây training data từ data gold của bạn

```bash
python build_training_data.py \
    --gold-dir path/to/your/gold_labels \
    --input-dir path/to/your/raw_texts \
    --out data/training_pairs.jsonl
```

Script in ra **recall của candidate pool trên mã gold** — đây là trần lý
thuyết của cả pipeline (reranker không thể chọn đúng mã không có trong
pool). Nếu số này thấp (dict/fuzzy/embedding không tìm ra mã đúng cho nhiều
case), làm Bước 2 trước khi train.

### Bước 2 (tùy chọn, khuyến nghị nếu recall Bước 1 thấp) — Mở rộng data bằng LLM

Chỉ dùng LLM ở bước sinh data này, không dùng để gán nhãn lúc chạy thật.

```bash
export OPEN_ROUTER_API=sk-or-...          # API key từ https://openrouter.ai

python generate_synonyms.py collect \
    --gold-dir path/to/your/gold_labels \
    --out-dir data/llm_generated

python generate_synonyms.py build-pairs \
    --llm-dir data/llm_generated \
    --out data/synthetic_pairs.jsonl
```

Lệnh `collect` gọi LLM sinh alias (tự động merge vào
`data/processed/icd_tt06_aliases.yaml`) + đoạn văn giả lập cho từng mã ICD
xuất hiện trong gold data của bạn. Lệnh `build-pairs` biến các đoạn giả lập
đó thành entity training bổ sung (cùng định dạng Bước 1, dùng chung được).

### Bước 3 — Train reranker

```bash
python train.py \
    --pairs data/training_pairs.jsonl data/synthetic_pairs.jsonl \
    --out models/reranker \
    --epochs 12
```

(Bỏ `data/synthetic_pairs.jsonl` nếu không làm Bước 2.) In ra `train_acc`/
`val_acc` mỗi epoch, tự lưu checkpoint tốt nhất theo `val_acc` vào
`models/reranker/best.pt`. Giảm `--meta-batch` nếu bị lỗi hết bộ nhớ GPU
(CUDA out of memory).

### Bước 4 — Đánh giá

```bash
python evaluate.py \
    --gold-dir path/to/your/gold_labels \
    --input-dir path/to/your/raw_texts \
    --model-dir models/reranker \
    --out report.json
```

Đo trên **toàn bộ** entity (train + val) theo đúng điều kiện suy luận thật
(candidate pool không được ép chứa đáp án đúng) — đây là con số nên báo
cáo, khác với `val_acc` lúc train (đo trên pool có trợ giúp, luôn cao hơn).
`report.json` liệt kê chi tiết từng case sai để soát tiếp.

### Sau khi train xong — dùng để dự đoán

```python
from predict import ICDLinker

linker = ICDLinker(model_dir="models/reranker")  # load 1 lần, dùng lại nhiều lần

code = linker.predict(context="...Các bệnh mãn tính\n- [[béo phì]]\n- tăng huyết áp...")
print(code)  # ví dụ: "E66.9"

code = linker.predict(
    text="tăng huyết áp",
    before="Tiền sử: đái tháo đường type 2, ",
    after=" độ II không kiểm soát.",
)
```

```bash
python predict.py --model-dir models/reranker \
    --context "...Các bệnh mãn tính\n- [[béo phì]]\n- tăng huyết áp..."
```

Trả về `"NONE"` nếu model quyết định không nên gán mã ICD nào cho lần nhắc
tới đó, tuỳ theo ngữ cảnh.
