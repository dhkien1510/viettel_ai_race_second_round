Được, tôi đi qua 4 ví dụ cụ thể — mỗi ví dụ đại diện cho một nhánh khác nhau trong cascade, dùng thẳng data thật từ plan1 để bạn thấy input thật sẽ chạy ra sao ở từng bước.

---

**Ví dụ 1 — đi thẳng qua Alias dictionary (fast path, phổ biến nhất)**

Input span: `"tăng huyết áp"` (do module [3] Symptom/Diagnosis extractor xuất ra)

1. **Normalize**: lowercase, trim khoảng trắng thừa — không có modifier gì cần xử lý, giữ nguyên `"tăng huyết áp"`.
2. **Alias dictionary**: tra trong `diagnoses_vi_icd.yaml` (đã build từ 14 alias thủ công + auto-expand từ full DB TT06) → match chính xác → `I10`.
3. Có match → **dừng ngay**, không chạy fuzzy/embedding.

Output: `{"code": "I10", "confidence": 1.0, "method": "dictionary_exact"}`

---

**Ví dụ 2 — Normalize phải "hiểu" modifier, không chỉ xóa nó (điểm quan trọng tôi muốn nhấn thêm)**

Input span: `"bệnh thận mạn, không đặc hiệu"` (nguyên văn plan1 mục 9)

1. **Normalize**: đây là chỗ cần xử lý khéo hơn ví dụ 1. `"không đặc hiệu"` **không phải negation** (đã xác nhận ở turn trước — đây là modifier). Nhưng modifier này có ý nghĩa đặc biệt trong ICD: nó thường tương ứng đúng với mã con `.9 (unspecified)`. Nên bước normalize nên tách ra: `base_term = "bệnh thận mạn"` + `qualifier = "unspecified"`, thay vì chỉ xóa chữ "không đặc hiệu" đi và mất thông tin.
2. **Alias dictionary**: tra `base_term` → match ở mức category `N18.*` (đúng alias plan1 đã note: bệnh thận mạn → N18.*). Nhưng đây mới chỉ là category, chưa ra mã 4 ký tự cụ thể.
3. **Backoff có điều kiện qualifier**: vì phát hiện `qualifier = "unspecified"`, hệ thống tra tiếp trong cây con của `N18` (lấy từ ICD-10 index offline đã pull) tìm node con có tên chứa "không xác định"/"không đặc hiệu" → khớp `N18.9 "Bệnh thận mạn, không xác định"`.
4. Có match đủ tin cậy ở dictionary layer (dù phải qua bước backoff nội bộ) → dừng, không cần fuzzy/embedding.

Output: `{"code": "N18.9", "confidence": 0.9, "method": "dictionary+qualifier_backoff"}`

*(Nếu bước 3 không tìm được node con khớp, hệ thống nên lùi về trả `N18` ở mức category + candidate list các mã con, thay vì đoán bừa một mã 4 ký tự — đúng nguyên tắc "candidates 2-3 mã" của plan1.)*

---

**Ví dụ 3 — rơi xuống Fuzzy match (lỗi gõ/thiếu từ)**

Input span: `"đái đường típ 2"` (giả định note bị thiếu từ "tháo", kiểu lỗi gõ mà plan1 đã ghi nhận rất nhiều — `canx toàn phầni`, `Kết quả xétí nghiệm`...)

1. **Normalize**: `"đái đường típ 2"` — không có modifier, giữ nguyên.
2. **Alias dictionary**: tra exact-match → **miss**, vì key trong dictionary là `"đái tháo đường típ 2"` (đủ chữ), không khớp tuyệt đối.
3. **Fuzzy match**: so token-overlap giữa `"đái đường típ 2"` (4 token) với các tên trong index — với `"đái tháo đường típ 2"` (5 token), overlap 4/5 token → score ~0.8, vượt ngưỡng → match.
4. Có match ở tầng fuzzy → dừng, không cần embedding.

Output: `{"code": "E11", "confidence": 0.8, "method": "fuzzy_token_overlap"}`

*(Confidence thấp hơn ví dụ 1 vì đây là match gần đúng, không tuyệt đối — nên giữ score này để về sau còn biết ưu tiên tin cậy cái nào khi cần debug.)*

---

**Ví dụ 4 — không match ở đâu cả, phải xuống Embedding fallback**

Input span: `"cường cận giáp nguyên phát"` (không nằm trong 14 alias thủ công của plan1, và DB TT06 có thể ghi hơi khác chữ, ví dụ `"Cường tuyến cận giáp nguyên phát"`)

1. **Normalize**: không có modifier.
2. **Alias dictionary**: miss (chưa từng thêm vào dictionary).
3. **Fuzzy match**: miss hoặc score dưới ngưỡng (nếu chênh lệch chữ quá nhiều, token overlap thấp).
4. **Embedding fallback**: encode span bằng `multilingual-e5` (đã pre-embed sẵn toàn bộ ~2600 tên bệnh offline), tính cosine similarity, lấy top-k:
   - `E21.0` "Cường cận giáp nguyên phát" — sim 0.89
   - `E21.3` "Cường cận giáp, không xác định" — sim 0.71
   - `E21.9` "Rối loạn tuyến cận giáp, không xác định" — sim 0.65
5. Vì đây là tầng cuối, ít tin cậy nhất trong 3 tầng → **không trả 1 mã duy nhất**, trả cả top-3 làm candidate list.

Output: `{"candidates": ["E21.0", "E21.3", "E21.9"], "confidence": "low", "method": "embedding_topk"}`

---

**Tổng kết logic điều hướng:**

| Input đặc điểm | Tầng xử lý | Confidence output |
|---|---|---|
| Đúng tên chuẩn, không lỗi | Dictionary exact | Cao (~1.0) |
| Đúng tên gốc nhưng có modifier "không đặc hiệu/xác định" | Dictionary + qualifier backoff | Khá cao (~0.9), tự tìm đúng subcode |
| Có lỗi gõ nhẹ/thiếu từ | Fuzzy match | Trung bình (~0.7-0.8), 1 mã |
| Thuật ngữ lạ, diễn đạt khác hẳn DB | Embedding fallback | Thấp, trả candidate list thay vì 1 mã |

Điểm mấu chốt xuyên suốt cả 4 ví dụ: **mỗi span chỉ đi tới tầng cần thiết rồi dừng ngay khi có match đủ tin cậy** — không phải lúc nào cũng chạy hết cả 3 tầng. Việc này vừa tiết kiệm thời gian (embedding là tầng chậm nhất) vừa giảm rủi ro embedding "đoán nhầm" cho những case vốn đã có sẵn trong dictionary.


Có — nhưng chỉ cần ở **tầng 3 (embedding fallback)**. Tầng 1 (dictionary) không cần model gì cả, chỉ là tra bảng. Tầng 2 (fuzzy match) cũng không cần model, chỉ là thuật toán string similarity (Levenshtein, token overlap) — thuần thuật toán, không có gì để "học".

**Về model cho tầng 3, có một điểm quan trọng cần điều chỉnh so với đề xuất ban đầu của tôi:**

Ở turn trước tôi đề xuất `multilingual-e5` hoặc `LaBSE` — lý do lúc đó (theo tài liệu tổng hợp gốc) là để tránh phải dịch câu tiếng Việt sang tiếng Anh trước khi query ICD-10 tiếng Anh chuẩn. Nhưng giờ bạn đã tìm ra DB TT06 **có sẵn tên bệnh bằng tiếng Việt**, nên bài toán thực chất đã đổi bản chất: đây không còn là **cross-lingual retrieval** (Việt → Anh) nữa, mà là **monolingual retrieval** (Việt → Việt). Multilingual-e5/LaBSE được train để giỏi bắc cầu *giữa* nhiều ngôn ngữ — cái đó giờ thừa, và thường sẽ *kém hơn* một model tiếng Việt thuần khi so khớp ngữ nghĩa trong cùng một ngôn ngữ.

**Nên đổi sang:**
- `bkai-foundation-models/vietnamese-bi-encoder` — bi-encoder tiếng Việt, train cho retrieval, có sẵn trên HuggingFace, load thẳng qua thư viện `sentence-transformers`, không cần fine-tune thêm.
- Nếu muốn thử thêm domain y tế: `ViHealthBERT` — nhưng đây là encoder thô (không phải sentence-embedding model), cần thêm bước mean-pooling/fine-tune để ra vector câu tốt, tốn thời gian hơn không đáng trong bối cảnh deadline gấp.

Khuyến nghị thực tế: dùng `vietnamese-bi-encoder` trước, chỉ cân nhắc ViHealthBERT nếu sau khi có dev-set đo thấy độ chính xác chưa đủ.

**Về quy mô — không cần hạ tầng nặng:**
- Không cần GPU. Encode offline toàn bộ ~2600 tên bệnh (batch một lần) và encode on-demand vài chục diagnosis span mỗi file — CPU base-size model xử lý trong vài giây là đủ.
- Không cần FAISS/ANN index. Ở scale 2600 vector, brute-force cosine similarity bằng `numpy`/`sklearn` (`cosine_similarity`) nhanh hơn cả thời gian setup FAISS — dùng ANN ở đây là over-engineering.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
icd_names = [...]  # ~2600 tên bệnh từ index offline
icd_vectors = model.encode(icd_names, normalize_embeddings=True)  # chạy 1 lần, cache ra file .npy

# lúc runtime, cho mỗi span rơi tới tầng embedding:
span_vec = model.encode([span_text], normalize_embeddings=True)
sims = span_vec @ icd_vectors.T
top_k_idx = np.argsort(-sims[0])[:3]
```

**Về RxNorm (thuốc) — không cần embedding model chút nào.** Danh sách thuốc trong plan1 gần như đã cover hết case thật (metoprolol, doxycycline, atenolol...), và tên thuốc Latin có dạng chính tả ổn định hơn nhiều so với chẩn đoán tiếng Việt diễn đạt tự nhiên. Dictionary + fuzzy match (Levenshtein cho lỗi gõ như `atenololtrong`) gần như đủ — embedding cho RxNorm là công sức bỏ ra cho phần rủi ro thấp nhất trong bài toán này.

**Về mặt timeline — có thể trì hoãn tầng embedding này:**
Đúng theo lịch doc tổng hợp đã đặt (Ngày 9-10, sau khi đã SUBMIT LẦN 1 ở Ngày 8), bạn hoàn toàn có thể **bỏ qua tầng 3 ở bản nộp đầu tiên**, chỉ chạy dictionary + fuzzy — vì như 4 ví dụ turn trước cho thấy, phần lớn diagnosis phổ biến (14+ alias plan1 đã liệt kê, cộng auto-expand từ full DB TT06) đã được cover ở 2 tầng đầu. Model embedding chỉ nên thêm vào sau khi có điểm public/dev-set để biết thật sự có bao nhiêu % span rơi xuống tầng 3 — nếu con số đó nhỏ, việc đầu tư thêm model có thể không đáng effort so với việc dành thời gian tune assertion detector (module rủi ro cao hơn nhiều theo phân tích trước).