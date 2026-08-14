# Đề bài chính thức — Vòng 1 Sơ loại (02/07/2026 – 30/07/2026)

Bản lưu nguyên văn quy định BTC (nguồn: trang nộp bài). Diễn giải & định hướng
annotate xem [annotation_guideline.md](annotation_guideline.md) phần B.

---

Bài toán yêu cầu xây dựng hệ thống AI xử lý văn bản y khoa tự do — ghi chú bác sĩ,
giấy xuất viện, kết quả xét nghiệm, hồ sơ EHR — để phát hiện và chuẩn hóa các khái
niệm y tế xuất hiện trong văn bản. Hệ thống cần xác định loại khái niệm (triệu
chứng, kết quả xét nghiệm, bệnh, thuốc, thông tin bệnh nhân), ánh xạ bệnh với chuẩn
ICD-10 và thuốc với ch`uẩn RxNorm, đồng thời suy luận mối liên hệ ngữ cảnh (phủ định,
người nhà, tiền sử) cũng như quan hệ giữa các khái niệm. Đây là bài toán nền tảng
cho chuyển đổi số y tế, giúp dữ liệu lâm sàng phi cấu trúc có thể liên thông và khai
thác trên quy mô lớn cho chẩn đoán, nghiên cứu dịch tễ và các ứng dụng AI y khoa.

## 1. Tổng quan
Bài toán tập trung vào việc sử dụng những giải pháp NLP, LLM hay kết hợp agents xây
dựng một hệ thống AI có khả năng thực hiện đồng thời: xác định và chuẩn hóa khái
niệm y tế chuyên môn và suy luận ontology (Ontological Reasoning) trên dữ liệu y
khoa dạng văn bản tự do (free-form clinical text) nhằm xác định quan hệ giữa các
khái niệm y tế trong một ngữ cảnh nhất định. Hệ thống AI được cung cấp các cơ sở
tri thức y khoa là ICD và RxNorm. Nhiệm vụ của hệ thống là: phát hiện các khái niệm
y tế và thông tin bệnh nhân xuất hiện trong văn bản, xác định loại khái niệm (bao
gồm triệu chứng, kết quả xét nghiệm, bệnh và thuốc điều trị), thực hiện ánh xạ các
khái niệm này với nguồn dữ liệu tương ứng và trả về danh sách các mã định danh phù
hợp nhất cho từng khái niệm, và xác định các mối liên hệ giữa các khái niệm này
trong đoạn văn. Bài toán cần xử lý hai nhóm giải pháp chính: xác định và chuẩn hóa
khái niệm y tế, và suy luận mối liên hệ giữa các khái niệm đã được xác định.

## 2. Bối cảnh
Trong lĩnh vực y tế, dữ liệu lâm sàng và hồ sơ bệnh án thường được ghi nhận dưới
nhiều định dạng và cách diễn đạt khác nhau, phụ thuộc vào cơ sở khám chữa bệnh,
chuyên khoa, ngôn ngữ chuyên môn cũng như thói quen nhập liệu của nhân viên y tế.
Để đảm bảo khả năng liên thông, thống nhất và khai thác dữ liệu trên quy mô lớn,
nhiều hệ thống chuẩn y khoa đã được xây dựng như ICD, SNOMED CT, RxNorm, LOINC,
UMLS,… cùng với danh mục dùng chung chứa thông tin bệnh nhân (patient database).
Các chuẩn này đóng vai trò như một "ngôn ngữ chung" giúp đồng bộ dữ liệu giữa các
bệnh viện, hệ thống bảo hiểm, nền tảng nghiên cứu và các ứng dụng trí tuệ nhân tạo
trong y tế. Tuy nhiên, trong thực tế vận hành, phần lớn dữ liệu y khoa vẫn tồn tại
dưới dạng văn bản tự do như ghi chú bác sĩ, mô tả triệu chứng, kết luận chẩn đoán
hay báo cáo cận lâm sàng, nơi cùng một khái niệm có thể được diễn đạt theo nhiều
cách khác nhau, sử dụng từ viết tắt, thuật ngữ địa phương hoặc chứa lỗi chính tả và
cấu trúc không chuẩn hóa.

Hiện nay, quá trình chuẩn hóa các khái niệm y tế từ văn bản tự do vẫn là một thách
thức lớn đối với các hệ thống xử lý dữ liệu y khoa. Việc ánh xạ chính xác giữa biểu
đạt ngôn ngữ tự nhiên và khái niệm chuẩn đòi hỏi mô hình phải hiểu được ngữ cảnh
chuyên môn sâu, xử lý hiện tượng đa nghĩa, đồng nghĩa và các biến thể diễn đạt phức
tạp trong tiếng nói lâm sàng. Đặc biệt, trong môi trường dữ liệu thực tế, văn bản
thường ngắn gọn, thiếu cấu trúc, chứa nhiều ký hiệu chuyên ngành hoặc kết hợp đồng
thời nhiều thông tin bệnh lý trong cùng một câu. Những khó khăn này làm hạn chế khả
năng khai thác dữ liệu phục vụ hỗ trợ chẩn đoán, nghiên cứu dịch tễ, thống kê y tế
và xây dựng các hệ thống AI y khoa quy mô lớn. Những hệ thống này nếu không thể kết
nối được với chuẩn y tế đã tồn tại thì không thể hiệu quả. Vì vậy, bài toán đang trở
thành một hướng nghiên cứu và ứng dụng quan trọng, đóng vai trò nền tảng cho quá
trình chuyển đổi số và phát triển trí tuệ nhân tạo trong lĩnh vực chăm sóc sức khỏe.

## 3. Mô tả bài toán

### 3.1 Input
Input của bài toán là một đoạn văn bản y khoa dạng tự do (free-form text). Input có
thể tồn tại ở các dạng: kết quả khám lâm sàng, giấy xuất viện, ghi chú của bác sĩ,
kết quả chẩn đoán hình ảnh, kết quả xét nghiệm, hồ sơ sức khỏe điện tử (EHR), hoặc
các ghi chú lâm sàng khác. Dữ liệu đầu vào có thể chứa: thuật ngữ y khoa, viết tắt,
thông tin bệnh nhân và nhiều loại khái niệm y tế khác nhau xuất hiện đồng thời trong
cùng một văn bản.

VD: "Bệnh nhân bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được
chẩn đoán mắc bệnh trào ngược dạ dày - thực quản."

### 3.2 Output
Output là danh sách các khái niệm y tế được phát hiện, cùng nội dung khái niệm, loại
khái niệm, danh sách candidate mapping và mối liên hệ giữa các khái niệm. Mỗi khái
niệm gồm các trường:

- **text**: cụm từ trong input mà hệ thống xác định là một khái niệm y tế.
- **position**: list 2 phần tử [start, end] — vị trí bắt đầu/kết thúc của cụm trong
  input (mặc định từ 0 đến n − 1, n = độ dài input theo ký tự).
- **type**: 1 trong 5 nhãn:
  - `TRIỆU_CHỨNG`: Tên triệu chứng bệnh nhân mắc phải
  - `TÊN_XÉT_NGHIỆM`: Tên xét nghiệm bệnh nhân thực hiện
  - `KẾT_QUẢ_XÉT_NGHIỆM`: Kết quả xét nghiệm, bao gồm giá trị và đơn vị
  - `CHẨN_ĐOÁN`: Tên chẩn đoán của bác sĩ về bệnh mà bệnh nhân mắc phải
  - `THUỐC`: Tên thuốc mà bệnh nhân điều trị
- **assertions**: mối liên hệ của khái niệm (chỉ giới hạn CHẨN_ĐOÁN, THUỐC,
  TRIỆU_CHỨNG), list tối đa 3 phần tử:
  - `isNegated`: khái niệm bị phủ định (VD: "không ho")
  - `isFamily`: liên quan đến người nhà/họ hàng (VD: "bố bệnh nhân xuất hiện đau
    bụng tương tự")
  - `isHistorical`: liên quan tiền sử bệnh nhân (VD: "có tiền sử hen suyễn")
- **candidates**: list candidate mapping dự đoán, chỉ xét trên CHẨN_ĐOÁN và THUỐC.
  Mỗi phần tử là mã chuẩn tương ứng (ICD với bệnh, RxNorm với thuốc).

### 3.3 Ví dụ đầy đủ (chính thức)
Input:
```
"Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ
hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử sử
dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân tích
tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung
tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;"
```
Output:
- `CHẨN_ĐOÁN`: "bệnh trào ngược dạ dày - thực quản" — mã ICD `K21.0`, `K21.9`
- `TRIỆU_CHỨNG`: "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi"
- `TÊN_XÉT_NGHIỆM`: "WBC", "NEUT% (Tỷ lệ % bạch cầu trung tính)",
  "LYPH% (Tỷ lệ bạch cầu lympho)"  *(bản đề ghi "TWBC" — coi là lỗi gõ của "WBC")*
- `KẾT_QUẢ_XÉT_NGHIỆM`: "14,43", "76,4", "12,8"
- `THUỐC`: "Chlorpheniramine 0.4 MG/ML" — RxNorm `360047`;
  "Capsaicin 0.38 MG/ML" — RxNorm `1660761`; assertion: `isHistorical`

> Lưu ý: thông tin cá nhân (tên, tuổi, địa chỉ, sđt) đều là giá trị **synthetic**,
> không phải người thật.

## 4. Dữ liệu bài toán
CSDL chuẩn cho candidate mapping: **ICD-10** cho bệnh, **RxNorm** cho thuốc.

**Tập test**: 100 bản ghi, cung cấp dưới dạng `test.zip`, giải nén ra:
```
test/
└── input/
    ├── 1.txt      # Văn bản đầu vào của bản ghi 1
    ├── 2.txt
    ├── …
    └── 100.txt
```
Các file `.txt` là văn bản free-form làm input; **mỗi văn bản chứa nhiều hơn 1 khái
niệm**. Với mỗi `.txt`, thí sinh trả về 1 file `.json` tương ứng (list các dictionary
khái niệm y tế).

> Các thí sinh **cần sử dụng các giải pháp nằm ngoài lời giải chính để tạo thêm dữ
> liệu** nhằm huấn luyện mô hình. (→ được dùng công cụ ngoài để TẠO DATA; "lời giải
> chính" = hệ inference nộp thì self-host ≤9B, không API ngoài.)

---

# 5. Thể thức nộp
Nộp file `output.zip`, giải nén ra:
```
output/
  ├── 1.json      # Nhãn của bản ghi 1
  ├── 2.json
  ├── …
  └── 100.json
```
Trước khi vòng 1 kết thúc, BTC yêu cầu **top ~15 đội** gửi source code để dựng lại và
đánh giá trên **private test** (chống hard-code output theo input đã cho). Source
code gồm: tất cả file code (data processing, training, inference), **data** đã dùng,
**model weights**, **README** hướng dẫn cài đặt. BTC cài không được → liên lạc hỗ
trợ trong thời gian nhất định; không hỗ trợ kịp → **loại**.

### Ví dụ output có `position` (danh sách thuốc)
Input: `"...1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily ... guaifenesin
ml po q6h:prn điều trị ho ..."` → mỗi thuốc là 1 THUỐC có `candidates` (RxNorm) +
`["isHistorical"]` + `position`; lý do dùng ("ho","đau nhức","sốt đau","táo bón",
"lo âu","mất ngủ") tách thành TRIỆU_CHỨNG riêng.
```json
{"text":"amlodipine 10 mg po daily","type":"THUỐC","candidates":["308135"],
 "assertions":["isHistorical"],"position":[58,83]}
{"text":"guaifenesin ml po q6h:prn","type":"THUỐC","candidates":["392085"],
 "assertions":["isHistorical"],"position":[161,186]}
{"text":"ho","type":"TRIỆU_CHỨNG","assertions":[],"position":[196,198]}
```

# 6. Metric đánh giá
```
final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score
```
- **text**: Word Error Rate (WER) trên trường `text`.
  `text_score = Σ_i (1 − WER(i)) / len(test)`
- **assertions**: Jaccard trên tập assertion (bệnh/thuốc/triệu chứng), trung bình →
  `J_assertions(i)`. `assertions_score = Σ_i J_assertions(i) / len(test)`
- **candidates**: Jaccard tập mã, **trọng số theo số candidate**.
  `candidates_score = Σ_i J_candidates(i)·w_i / Σ_i w_i`, `w_i = Σ_k (len(gt(k))+1)`.
- Quy ước Jaccard `J_X(i)`: gold rỗng & pred rỗng → **1**; gold rỗng & pred khác
  rỗng → **0**; còn lại → `|gold ∩ pred| / |gold ∪ pred|`.

> **Lưu ý chấm điểm**: Đoán ĐÚNG text nhưng SAI loại (VD đoán CHẨN_ĐOÁN nhưng ground
> truth là TRIỆU_CHỨNG) → khái niệm bị **tính 2 lần** và **mỗi lần 0 điểm với cả 3
> loại metric**.

# 7. Timeline
- 02/07/2026: công bố tập test public (100 file).
- 30/07/2026: deadline nộp. Tối đa 5 lần submit/ngày. Top 8 public → Vòng 2.
