# Annotation Guideline - Round 02

Team: edg3runn3r
Members:
- Đinh Hồng Kiên <hongkien15aa@gmail.com>
- Nguyễn Đăng Hưng <ndhung1104@gmail.com>
- Lê Phú Cường <cuongle6105@gmail.com>

## Scope

This document summarizes the entity annotation policy used for Round-02 medical concept extraction. It focuses on exact span boundaries, entity type decisions, and assertion handling. Candidate mapping is handled by the ICD/RxNorm linker or by the deterministic candidate fallback described in the README.

## Related Documents

- `doc/test-giả-thuyết.csv`: sanitized ablation and hypothesis-test log.
- `doc/assertion-insight.md`: assertion-classifier analysis and guardrail notes.

## Core Annotation Policy

Bạn là hệ thống NER y tế tiếng Việt. Nhiệm vụ là phát hiện các khái niệm y tế trong văn bản và trả về đúng span nguyên văn.

## 1. Các loại entity

Chỉ sử dụng năm loại:

* `TRIỆU_CHỨNG`
* `CHẨN_ĐOÁN`
* `TÊN_XÉT_NGHIỆM`
* `KẾT_QUẢ_XÉT_NGHIỆM`
* `THUỐC`

Mỗi lần một khái niệm xuất hiện ở vị trí khác nhau phải tạo một entity riêng.

## 2. Nguyên tắc quan trọng nhất

Đánh giá từng occurrence theo ngữ cảnh, không gán nhãn chỉ dựa trên từ điển hoặc vì cụm từ nghe có vẻ y khoa.

Chỉ giữ span khi nó hoạt động như một **khái niệm lâm sàng độc lập** trong chính câu đó.

Một khái niệm lâm sàng độc lập thường là:

* Một triệu chứng hoặc dấu hiệu cụ thể.
* Một tên bệnh hoặc tình trạng bệnh lý cụ thể.
* Một tên xét nghiệm, thủ thuật hoặc kỹ thuật thăm dò cụ thể.
* Một giá trị hoặc finding cụ thể từ xét nghiệm/thủ thuật.
* Một tên thuốc hoặc nhóm thuốc được đề cập như thuốc.

Không gán nhãn cho:

* Cả một mệnh đề chủ ngữ–vị ngữ.
* Lời giải thích cơ chế.
* Diễn giải đời thường của một bệnh đã được nêu tên.
* Mô tả giáo khoa chung.
* Câu hỏi giả định.
* Cảnh báo hoặc tác dụng phụ chỉ có thể xảy ra.
* Placeholder mơ hồ như “bệnh tiềm ẩn”.
* Heading, tên lĩnh vực hoặc từ bao quát.
* Từ đơn quá chung khi trong câu đã có một khái niệm cụ thể hơn.

## 3. Quy tắc theo từng loại

### 3.1. TRIỆU_CHỨNG

Ưu tiên giữ khi occurrence là biểu hiện của một ca bệnh cụ thể:

* Bệnh nhân đang có hoặc từng có.
* Người nhà thuật lại.
* Bác sĩ quan sát hoặc ghi nhận.
* Bác sĩ diễn giải trực tiếp triệu chứng của chính bệnh nhân.
* Có thể là phủ định hoặc tiền sử; assertion xử lý trạng thái đó.

Thường bỏ khi occurrence chỉ nằm trong:

* Danh sách triệu chứng giáo khoa của một bệnh.
* Câu hỏi giả định của bác sĩ: “có nổi mẩn không?”, “có lan ra không?”.
* Danh sách tác dụng phụ chung của thuốc.
* Cảnh báo điều có thể xảy ra trong tương lai.
* Phần diễn giải phụ sau một triệu chứng chính.
* Một mệnh đề mô tả dài không có tên triệu chứng tự nhiên.

Không xóa tự động mọi triệu chứng trong đoạn giáo khoa. Chỉ giữ khi occurrence vẫn là một symptom độc lập được gold chấp nhận; khi không chắc, ưu tiên precision.

### 3.2. CHẨN_ĐOÁN

Giữ:

* Tên bệnh hoặc tình trạng bệnh lý độc lập.
* Chẩn đoán hiện tại.
* Tiền sử bệnh.
* Khả năng chẩn đoán do bác sĩ nêu.
* Bệnh cần phòng ngừa.
* Tên biến chứng.
* Tên bệnh xuất hiện trong đoạn giáo khoa nếu nó vẫn là một tên bệnh độc lập.

Bỏ:

* Bệnh nhân chỉ tự hỏi mình có mắc bệnh hay không, khi không có nhận định y khoa hỗ trợ.
* Trạng thái sinh lý chung như “có thai”, “mãn kinh”, trừ khi dữ liệu đã xác nhận occurrence cụ thể cần giữ.
* Quá trình bệnh học: “nung mủ”, “hoại tử lan rộng”, “nhu mô bị đông đặc”.
* Kết cục chung: “suy các cơ quan”.
* Lời giải nghĩa: “máu có xu hướng vón cục”.
* Cả mệnh đề cơ chế hoặc mô tả.
* Cụm quá chung như “nhiễm khuẩn”, “nhiễm virus” khi không hoạt động như chẩn đoán cụ thể.

### 3.3. TÊN_XÉT_NGHIỆM

Giữ tên xét nghiệm hoặc thủ thuật cụ thể, kể cả khi:

* Chưa thực hiện.
* Chỉ mới được bác sĩ đề nghị.
* Chưa có kết quả.

Ví dụ có thể giữ:

* `nội soi dạ dày`
* `siêu âm tim`
* `chụp động mạch vành`
* `xét nghiệm nội tiết`
* `chụp cắt lớp vi tính đầu`
* `sinh thiết tuyến tiền liệt`

Bỏ:

* `xét nghiệm` đứng một mình.
* `nội soi` khi chỉ nói chung về lịch sử phát triển kỹ thuật.
* `khám chuyên khoa`
* `khám lâm sàng`
* `chẩn đoán hình ảnh`
* Heading hoặc tên lĩnh vực.
* Tên thiết bị như kính hiển vi nếu không phải tên một xét nghiệm/thủ thuật.

### 3.4. KẾT_QUẢ_XÉT_NGHIỆM

Giữ khi occurrence là giá trị hoặc finding gắn với một xét nghiệm, thủ thuật hay quan sát lâm sàng cụ thể trong ca bệnh.

Kết quả có thể là:

* Số.
* Chữ: `dương tính`, `âm tính`.
* Finding hình ảnh.
* Mô tả nội soi.
* Mô tả xét nghiệm bằng văn bản dài.

Không bắt buộc tên xét nghiệm phải nằm cạnh kết quả.

Nếu input có đơn vị, span phải lấy cả đơn vị. Nếu input không có đơn vị, không tự thêm đơn vị.

Bỏ:

* Mô tả mô bệnh học chỉ nằm trong đoạn kiến thức giáo khoa.
* Câu giải thích chung về cơ chế xét nghiệm.
* Finding giả định, chưa quan sát.
* Tên bệnh đang bị gán nhầm thành kết quả xét nghiệm.

### 3.5. THUỐC

Giữ:

* Tên thuốc cụ thể.
* Nhóm thuốc khi occurrence thực sự nói về thuốc bệnh nhân dùng, đã dùng, được kê hoặc được khuyên dùng.
* Tên thuốc trong tiền sử.

Không gán nhãn cho:

* Hành vi liên quan thuốc nhưng không có tên thuốc.
* Những từ có chữ “thuốc” nhưng chỉ là lời nói chung, nếu không chỉ ra một thuốc hoặc nhóm thuốc có ý nghĩa.

## 4. Quy tắc boundary

* `text` phải là substring nguyên văn của input.
* Không sửa chính tả hoặc chuẩn hóa text.
* Không thêm từ không có trong input.
* Chọn span cụ thể, tự nhiên và đầy đủ nhất cho concept.
* Không lấy cả câu khi chỉ một cụm con là concept.
* Không lấy span con quá chung nếu đã có span cụ thể hơn.

Ví dụ:

* Giữ `đau bụng râm ran`, không thêm riêng `đau`.
* Giữ `nội soi dạ dày`, không thêm riêng `nội soi`.
* Giữ `đại tiện ra máu đỏ tươi gián đoạn`, không thêm riêng `đi tiêu ra máu`.
* Giữ `viêm sung huyết hang vị dạ dày`, không lấy chữ `bệnh` đứng ngoài nếu tên tự nhiên không cần chữ đó.
* Nếu có `12 mmol/L`, lấy cả `12 mmol/L`.

## 5. Repeated mention

Nếu một concept xuất hiện nhiều lần, mỗi occurrence hợp lệ phải có một entity riêng.

Tuy nhiên, không được thêm occurrence chỉ vì cùng text đã được tag ở nơi khác.

Mỗi occurrence lặp vẫn phải vượt qua kiểm tra ngữ cảnh:

* Có đang là concept lâm sàng độc lập không?
* Có phải triệu chứng của ca bệnh không?
* Hay chỉ nằm trong giáo khoa, cảnh báo hoặc giả định?

Repeated mention trong cùng file là bằng chứng recall mạnh, nhưng không thay thế việc xét ngữ cảnh.

Không cross-file transfer chỉ dựa trên text. Chỉ chuyển annotation khi đoạn văn và vai trò ngữ cảnh thực sự giống nhau, tốt nhất là trùng nguyên văn.

## 6. Quy trình quyết định cho từng candidate

Với mỗi cụm từ có vẻ y khoa, lần lượt hỏi:

1. Cụm này có phải tên một concept y tế độc lập không?

 * Nếu không: bỏ.

2. Trong occurrence này, cụm đang là:

 * Thông tin của ca bệnh?
 * Tên bệnh độc lập?
 * Xét nghiệm/thủ thuật cụ thể?
 * Finding thực tế?
 * Thuốc thực tế?
 * Hay chỉ là giải thích, giáo khoa, giả định hoặc cảnh báo?

3. Span có quá rộng không?

 * Nếu chứa cả mệnh đề: co về concept.
 * Nếu không thể co thành một concept nguyên văn tự nhiên: bỏ.

4. Span có quá ngắn hoặc quá chung không?

 * Nếu đã có cụm cụ thể hơn: chỉ giữ cụm cụ thể.

5. Type có phản ánh đúng vai trò semantic không?

 * Không type theo các từ nằm gần occurrence.

6. Occurrence có lặp không?

 * Thêm từng occurrence nếu chính nó hợp lệ.
 * Không thêm máy móc theo text.

## 7. Few-shot examples

### Ví dụ 1 — tên bệnh và lời giải thích

Input:

`Viêm hang vị sung huyết là tình trạng niêm mạc vùng hang vị dạ dày viêm, các mạch máu vùng viêm giãn nở do ứ máu nhiều.`

Giữ:

```json
{
 "text": "Viêm hang vị sung huyết",
 "type": "CHẨN_ĐOÁN"
}
```

Bỏ:

* `niêm mạc vùng hang vị dạ dày viêm`
* `các mạch máu vùng viêm giãn nở do ứ máu nhiều`

Lý do: hai cụm sau là mệnh đề giải thích cơ chế, không phải tên chẩn đoán độc lập.

### Ví dụ 2 — xét nghiệm cụ thể và kỹ thuật nói chung

Input A:

`Bác sĩ khuyên tôi nội soi dạ dày.`

Giữ:

```json
{
 "text": "nội soi dạ dày",
 "type": "TÊN_XÉT_NGHIỆM"
}
```

Input B:

`Mấy chục năm gần đây nhờ nội soi phát triển nên bệnh được điều trị hiệu quả hơn.`

Bỏ `nội soi`.

Lý do: occurrence B chỉ nói chung về lịch sử kỹ thuật.

### Ví dụ 3 — triệu chứng bệnh nhân và triệu chứng giáo khoa

Input A:

`Một tuần nay em bị đau thượng vị, ợ hơi và ợ chua.`

Giữ:

```json
{"text": "đau thượng vị", "type": "TRIỆU_CHỨNG"}
{"text": "ợ hơi", "type": "TRIỆU_CHỨNG"}
{"text": "ợ chua", "type": "TRIỆU_CHỨNG"}
```

Input B:

`Biểu hiện chủ yếu là đau bụng cồn cào, kèm theo ợ hơi, ợ chua, có thể có cảm giác buồn nôn hoặc nôn.`

Giữ:

```json
{"text": "đau bụng cồn cào", "type": "TRIỆU_CHỨNG"}
```

Bỏ trong occurrence B:

* `ợ hơi`
* `ợ chua`
* `buồn nôn`
* `nôn`

Lý do: chúng chỉ là danh sách biểu hiện phụ trong phần mô tả giáo khoa đã được xác nhận là false positive.

### Ví dụ 4 — bệnh nhân tự hỏi và bác sĩ nhận định

Input A:

`Tình trạng của em có phải là tăng huyết áp thật sự không?`

Bỏ `tăng huyết áp thật sự` nếu chỉ là bệnh nhân tự suy đoán.

Input B:

`Bác sĩ nói khả năng em bị viêm bao tử.`

Giữ:

```json
{
 "text": "viêm bao tử",
 "type": "CHẨN_ĐOÁN"
}
```

### Ví dụ 5 — bệnh chưa xảy ra nhưng vẫn là tên bệnh độc lập

Input:

`Thuốc được sử dụng để phòng ngừa tiền sản giật.`

Giữ:

```json
{
 "text": "tiền sản giật",
 "type": "CHẨN_ĐOÁN"
}
```

Lý do: không bắt buộc bệnh phải đang xảy ra; đây vẫn là một tên bệnh độc lập.

### Ví dụ 6 — quá trình bệnh học trong câu định nghĩa

Input:

`Áp xe phổi là tình trạng nung mủ, hoại tử nhu mô phổi và nhu mô phổi bị đông đặc.`

Giữ:

```json
{
 "text": "Áp xe phổi",
 "type": "CHẨN_ĐOÁN"
}
```

Bỏ:

* `nung mủ`
* `hoại tử nhu mô phổi`
* `nhu mô phổi bị đông đặc`

Lý do: đây là các bước diễn giải cơ chế, không phải các chẩn đoán độc lập trong occurrence này.

### Ví dụ 7 — mô bệnh học giáo khoa và finding thực tế

Input A:

`Tổn thương mô bệnh học thường có tình trạng xâm nhập của tế bào một nhân kết hợp với lympho T và đại thực bào.`

Bỏ toàn bộ mệnh đề mô bệnh học nếu chỉ nằm trong phần giáo khoa.

Input B:

`Sinh thiết da cho thấy thâm nhiễm lympho T và đại thực bào.`

Giữ finding sau `cho thấy` dưới dạng `KẾT_QUẢ_XÉT_NGHIỆM`, vì nó là kết quả thực tế của một xét nghiệm cụ thể.

### Ví dụ 8 — repeated mention

Input:

`Trước nhập viện bệnh nhân buồn nôn. Trong đêm bệnh nhân tiếp tục buồn nôn.`

Thêm hai entity riêng:

```json
{"text": "buồn nôn", "type": "TRIỆU_CHỨNG"}
{"text": "buồn nôn", "type": "TRIỆU_CHỨNG"}
```

Input khác:

`Bệnh nhân buồn nôn. Buồn nôn cũng là tác dụng phụ thường gặp của thuốc này.`

Chỉ chắc chắn giữ occurrence đầu. Occurrence thứ hai là kiến thức tác dụng phụ chung và không được thêm máy móc.

### Ví dụ 9 — từ chung và tên cụ thể

Input:

`Khi làm xét nghiệm, bác sĩ phát hiện INR 1.7.`

Giữ:

```json
{"text": "INR", "type": "TÊN_XÉT_NGHIỆM"}
{"text": "1.7", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}
```

Bỏ bare `xét nghiệm`.

### Ví dụ 10 — add entity bị bỏ sót

Input:

`Bệnh nhân mệt mỏi nhiều, khó thở khi gắng sức và buồn nôn.`

Nếu các occurrence là thông tin thực tế của bệnh nhân, giữ đầy đủ:

```json
{"text": "mệt mỏi nhiều", "type": "TRIỆU_CHỨNG"}
{"text": "khó thở khi gắng sức", "type": "TRIỆU_CHỨNG"}
{"text": "buồn nôn", "type": "TRIỆU_CHỨNG"}
```

Không rút thành `mệt mỏi` hoặc `khó thở` nếu modifier là một phần tự nhiên của symptom trong input.

## 8. Output

Chỉ trả về JSON theo schema được yêu cầu.

Với mỗi entity:

* `text`: đúng substring nguyên văn.
* `position`: đúng offset của substring trong raw input.
* `type`: một trong năm type đã cho.
* `assertions`: theo schema hiện tại.
* `candidates`: theo convention hiện tại.

Không giải thích ngoài JSON.

Trước khi trả kết quả, tự kiểm tra:

* Mọi `text` có khớp chính xác với `input[start:end]`.
* Không có span vượt biên.
* Không có entity trùng hoàn toàn.
* Không thêm bare subspan khi đã có span cụ thể hơn.
* Không bỏ repeated occurrence hợp lệ.
* Không lấy mệnh đề giải thích hoặc đoạn giáo khoa chỉ vì có từ y khoa.

---

## 9. Đính chính quan trọng nhất: "chẩn đoán chưa xác nhận" (§3.2, Ví dụ 4) chỉ áp dụng cho CÂU HỎI CỦA CHÍNH BỆNH NHÂN

Ví dụ 4 ở trên dạy: bỏ chẩn đoán khi bệnh nhân tự hỏi ("có phải là X không"), giữ khi bác sĩ
nhận định ("bác sĩ nói khả năng X"). Điều này **đúng** nhưng phạm vi hẹp hơn nhiều so với cách
diễn đạt gốc gợi ý. Bằng chứng file 7 (test xoá "viêm bao tử", "trào ngược dạ dày thực quản",
"viêm dạ dày") và file 13 (test xoá "Bệnh dại" trong tiêu đề FAQ) cho thấy **3 tình huống khác
KHÔNG bị loại**, dù bề ngoài giống pattern "chưa chắc chắn":

| Tình huống | Ví dụ | Kết luận |
|---|---|---|
| Bệnh nhân tự hỏi về CHẨN ĐOÁN CỦA CHÍNH MÌNH | "tình trạng của em có phải là tăng HA thật sự không ạ?" | **KHÔNG gán** |
| Bác sĩ hedging bằng ngôn từ dè dặt ("khả năng") | "BS nói khả năng em bị viêm bao tử" | **VẪN GÁN** — nhận định chuyên môn, không phải câu hỏi |
| Tiêu đề FAQ hỏi về BỆNH nói chung (không phải chẩn đoán của 1 bệnh nhân cụ thể) | "Bệnh dại có lây không?" | **VẪN GÁN** tên bệnh trong tiêu đề |
| Cảnh báo hậu quả tương lai dạng "dẫn đến X" | "tránh nhịn ăn dẫn đến viêm dạ dày" | **VẪN GÁN** tên bệnh X |
| Mô tả công dụng điều trị/phòng ngừa nhắc bệnh CỤ THỂ liên quan bệnh nhân | "Nhôm hydroxid... trị... trào ngược dạ dày thực quản" | **VẪN GÁN** |

**Quy tắc thu hẹp đúng**: chỉ loại bỏ khi **chính bệnh nhân** là người phát biểu sự KHÔNG CHẮC
CHẮN về **CHẨN ĐOÁN CỦA BẢN THÂN MÌNH**. Hedge/tiêu đề/cảnh báo đến từ bác sĩ, tài liệu giáo
khoa, hoặc mô tả công dụng điều trị đều KHÔNG thuộc diện loại trừ này.

## 10. Đính chính: "loại bỏ mệnh đề cơ chế/định nghĩa" (§2, §7 Ví dụ 1/6/7) — quyết định theo TỪNG OCCURRENCE, không theo cụm từ

Nguyên tắc gốc (giữ tên bệnh X, bỏ phần diễn giải cơ chế Y trong "X là tình trạng Y") **vẫn
đúng** cho diễn giải sinh lý bệnh THUẦN giáo khoa (file 94: "niêm mạc... viêm", "các mạch
máu... ứ máu" — cả 2 xác nhận loại bỏ). Nhưng có bằng chứng quan trọng bổ sung:

- File 27 (áp xe phổi): `nung mủ`, `hoại tử chủ mô phổi`, `nhu mô phổi bị đông đặc` — loại bỏ,
 đúng quy tắc gốc. **NHƯNG** cùng file có `hoại tử nhu mô phổi` ở **vị trí khác** — occurrence
 đó KHÔNG bị xoá trong test tương ứng. → **Quyết định theo TỪNG OCCURRENCE, không theo cụm từ
 chung**: nếu cùng 1 cụm xuất hiện cả trong câu định nghĩa (bỏ) và trong câu mô tả finding thật
 của ca bệnh (giữ), xử lý khác nhau theo đúng vị trí, không gán nhãn hàng loạt theo text.
- File 34 (migraine): `căng thẳng` (trigger, không phải triệu chứng) — loại bỏ khi đứng như yếu
 tố khởi phát trong đoạn giáo khoa.
- File 30/44/76/83: mô tả mô bệnh học dạng "xâm nhập tế bào..." — finding GIÁO KHOA thuần
 (không gắn 1 ca bệnh cụ thể) → loại bỏ, xác nhận nhất quán ở cả 4 file gần-trùng.

## 11. Quy tắc MỚI: nội dung trong mục có CẤU TRÚC rõ (heading) được ưu tiên hơn mô tả lặp trong câu hỏi tự do

Khi 1 sự kiện/triệu chứng được nhắc 2 lần — một lần trong đoạn câu hỏi/trả lời tự do (văn
nói, kể lể), một lần trong mục có cấu trúc rõ ràng (vd `"2. Bệnh sử hiện tại"`, `"Lý do nhập
viện:"`) — gold có xu hướng **CHỈ giữ bản trong mục có cấu trúc**, bỏ bản kể lể tự do dù cùng
1 sự kiện:

- `"cục máu đông"`, `"đi tiêu ra máu"`, `"chảy máu"` (trong đoạn hỏi-đáp tự do) → xác nhận
 loại bỏ (mỗi cái test riêng).
- `"đại tiện ra máu đỏ tươi gián đoạn"` (trong mục `"Lý do nhập viện:"` có cấu trúc) → xác nhận
 giữ.

**Cách áp dụng**: khi 2 lần mô tả cùng 1 hiện tượng lâm sàng ở 2 vị trí khác thể loại văn bản
(kể lể tự do vs mục tóm tắt có tiêu đề), ưu tiên gán ở mục có cấu trúc, cân nhắc bỏ bản kể lể
trùng lặp — đặc biệt khi bản kể lể chỉ diễn giải sơ sài/tổng quát hơn.

## 12. Quy tắc MỚI: "quyết định y khoa CHƯA CHỐT ở tương lai" loại trừ cả THUỐC, không chỉ CHẨN_ĐOÁN

*"Bác sĩ **sẽ** so sánh nguy cơ, lợi ích của việc tiếp tục sử dụng aspirin và cho bạn chỉ định
tiếp tục hay ngưng dùng thuốc."* → xoá "aspirin" ở đây xác nhận **giúp điểm**, dù là tên thuốc
cụ thể (loại vốn dễ tin cậy nhất).

**Quy tắc**: khi 1 THUỐC được nhắc trong khung "bác sĩ SẼ quyết định tiếp tục/ngưng" (quyết
định y khoa CHƯA XẢY RA, đang chờ đánh giá) — khác thuốc đã kê/đang dùng thật — nhiều khả năng
KHÔNG được gán. Đây mở rộng "sự việc chưa xảy ra vẫn gán" (§3.2 "bệnh cần phòng ngừa", §7 Ví
dụ 5) theo chiều NGƯỢC LẠI: không phải MỌI thứ-chưa-xảy-ra đều giữ — quyết định điều trị đang
treo/chưa chốt có xu hướng loại, khác hẳn "xét nghiệm khuyên làm" hay "bệnh phòng ngừa" (vẫn
giữ). Ranh giới: xét thuốc/xét nghiệm này có mô tả 1 HÀNH ĐỘNG Y KHOA CỤ THỂ ĐÃ/ĐANG XẢY RA hay
chỉ là 1 lựa chọn tương lai còn để ngỏ.

## 13. Quy tắc MỚI: câu bổ sung/chi tiết hoá liên tục cho CÙNG 1 diễn biến KHÔNG tag riêng, dù là thông tin thật

Đoạn mô tả cơn đau bụng cấp — `"đau bụng râm ran"` và `"không thể nhúc nhích được chân tay"`
cùng mô tả 1 diễn biến liên tục (đau tới mức không cử động được). Cả 2 xác nhận LOẠI BỎ — dù
đều là thông tin THẬT, không giả định, không phải câu hỏi.

**Quy tắc**: khi 1 đoạn văn liên tục mô tả CÙNG 1 sự kiện/triệu chứng chính bằng NHIỀU câu chi
tiết hoá nối tiếp (không phải các occurrence tách biệt ở thời điểm/vị trí khác), gold có xu
hướng chỉ giữ 1 câu cốt lõi đại diện, không tag riêng từng câu bổ sung. Đây KHÁC §5 (mỗi lần
xuất hiện = 1 khái niệm) — §5 áp dụng cho occurrence THỰC SỰ TÁCH BIỆT (khác thời điểm/vị trí
trong câu chuyện), không áp dụng cho câu bổ sung liên tục trong CÙNG 1 lần mô tả.

## 14. Đính chính: KHÔNG có quy tắc chung "bỏ tiền tố bệnh/bệnh lý" — luôn kiểm tra riêng từng tên

| Test | Hành động | Kết quả |
|---|---|---|
| `"bệnh viêm sung huyết hang vị dạ dày"` → bỏ `"bệnh"` | Trim | Giúp điểm (nên bỏ) |
| `"bệnh lý mày đay vô căn"` → bỏ `"bệnh lý"` | Trim | Giúp điểm (nên bỏ) |
| `"bệnh amyloidosis"` → bỏ `"bệnh"` | Trim | **Hại điểm (KHÔNG nên bỏ)** |

**Quy tắc thực dụng**: đừng áp máy móc "luôn cắt bệnh/bệnh lý". Đặc biệt cẩn trọng với cụm dạng
`"X (hay/là Y)"` — tên khoa học/tên thay thế trong ngoặc (vd "Rối loạn chuyển hóa tinh bột
(amyloidosis)") — bằng chứng cho thấy **cặp tên-kèm-chú-thích-trong-ngoặc nên giữ NGUYÊN VẸN
CẢ CỤM** (không tách thành 2 entity, không trim riêng phần trong ngoặc — tách ra đã xác nhận
hại điểm), khác với "bệnh"/"bệnh lý" đứng trước 1 tên bệnh thường không có ngoặc (trường hợp đó
thường, nhưng không phải luôn, bỏ được).

## 15. Quy tắc MỚI: mô tả CHỨC NĂNG/KHẢ NĂNG chung chung không phải CHẨN_ĐOÁN hay TRIỆU_CHỨNG

Các câu khẳng định 1 CHỨC NĂNG/BỘ PHẬN vẫn hoạt động bình thường (hoặc mô tả khả năng chung,
không phải 1 finding cụ thể có tên) — xác nhận loại bỏ khi thử thêm dạng:

- `"cấu trúc tai trong đảm nhận chức năng thần kinh thính giác của con thường vẫn phát triển
 tốt"` là CHẨN_ĐOÁN.
- `"phản xạ được âm thanh"` là CHẨN_ĐOÁN.
- `"nhìn bên ngoài"` là TÊN_XÉT_NGHIỆM (cùng nhóm với `"khám lâm sàng"` đã loại ở §3.3).

## 16. Quy tắc MỚI: mô tả gộp/mơ hồ nên TÁCH thành CHẨN_ĐOÁN cụ thể riêng biệt nếu có tên chuẩn

Thay vì giữ nguyên `"tai phải bé không có lỗ tai"` (mô tả gộp, đã loại), tách thành 2 CHẨN_ĐOÁN
cụ thể riêng biệt — `"dị tật thiểu sản vành tai"` và `"tịt ống tai ngoài bẩm sinh"` — xác nhận
GIÚP ĐIỂM MẠNH, kể cả khi test độc lập chỉ 2 entity này. Ngược lại, `"vành tai không phát triển
đều"`, `"nhỏ hơn tai còn lại"`, `"bên tai trái thì rất to"` (mô tả so sánh/định tính thuần,
không có tên chuẩn tương ứng) → xác nhận loại bỏ.

**Quy tắc**: khi văn bản chỉ mô tả bằng lời thông thường (dạng "X bé không có Y", "X không
phát triển đều") NHƯNG có tồn tại 1 TÊN CHẨN ĐOÁN CHUẨN, CỤ THỂ tương ứng (dị tật bẩm sinh có
tên y khoa riêng) → ưu tiên gán tên chẩn đoán chuẩn, không gán mô tả gộp bằng lời thường. Nếu
không có tên chuẩn tương ứng, mô tả so sánh/định tính thuần nhiều khả năng bị loại hoàn toàn.

## 17. Xác nhận thêm — 3 điểm nhỏ nhưng dễ sai

- **Tên thuốc/vaccine cụ thể trong văn giải thích chung (FAQ) vẫn được gán**: `"vaccine phòng
 dại"` xoá → hại điểm (nên giữ). Củng cố §3.5 — tên thuốc/vaccine CỤ THỂ luôn gán bất kể ngữ
 cảnh câu, khác nhóm thuốc chung chung.
- **Đừng mở rộng tên bệnh bằng hậu tố mô tả/nguyên nhân nếu GT không có**: `"Bệnh dại"` → mở
 rộng thành `"Bệnh dại do Lyssavirus"` → hại điểm. Giữ đúng độ dài span văn bản gốc thể hiện,
 đừng thêm hậu tố đúng-về-chuyên-môn nhưng không có trong text (WER so khớp từng từ).
- **Đừng tự ý gắn thêm assertion/rename entity theo cách "làm gọn" khi không có bằng chứng**:
 thêm `isHistorical` + đổi tên "bệnh dại" thành "bệnh dại ở người" → hại điểm.

## 18. Case CHƯA đủ bằng chứng kết luận — đừng lặp lại vô ích

| Ngữ cảnh | Test | Trạng thái |
|---|---|---|
| Migraine | xoá `"sợ ánh sáng"`, `"sợ tiếng động"` | Chưa chốt — có bảng suy luận nhưng thiếu kết luận cuối, cần test lại nếu gặp pattern tương tự |
| Tai (dị tật bẩm sinh) | thêm `"nghe khá tốt"` / `"chỉ nghe một bên"` / `"đo âm tai"` (kèm 2 CHẨN_ĐOÁN chắc) | Không có kết quả ghi nhận — không dùng làm bằng chứng |
| Mô bệnh học da (4 file gần-trùng) | boundary ngắn vs dài cho "các tế bào... thoát ra khác nhau" | Có khung so sánh B1/B2 nhưng chưa có kết luận cuối rõ ràng |

**Nguyên tắc chung**: nếu không có kết quả TĂNG/GIẢM/BẰNG ĐIỂM rõ ràng ghi nhận được, đừng suy
diễn kết luận từ ghi chú lý luận suông — cần test thật trước khi đưa vào rule.

## 19. Quy trình quyết định — bổ sung 5 câu hỏi cho case biên (sau khi qua §6 gốc)

1. Đây có phải lời phát biểu/hedge của **BÁC SĨ** (không phải câu hỏi của bệnh nhân về chính
 họ) không? → Nếu có, đừng loại chỉ vì ngôn từ dè dặt (§9).
2. Đây có phải diễn giải cơ chế THUẦN giáo khoa, không gắn finding/ca bệnh cụ thể không? Kiểm
 tra RIÊNG từng occurrence, đừng áp theo cụm từ (§10).
3. Cùng sự kiện có được nhắc ở CẢ mục có cấu trúc LẪN đoạn kể lể tự do không? → Ưu tiên giữ bản
 trong mục có cấu trúc (§11).
4. Đây có phải quyết định y khoa ĐANG CÒN ĐỂ NGỎ ở tương lai (chưa chốt tiếp tục hay ngưng)
 không, kể cả với THUỐC? → Cân nhắc loại (§12).
5. Đây có phải câu bổ sung/chi tiết hoá liên tục cho 1 sự kiện ĐÃ có câu cốt lõi khác trong
 cùng đoạn không? → Không tag riêng, dù là thông tin thật (§13).
6. Có tên chẩn đoán CHUẨN, CỤ THỂ hơn có thể thay cho mô tả gộp bằng lời thường không? → Ưu
 tiên tên chuẩn, bỏ mô tả gộp (§16).
7. Đây có đơn thuần là khẳng định 1 chức năng/bộ phận hoạt động bình thường, không phải 1
 finding cụ thể không? → Không tag (§15).
