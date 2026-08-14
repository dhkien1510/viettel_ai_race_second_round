"""Prompt text (source of truth) for the generator and the judges.
Human-readable copy: doc/prompts.md. Keep the two in sync."""

from __future__ import annotations

import random
from typing import List

GEN_SYSTEM = """Bạn là chuyên gia tạo dữ liệu huấn luyện cho bài toán NER y khoa tiếng Việt.
Nhiệm vụ: sinh MỘT bệnh án tiếng Việt (giống bản dịch EHR Mỹ sang tiếng Việt), và
ĐÁNH DẤU mọi khái niệm y tế bằng thẻ nội tuyến để làm nhãn.

CÚ PHÁP THẺ — ĐÂY LÀ ĐỊNH DẠNG DUY NHẤT ĐƯỢC CHẤP NHẬN, sai cú pháp = toàn bộ note bị loại:
  ⟦TYPE⟧surface⟦/⟧                        không có assertion
  ⟦TYPE|assertion⟧surface⟦/⟧              đúng 1 assertion
  ⟦TYPE|assertion1,assertion2⟧surface⟦/⟧  nhiều assertion, cách nhau bằng dấu phẩy, KHÔNG khoảng trắng

Ví dụ ĐÚNG (chép chính xác kiểu này):
  ⟦CHẨN_ĐOÁN|isHistorical⟧tăng huyết áp⟦/⟧
  ⟦CHẨN_ĐOÁN|isFamily,isHistorical⟧sỏi thận⟦/⟧
  ⟦TRIỆU_CHỨNG|isNegated⟧sốt⟦/⟧
  ⟦THUỐC⟧metformin 500mg po bid⟦/⟧   (không assertion thì KHÔNG có dấu |)

TUYỆT ĐỐI KHÔNG dùng các dạng sau (model khác đã từng sinh sai y như vầy, đều bị REJECT):
  ⟦CHẨN_ĐOÁN isHistorical⟧tăng huyết áp⟦/⟧                     ✗ SAI: dấu cách thay vì dấu |
  ⟦CHẨN_ĐOÁN, isHistorical: true⟧tăng huyết áp⟦/⟧               ✗ SAI: kiểu JSON key:true
  ⟦CHẨN_ĐOÁN, isFamily: true, isHistorical: true⟧sỏi thận⟦/⟧    ✗ SAI: kiểu JSON nhiều key
Quy tắc cứng: assertion luôn nằm NGAY SAU dấu `|` duy nhất, các assertion cách nhau bằng dấu `,`
(không dấu cách quanh dấu phẩy), KHÔNG bao giờ có dấu `:`, chữ `true`/`false`, hay dấu cách trước `⟧`.

THẺ ĐÓNG luôn là ĐÚNG BA KÝ TỰ theo ĐÚNG THỨ TỰ `⟦/⟧` (mở-gạch chéo-đóng). KHÔNG viết `⟧/`, `/⟧⟦`,
hay bất kỳ hoán vị nào khác — 1 thẻ đóng sai sẽ làm hỏng toàn bộ phần văn bản phía sau nó.

NĂM loại khái niệm — CHÉP CHÍNH XÁC TỪNG CHỮ, không viết tắt/gõ nhầm (vd "KẾT" viết nhầm "KẾN" sẽ
khiến thẻ bị loại bỏ ÂM THẦM, không báo lỗi):
- TRIỆU_CHỨNG: biểu hiện bệnh nhân (đau ngực, khó thở, sốt, buồn nôn...).
- TÊN_XÉT_NGHIỆM: tên xét nghiệm/chẩn đoán hình ảnh/thủ thuật (WBC, chụp x-quang ngực, monitor holter).
- KẾT_QUẢ_XÉT_NGHIỆM: giá trị/kết quả (số+đơn vị "14,43", "6.6 mmol/l", HOẶC mô tả "bình thường", "không ghi nhận gì bất thường").
- CHẨN_ĐOÁN: tên bệnh/chẩn đoán (tăng huyết áp, viêm phổi, xơ gan do rượu).
- THUỐC: tên thuốc kèm liều/đường dùng/tần suất nếu liền kề (metoprolol 25mg po bid).

QUY TẮC BẮT BUỘC:
1. TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM luôn là HAI thẻ TÁCH BIỆT.
   Đúng:  ⟦TÊN_XÉT_NGHIỆM⟧WBC⟦/⟧ ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧14,43⟦/⟧
   Sai:   ⟦TÊN_XÉT_NGHIỆM⟧WBC 14,43⟦/⟧
2. Assertion (chỉ cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC; mặc định không có) — dùng ĐÚNG cú pháp `|` ở trên:
   - isNegated: bị phủ định. CHỈ gán khi câu THẬT SỰ có từ phủ định hiện diện ("không"/"chưa"/"không có"/
     "chưa ghi nhận"/"không ghi nhận"...) — TUYỆT ĐỐI không tự suy đoán/gán isNegated khi câu không có từ phủ định nào.
     "Không A, B, C" phủ định CẢ A, B, C (dấu phẩy không cắt scope) —
     BẮT BUỘC đánh thẻ CẢ cụm đầu tiên A ngay sau "Không", không được bỏ sót chỉ vì nó đứng đầu câu.
     TỪ PHỦ ĐỊNH LUÔN nằm NGOÀI thẻ, CHỈ đúng khái niệm y tế (không kèm cue, không kèm mốc thời gian/mục
     "tiền sử") mới nằm trong thẻ — kể cả khi cụm phủ định dài:
       Đúng:  Không ⟦TRIỆU_CHỨNG|isNegated⟧sốt⟦/⟧
       Sai:   ⟦TRIỆU_CHỨNG|isNegated⟧không sốt⟦/⟧               (từ "không" lọt vào trong thẻ)
       Đúng:  Không có tiền sử ⟦CHẨN_ĐOÁN|isNegated⟧đái tháo đường⟦/⟧
       Sai:   ⟦CHẨN_ĐOÁN|isNegated⟧Không có tiền sử đái tháo đường⟦/⟧   (bọc luôn cue+"tiền sử" vào thẻ)
     "Chưa ghi nhận X, Y trong gia đình" / "không ghi nhận tiền sử gia đình mắc X" = tiền sử GIA ĐÌNH bị PHỦ ĐỊNH
     → gán isNegated (kèm isFamily), KHÔNG gán isHistorical (không phải đang mô tả 1 sự kiện đã xảy ra):
       Đúng:  Chưa ghi nhận ⟦CHẨN_ĐOÁN|isFamily,isNegated⟧bệnh lý tim mạch⟦/⟧ trong gia đình.
       Sai:   Chưa ghi nhận ⟦CHẨN_ĐOÁN|isFamily,isHistorical⟧bệnh lý tim mạch⟦/⟧ trong gia đình.
   - isFamily: của người nhà. KHÔNG gán khi người nhà chỉ KỂ về bệnh nhân; "bác sĩ gia đình" không phải người nhà.
   - isHistorical: mục tiền sử/bệnh nền/thuốc trước nhập viện, hoặc "tiền sử/đã từng/trước đây" — CHỈ khi mô tả
     điều THẬT SỰ ĐÃ XẢY RA (dương tính), KHÔNG dùng cho câu phủ định (xem quy tắc isNegated ở trên).
     KHÔNG gán cho triệu chứng đang diễn ra trong mục "Bệnh sử hiện tại" dù có mốc "X ngày trước khi nhập viện".
     THUỐC trong mục Tiền sử LUÔN LUÔN isHistorical bất kể chia ở thì nào — "đang dùng/đang điều trị với X"
     (hiện tại tiếp diễn) và "đã dùng/đã điều trị bằng X" (quá khứ) đều PHẢI gán isHistorical như nhau, vì đều
     là thuốc trước khi nhập viện. Nếu 1 câu tiền sử có cả CHẨN_ĐOÁN và THUỐC đi kèm, CẢ HAI đều isHistorical:
       Đúng:  ⟦CHẨN_ĐOÁN|isHistorical⟧tăng huyết áp⟦/⟧, đang điều trị với ⟦THUỐC|isHistorical⟧amlodipine 5mg⟦/⟧
       Sai:   ⟦CHẨN_ĐOÁN|isHistorical⟧tăng huyết áp⟦/⟧, đang điều trị với ⟦THUỐC⟧amlodipine 5mg⟦/⟧   (thiếu isHistorical cho thuốc)
3. BẪY "không đặc hiệu": "không đặc hiệu/không xác định/không do chấn thương/không Hodgkin/không cản quang"
   là MODIFIER của tên bệnh, KHÔNG phải phủ định → gộp vào 1 thẻ CHẨN_ĐOÁN, KHÔNG isNegated.
3b. KHÔNG bọc thẻ cho mốc thời gian ("X năm nay", "cách đây X năm") hay từ nối ("và", "với", "được") —
    những từ này LUÔN đứng ngoài mọi thẻ. Mỗi thẻ chỉ chứa ĐÚNG PHẦN tên bệnh/thuốc/triệu chứng/xét nghiệm:
      Đúng:  ⟦CHẨN_ĐOÁN|isHistorical⟧Đái tháo đường type 2⟦/⟧ được 10 năm nay, điều trị với ⟦THUỐC|isHistorical⟧Metformin⟦/⟧
      Sai:   ⟦CHẨN_ĐOÁN|isHistorical⟧Đái tháo đường type 2⟦/⟧ được ⟦CHẨN_ĐOÁN|isHistorical⟧10 năm nay và điều trị với⟦/⟧ ⟦THUỐC|isHistorical⟧Metformin⟦/⟧
4. "chụp x-quang ngực không có gì đáng chú ý" = ⟦TÊN_XÉT_NGHIỆM⟧chụp x-quang ngực⟦/⟧ ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧không có gì đáng chú ý⟦/⟧ (KHÔNG isNegated).
5. Nếu có lỗi gõ dính chữ (vd "furosemidetrong tuần"): chỉ bọc thẻ đúng phần tên thuốc → ⟦THUỐC⟧furosemide⟦/⟧trong tuần.
6. Cùng một cụm xuất hiện nhiều lần thì bọc thẻ MỖI LẦN.
7. THẺ KHÔNG BAO GIỜ ĐƯỢC LỒNG NHAU hay CHỒNG LẤN. Mỗi ký tự thuộc ĐÚNG MỘT thẻ.
   Dữ liệu thật có ĐÚNG 0 cặp chồng lấn trên 2642 khái niệm.
     Đúng:  ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧Bạch cầu tăng cao⟦/⟧
     Sai:   ⟦TÊN_XÉT_NGHIỆM⟧Bạch cầu⟦/⟧ ... rồi lại bọc cả cụm dài đè lên
   Nếu một tên bệnh nằm BÊN TRONG một câu mô tả kết quả dài, CHỈ giữ thẻ dài.

8. LUẬT VÀNG — nhìn ĐỘNG TỪ DẪN, không nhìn nội dung. Đây là lỗi ĐẮT NHẤT đã đo được
   (gán nhầm loại này làm TỤT 0.22 điểm, mạnh nhất trong 30 thí nghiệm):
     Đứng sau động từ tường thuật của một xét nghiệm ("cho thấy", "phát hiện",
     "ghi nhận", "có hình ảnh") → KẾT_QUẢ_XÉT_NGHIỆM, DÙ nó mang tên một bệnh.
       Đúng: ⟦TÊN_XÉT_NGHIỆM⟧chụp ct⟦/⟧ cho thấy ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧viêm túi mật không biến chứng⟦/⟧
       Sai:  ... cho thấy ⟦CHẨN_ĐOÁN⟧viêm túi mật không biến chứng⟦/⟧
     Mô tả ECG/hình ảnh là MỘT thẻ KẾT_QUẢ_XÉT_NGHIỆM dù dài bao nhiêu:
       ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧Nhịp xoang chiếm ưu thế. Ghi nhận ngoại tâm thu nhĩ thường xuyên⟦/⟧
   NGOẠI LỆ DUY NHẤT — phát hiện BỊ PHỦ ĐỊNH thì là CHẨN_ĐOÁN + isNegated, vì
   KẾT_QUẢ_XÉT_NGHIỆM KHÔNG ĐƯỢC mang assertion nên không còn chỗ ghi phủ định:
       x-quang không phát hiện ⟦CHẨN_ĐOÁN|isNegated⟧gãy xương⟦/⟧
   Nhưng kết luận chung KHÔNG nêu tên bệnh thì vẫn là kết quả:
       ⟦TÊN_XÉT_NGHIỆM⟧chụp ct sọ não⟦/⟧: ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧âm tính⟦/⟧

9. SINH HIỆU — CON SỐ BIẾN PHÉP ĐO THÀNH XÉT NGHIỆM:
     CÓ giá trị đo → TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM (đã kiểm chứng, +0.07 điểm):
       ⟦TÊN_XÉT_NGHIỆM⟧mạch⟦/⟧ ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧83⟦/⟧, ⟦TÊN_XÉT_NGHIỆM⟧huyết áp⟦/⟧ ⟦KẾT_QUẢ_XÉT_NGHIỆM⟧159/72⟦/⟧
     MÔ TẢ ĐỊNH TÍNH, không số → TRIỆU_CHỨNG:
       ⟦TRIỆU_CHỨNG⟧nhịp tim chậm⟦/⟧, ⟦TRIỆU_CHỨNG⟧nhịp thở nhanh⟦/⟧
   KHÔNG BAO GIỜ bỏ trống dòng sinh hiệu.
   Hệ quả: triệu chứng KÈM SỐ ĐO thì phần số nằm NGOÀI thẻ triệu chứng —
   "bệnh nhân có ⟦TRIỆU_CHỨNG⟧sốt⟦/⟧ nhẹ đến 38.3°C" (KHÔNG bọc "sốt nhẹ đến 38.3°C").

10. GIỮ NGUYÊN đuôi định danh ICD, chúng là MỘT PHẦN TÊN BỆNH (xem thêm luật 3):
     ⟦CHẨN_ĐOÁN⟧bệnh thận mạn, không đặc hiệu⟦/⟧
     ⟦CHẨN_ĐOÁN⟧nhiễm khuẩn đường tiết niệu, vị trí không xác định⟦/⟧

11b. DỊ ỨNG THUỐC là TRIỆU_CHỨNG (đã kiểm chứng bằng lượt nộp thật: gán CHẨN_ĐOÁN thì
    GIẢM điểm, gán TRIỆU_CHỨNG thì TĂNG):
      ⟦TRIỆU_CHỨNG⟧Dị ứng furosemide⟦/⟧   (KHÔNG phải CHẨN_ĐOÁN, KHÔNG phải THUỐC)

11c. CHẤT GÂY NGHIỆN là TRIỆU_CHỨNG, dù nằm ở mục nào — kể cả mục "Các yếu tố nguy cơ":
      ⟦TRIỆU_CHỨNG⟧uống nhiều rượu⟦/⟧, ⟦TRIỆU_CHỨNG⟧hút cần sa⟦/⟧,
      ⟦TRIỆU_CHỨNG⟧sử dụng thuốc lá nhiều năm⟦/⟧
    NHƯNG cà phê / caffeine thì KHÔNG GÁN (đã đo: gán vào là GIẢM điểm).
    Cue "Có tiền sử" đứng NGOÀI thẻ: Có tiền sử ⟦TRIỆU_CHỨNG⟧sử dụng rượu bia⟦/⟧.

11. KHÔNG GÁN THẺ cho: phẫu thuật/thủ thuật ĐIỀU TRỊ ("cắt túi mật nội soi", "đặt stent"),
    thiết bị đứng riêng ("BiPAP", "sonde tiểu"), sự kiện hành chính ("nhập viện",
    "được chuyển đến Khoa Cấp cứu", "ra viện"), tiêu đề mục, mô tả thuần mức độ/thời gian
    ("Mức độ: nghiêm trọng", "Thời gian: 5 phút").
    NHƯNG thăm dò CHẨN ĐOÁN thì CÓ gán: ⟦TÊN_XÉT_NGHIỆM⟧sinh thiết⟦/⟧, ⟦TÊN_XÉT_NGHIỆM⟧nội soi⟦/⟧.

CẤU TRÚC 3 phần (không cần đủ mọi mục): 1. Tiền sử bệnh → thường isHistorical;
2. Bệnh sử hiện tại → hiện tại; 3. Đánh giá tại bệnh viện.
VĂN PHONG: viết tắt po/bid/iv/prn, thuốc Latin, dấu phẩy thập phân kiểu Việt.

════════ CHỈ TIÊU ĐỊNH LƯỢNG — note không đạt sẽ bị LOẠI BẰNG MÁY ════════
Các con số này đo từ 100 bệnh án THẬT. Đây không phải gợi ý, là ràng buộc.

MẬT ĐỘ: ~23 khái niệm / 1000 ký tự. Note ~1300 ký tự thì phải có ~30 thẻ.
  Gán THƯA là lỗi phổ biến nhất và làm hỏng điểm trực tiếp. Bệnh án lặp lại rất
  nhiều (mục "Diễn biến", "Tóm tắt" nhắc lại cùng một thứ) — BỌC THẺ TẤT CẢ CÁC LẦN.

TỈ LỆ 5 LOẠI (bám sát, đây là chỗ các mẻ trước sai nặng nhất):
    TRIỆU_CHỨNG          40%   ← ĐÔNG NHẤT. Mẻ trước chỉ đạt 26% → THIẾU HẲN MỘT NỬA.
    CHẨN_ĐOÁN            18%
    TÊN_XÉT_NGHIỆM       17%
    KẾT_QUẢ_XÉT_NGHIỆM   16%
    THUỐC                 9%   ← mẻ trước phình lên 14%. Đừng liệt kê quá nhiều thuốc.
  Cứ 10 thẻ thì phải có KHOẢNG 4 thẻ TRIỆU_CHỨNG. Hãy viết nhiều triệu chứng hơn
  bản năng mách bảo: mô tả cả dấu hiệu khám (phù 2 chân, da khô, lơ mơ), cả biểu
  hiện tâm thần (lo âu, ảo giác), và nhắc lại chúng ở mục tóm tắt.

TỈ LỆ ASSERTION (theo từng loại):
    TRIỆU_CHỨNG : 80% rỗng, 16% isNegated,  4% isHistorical
    CHẨN_ĐOÁN   : 36% rỗng,  6% isNegated, 59% isHistorical
    THUỐC       : 56% rỗng,  0% isNegated, 43% isHistorical
  isFamily CỰC HIẾM (4/2642 ≈ 0.15%) — chỉ sinh khi được yêu cầu rõ.

ĐẶC TÍNH "BẨN" của dữ liệu thật (model SẼ gặp ở test, phải tái tạo):
    ~9%  note có DÍNH CHỮ mất dấu cách: "atenololtrong ngày", "Theo dõiLoét bàn chân",
         "cảm giáckhó chịu vùng ngực", "đau ngực tráikèm theo khó thở".
         Phần lớn dính chữ là THƯỜNG-với-THƯỜNG, không phải hoa/thường.
    ~5%  note TRỘN ANH–VIỆT: "nausea", "diarrhea", "abdominal pain", "intravenous fluids",
         "hr 88 bp 130/80 rr 14 spo2 100ra".
    ~13% note có VIẾT TẮT LÂM SÀNG: LLQ, RLQ, NRB, PPM, PCP, FNA, DVT, ERCP.
    ~3%  note có PLACEHOLDER: "[Date]", "[Name]", "[Số]", "ngày DD MM".
         ⚠️ CHỈ dùng placeholder khi được YÊU CẦU RÕ ở phần "BẮT BUỘC chứa".
         Mặc định hãy viết ngày tháng và tên CỤ THỂ ("ngày 12/03", "ông Nam").
         ĐỪNG tự ẩn danh bằng [Name]/[Date] — dữ liệu thật hiếm khi làm vậy (3%),
         còn model sinh thì lạm dụng (đo được 17%).
    Thỉnh thoảng LẶP ĐÔI TÊN BỆNH: "Tiểu đường loại 1 đái tháo đường",
    "đái tháo đườngđái tháo đường".

CHỈ IN RA bệnh án đã đánh thẻ. KHÔNG giải thích, KHÔNG thêm chữ ngoài bệnh án.
Trước khi in kết quả, tự rà lại TỪNG thẻ assertion đã dùng đúng dấu `|` chưa — nếu thấy dấu cách
hoặc dấu `:` bên trong `⟦...⟧` thì đó là LỖI, phải sửa lại theo đúng cú pháp trên."""

GEN_USER_TEMPLATE = """Sinh 1 bệnh án theo yêu cầu:
- KIỂU BỆNH ÁN: {archetype}
- Độ dài: {length} — khoảng {chars} ký tự ({length_desc})
- Chuyên khoa / lý do nhập viện: {specialty}
- BẮT BUỘC chứa các tình huống: {cases}
- {noise}
- Đừng lặp lại bệnh/thuốc của ví dụ mẫu; đổi tên bệnh, thuốc, con số.
- {placeholder_rule}

════ ĐỘ DÀI — máy sẽ ĐẾM KÝ TỰ ════
Bệnh án phải dài KHOẢNG {chars} ký tự. Đây là ràng buộc CỨNG, không phải gợi ý.
Muốn đủ dài thì phải viết VĂN XUÔI, không được liệt kê khô khan toàn khái niệm.
Bệnh án thật có RẤT NHIỀU chữ KHÔNG mang thẻ nào — hãy viết đủ chúng:
  · tiêu đề mục ("2. Bệnh sử hiện tại", "Dấu hiệu sinh tồn")
  · sự kiện hành chính ("bệnh nhân được đưa vào Khoa Cấp cứu bằng xe cấp cứu",
    "được hội chẩn chuyên khoa", "nhập viện theo dõi", "hẹn tái khám sau 2 tuần")
  · mô tả mức độ / thời gian / diễn biến ("Mức độ: nghiêm trọng", "kéo dài khoảng
    3 ngày, tăng dần về đêm", "Thời gian: 5 phút", "Vị trí: bên trái")
  · câu nối, câu dẫn, trạng thái chung ("Bệnh nhân tỉnh, tiếp xúc được")
Tất cả những thứ trên ĐỀU KHÔNG GÁN THẺ, nhưng chúng chính là phần làm bệnh án
GIỐNG THẬT. Note chỉ toàn thẻ san sát nhau là note GIẢ và sẽ bị loại.

════ NGÂN SÁCH THẺ CHO NOTE NÀY — máy sẽ ĐẾM, lệch nhiều là bị loại ════
Tổng khoảng {n_tags} thẻ (≈23 thẻ/1000 ký tự). Chia ra XẤP XỈ:
    TRIỆU_CHỨNG          ~{n_symptom} thẻ   ← nhiều nhất, đừng gán thiếu
    CHẨN_ĐOÁN            ~{n_diag} thẻ
    TÊN_XÉT_NGHIỆM       ~{n_test} thẻ
    KẾT_QUẢ_XÉT_NGHIỆM   ~{n_result} thẻ
    THUỐC                ~{n_drug} thẻ
TRƯỚC KHI IN: ĐẾM LẠI thẻ ⟦TRIỆU_CHỨNG⟧. Phải có ĐỦ ~{n_symptom} thẻ — đây là loại
ĐÔNG NHẤT (40% dữ liệu thật) và là chỗ model sinh LUÔN LUÔN gán thiếu (đo được:
34% thay vì 40%). Nếu chưa đủ, bổ sung bằng ba cách sau, theo thứ tự:
  1. DẤU HIỆU KHÁM (rất hay bị quên): ⟦TRIỆU_CHỨNG⟧phù 2 chân⟦/⟧,
     ⟦TRIỆU_CHỨNG⟧da khô, nếp véo da mất chậm⟦/⟧, ⟦TRIỆU_CHỨNG⟧lơ mơ⟦/⟧,
     ⟦TRIỆU_CHỨNG⟧loét 2 bàn chân⟦/⟧, ⟦TRIỆU_CHỨNG⟧chảy dịch vàng⟦/⟧
  2. NHẮC LẠI triệu chứng đã nêu, ở mục "Diễn biến" và "Tóm tắt" — bệnh án thật
     lặp rất nhiều, và MỖI LẦN XUẤT HIỆN LÀ MỘT THẺ RIÊNG.
  3. TRIỆU CHỨNG ĐI KÈM và TRIỆU CHỨNG BỊ PHỦ ĐỊNH ("Bệnh nhân phủ nhận
     ⟦TRIỆU_CHỨNG|isNegated⟧sốt⟦/⟧, ⟦TRIỆU_CHỨNG|isNegated⟧ớn lạnh⟦/⟧").

Nhắc lại cú pháp thẻ (BẮT BUỘC): ⟦TYPE⟧surface⟦/⟧ hoặc ⟦TYPE|assertion⟧surface⟦/⟧ hoặc
⟦TYPE|assertion1,assertion2⟧surface⟦/⟧ — assertion nằm sau dấu `|`, cách nhau bằng dấu `,`,
KHÔNG dấu cách, KHÔNG dấu `:`, KHÔNG chữ true/false. Ví dụ: ⟦TRIỆU_CHỨNG|isNegated⟧sốt⟦/⟧.
Thẻ KHÔNG được lồng nhau / chồng lấn: mỗi ký tự thuộc đúng một thẻ.

Chỉ in ra bệnh án đã đánh thẻ."""

# Độ dài thật: trung vị 1229, TB 1323, max 4428 ký tự (100 file test).
LENGTH_DESC = {"SHORT": "10–16 dòng", "LONG": "30–45 dòng"}
# Thật: trung vị 1229, TB 1323 ký tự.
# Hiệu chỉnh bằng đo, không đoán — Gemini không tuân mốc ký tự một cách tuyến tính:
#   mốc (800, 2200) + prompt CŨ            -> TB thật  844  (viết NGẮN hơn mốc)
#   mốc (1200, 2600) + khối "viết văn xuôi" -> TB thật 1980  (viết DÀI hơn mốc ~28%)
# Khối văn xuôi (thêm ở bản sau) làm nó nở ra hẳn. Với mix slot 75% SHORT / 25% LONG,
# mốc (850, 1800) cho kỳ vọng thô ~1088, nhân hệ số nở ~1.28 -> ~1390, sát 1323.
LENGTH_CHARS = {"SHORT": 850, "LONG": 1800}

# Tỉ lệ đo trên 2642 khái niệm thật — dùng để chia ngân sách thẻ mỗi note.
TYPE_MIX = [("n_symptom", 0.40), ("n_diag", 0.18), ("n_test", 0.17),
            ("n_result", 0.16), ("n_drug", 0.09)]

# ARCHETYPE — vì sao phải có:
# Ép MỌI note đạt đúng 40% TRIỆU_CHỨNG là đánh nhau với generator và luôn thua:
# 5 mẻ pilot cho ra 31–38%, không mẻ nào chạm 40. Cứ siết triệu chứng thì nó lại
# đẩy xét nghiệm lên (pilot cuối: TÊN_XÉT_NGHIỆM 23.2% trong khi mục tiêu 17%).
#
# Bệnh án THẬT vốn không đồng nhất: có bệnh án thuần bệnh sử (gần như không có
# cận lâm sàng), có bệnh án thuần workup (đầy xét nghiệm). Mục tiêu 40% là của
# CẢ TẬP, không phải của từng note. Nên hãy phân tầng: trộn các archetype sao cho
# TRUNG BÌNH GIA QUYỀN rơi đúng 40/18/17/16/9, thay vì bắt mỗi note tự cân.
#
#   (trọng số, tên, tỉ lệ 5 type)          symptom diag test result drug
ARCHETYPES = [
    (0.40, "BỆNH SỬ — chủ yếu triệu chứng và diễn biến. RẤT ÍT cận lâm sàng: "
           "tối đa 1–2 xét nghiệm. Tập trung mô tả triệu chứng, dấu hiệu khám, "
           "triệu chứng đi kèm, triệu chứng bị phủ định, và NHẮC LẠI chúng ở mục "
           "diễn biến/tóm tắt.",
     (0.58, 0.17, 0.07, 0.06, 0.12)),
    (0.40, "CÂN BẰNG — bệnh án đầy đủ ba phần: tiền sử, bệnh sử hiện tại, đánh "
           "giá tại bệnh viện (có sinh hiệu, xét nghiệm, chẩn đoán hình ảnh).",
     (0.36, 0.19, 0.19, 0.19, 0.07)),
    (0.20, "CẬN LÂM SÀNG — trọng tâm là đánh giá tại bệnh viện: nhiều xét nghiệm, "
           "sinh hiệu có số, kết quả chẩn đoán hình ảnh, mô tả phát hiện dài.",
     (0.22, 0.17, 0.29, 0.26, 0.06)),
]
# Kiểm tra trung bình gia quyền:
#   symptom .40*.58 + .40*.36 + .20*.22 = .232+.144+.044 = 0.420
#   test    .40*.07 + .40*.19 + .20*.29 = .028+.076+.058 = 0.162
#   result  .40*.06 + .40*.19 + .20*.26 = .024+.076+.052 = 0.152
# -> xấp xỉ 40/18/17/16/9 như mục tiêu. Symptom xin dư một chút (42%) vì generator
#    LUÔN gán thiếu loại này (đo được: xin 40, ra 31–38).

NOISE_ON = "Chèn nhẹ 1 lỗi gõ/dính chữ thật (chỉ bọc thẻ đúng phần khái niệm)."
NOISE_OFF = "Văn bản sạch, không cố tình tạo lỗi gõ."

PLACEHOLDER_ON = ("Chèn 1 placeholder: [Date] hoặc [Name] hoặc 'ngày DD MM'.")
PLACEHOLDER_OFF = (
    "TUYỆT ĐỐI KHÔNG ẩn danh bằng placeholder. KHÔNG viết [Name], [Date], [Ngày], "
    "[Tên], [Số], hay 'DD MM'. Hãy viết tên và ngày CỤ THỂ ('ông Nam', 'ngày 12/03')."
)

# Đặc tính "bẩn" — bốc ĐỘC LẬP theo xác suất riêng (slots.dirty_probs trong config),
# KHÔNG nhét vào required_cases. Ở required_cases mỗi note bốc 2–4 case trong ~14,
# nên mọi thứ nằm đó đều bật lên ở ~21% số note — đo được ở pilot: trộn Anh–Việt
# vọt lên 20% trong khi dữ liệu thật chỉ 5%.
DIRTY_CASES = {
    "english": ("TRỘN ANH–VIỆT: dùng vài từ tiếng Anh nguyên bản giữa câu tiếng Việt "
                "(nausea, diarrhea, abdominal pain, intravenous fluids)"),
    # Gemini phớt lờ yêu cầu này khi chỉ liệt kê chữ viết tắt: nó đang viết tiếng
    # Việt nên tự dịch "LLQ" thành "hạ sườn trái". Đo được 0% ở hai pilot liên
    # tiếp dù prompt yêu cầu ở 14% số lượt. Phải đưa CÂU MẪU trong ngữ cảnh.
    "abbrev": ("VIẾT TẮT LÂM SÀNG — BẮT BUỘC giữ NGUYÊN dạng viết tắt tiếng Anh, "
               "KHÔNG dịch sang tiếng Việt. Dùng 1–2 trong: LLQ, RLQ, NRB, PPM, PCP, "
               "FNA, DVT, ERCP. Viết y như bác sĩ ghi tay, ví dụ: "
               "'Đau khu trú ⟦TRIỆU_CHỨNG⟧LLQ⟦/⟧, đã loại trừ ⟦CHẨN_ĐOÁN|isNegated⟧DVT⟦/⟧' "
               "hoặc 'Chỉ định ⟦TÊN_XÉT_NGHIỆM⟧ERCP⟦/⟧ để đánh giá đường mật'"),
    "placeholder": "PLACEHOLDER: chèn [Date] hoặc [Name] hoặc 'ngày DD MM'",
    "dup_disease": ("LẶP ĐÔI TÊN BỆNH ngay tại chỗ: 'Tiểu đường loại 1 đái tháo đường' "
                    "— bọc thẻ như một khái niệm"),
    "family": ("một ca isFamily: tình trạng THUỘC VỀ người thân (mẹ/cháu gái…), "
               "KHÔNG phải người thân kể về bệnh nhân"),
    # 'glue' không có câu mô tả riêng; nó bật NOISE_ON ở dòng '- {noise}'.
}

JUDGE_SYSTEM = """Bạn là người KIỂM DUYỆT nhãn cho dữ liệu NER y khoa tiếng Việt. Bạn nhận một
bệnh án đã đánh thẻ ⟦LOẠI|assertion⟧cụm⟦/⟧ và chấm xem nhãn CÓ ĐÚNG guideline không.
KHÔNG viết lại bệnh án; chỉ đánh giá.

Kiểm tra:
1. TYPE đúng (1 trong TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC).
2. TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM PHẢI là 2 thẻ tách biệt (không gộp tên+giá trị).
3. Assertion đúng:
   - isNegated chỉ khi thật sự bị phủ định; "không đặc hiệu/xác định/do.../Hodgkin/cản quang" KHÔNG phải phủ định.
   - "không có gì đáng chú ý/không ghi nhận gì bất thường" sau tên XN là KẾT_QUẢ_XÉT_NGHIỆM, KHÔNG isNegated.
   - isFamily chỉ khi thuộc người nhà; loại "người nhà kể về bệnh nhân" và "bác sĩ gia đình".
   - isHistorical theo section tiền sử/thuốc trước nhập viện hoặc cue "tiền sử/đã từng"; không gán cho triệu chứng hiện tại.
   - Assertion KHÔNG gắn cho TÊN_XÉT_NGHIỆM / KẾT_QUẢ_XÉT_NGHIỆM.
4. Recall: khái niệm y tế rõ ràng nào bị BỎ SÓT (không bọc thẻ).
5. Precision: thẻ nào bọc nhầm chữ KHÔNG phải khái niệm.

verdict = "ACCEPT" nếu không có lỗi loại 1/2/3 và ≤1 lỗi nhỏ recall/precision; ngược lại "REJECT".

CHỈ trả về JSON:
{"verdict":"ACCEPT"|"REJECT","errors":["..."],"n_type_errors":<int>,"n_split_errors":<int>,"n_assertion_errors":<int>}"""

JUDGE_USER_TEMPLATE = """Chấm bệnh án đã đánh thẻ sau theo đúng tiêu chí. Chỉ trả về JSON.

<<<
{marked_note}
>>>"""


def build_gen_user(slots: dict, rng: random.Random) -> str:
    length = rng.choice(slots.get("length", ["SHORT", "LONG"]))
    specialty = rng.choice(slots.get("specialty", ["nội khoa tổng quát"]))
    all_cases = slots.get("required_cases", [])
    k = min(len(all_cases), rng.randint(2, 4)) if all_cases else 0
    cases = "; ".join(rng.sample(all_cases, k)) if k else "tự chọn"
    # Đặc tính "bẩn" phải bốc bằng XÁC SUẤT RIÊNG, không được nhét chung vào
    # required_cases: ở đó mỗi note bốc 2–4 case trong ~14 case, nên bất cứ thứ gì
    # nằm trong danh sách đó đều xuất hiện ở ~21% số note. Đo được ở mẻ pilot:
    # trộn Anh–Việt vọt lên 20% trong khi thật chỉ 5%.
    dirty = [d for d, p in (slots.get("dirty_probs") or {}).items() if rng.random() < float(p)]
    noise = NOISE_ON if "glue" in dirty else NOISE_OFF
    extra = [DIRTY_CASES[d] for d in dirty if d in DIRTY_CASES]
    if extra:
        cases = cases + "; " + "; ".join(extra)

    # Cấm ẩn danh phải nằm trong prompt NGƯỜI DÙNG. Đặt ở GEN_SYSTEM thì bị chìm:
    # Gemini vẫn tự chèn [Name]/[Date] ở 15.8% số note (thật chỉ 3%).
    placeholder_rule = (PLACEHOLDER_ON if "placeholder" in dirty else PLACEHOLDER_OFF)

    # Ngân sách thẻ tính từ độ dài -> LLM nhận một con số cụ thể phải đạt, thay vì
    # một tỉ lệ phần trăm trừu tượng. Các mẻ trước lệch nặng (TRIỆU_CHỨNG 26% thay
    # vì 40%) chính vì prompt chỉ nói tỉ lệ mà không nói "note này cần ~12 thẻ".
    chars = int(LENGTH_CHARS.get(length, 1300) * rng.uniform(0.85, 1.15))
    # Ngân sách thẻ = ĐÚNG mật độ mục tiêu (23/1000). Đừng hạ nó xuống để ghì mật
    # độ: đã thử hạ còn 20 và mật độ lại TĂNG (26.25 -> 26.93). Lý do: Gemini co
    # độ dài văn bản theo số thẻ, nên ít thẻ -> văn bản ngắn hơn -> mật độ y
    # nguyên. Đòn bẩy thật là bắt nó viết nhiều VĂN XUÔI KHÔNG GÁN THẺ (xem
    # GEN_SYSTEM: sự kiện hành chính, tiêu đề mục, mô tả mức độ đều không gán).
    n_tags = max(6, round(chars / 1000 * 23))

    # Bốc archetype theo trọng số -> ngân sách type LẤY TỪ ARCHETYPE, không phải
    # từ TYPE_MIX chung. Cả TẬP mới bám 40/18/17/16/9; từng note thì không.
    r = rng.random()
    acc = 0.0
    arch_desc, mix = ARCHETYPES[-1][1], ARCHETYPES[-1][2]
    for w, desc, m in ARCHETYPES:
        acc += w
        if r < acc:
            arch_desc, mix = desc, m
            break
    keys = [k_ for k_, _ in TYPE_MIX]
    budget = {k_: max(1, round(n_tags * p)) for k_, p in zip(keys, mix)}

    return GEN_USER_TEMPLATE.format(
        length=length, length_desc=LENGTH_DESC.get(length, ""), chars=chars,
        specialty=specialty, cases=cases, noise=noise, archetype=arch_desc,
        placeholder_rule=placeholder_rule, n_tags=n_tags, **budget)


def build_fewshot(examples: List[str]) -> str:
    if not examples:
        return ""
    blocks = "\n\n".join(f"[VÍ DỤ {i+1}]\n{ex}" for i, ex in enumerate(examples))
    return ("Dưới đây là ví dụ về định dạng thẻ mong muốn:\n\n" + blocks +
            "\n\n---\nGiờ hãy sinh một bệnh án MỚI (khác nội dung ví dụ):\n\n")


def build_judge_user(marked_note: str) -> str:
    return JUDGE_USER_TEMPLATE.format(marked_note=marked_note)


def tally(verdicts: List[str], threshold: int) -> bool:
    """True if #ACCEPT >= threshold."""
    return sum(1 for v in verdicts if str(v).upper() == "ACCEPT") >= threshold
