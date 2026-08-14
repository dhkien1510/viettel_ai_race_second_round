"""Genre-specific prompt supplements for the LLM backend.

Rationale (see `doc/annotation-guideline.md`): a small zero-shot model
(1.5B-4B) struggles to reliably apply the full set of
nuanced, sometimes-conflicting exclusion rules simultaneously. Splitting the
rule set by genre means each single call only needs to hold the handful of
rules that actually matter for THAT document's structure — the system prompt
in `llm.py` stays the stable, always-true core (5 types, span boundaries,
output format); this module supplies the SHORT, genre-targeted addendum.

Genre keys match `src.pipeline.genre_router.GenreType`.
"""

from __future__ import annotations

# Q&A_FULL: patient asks, doctor answers, no clinical record scaffolding.
_QA_FULL = """
LƯU Ý RIÊNG CHO VĂN BẢN HỎI-ĐÁP (bệnh nhân hỏi, bác sĩ trả lời):
- Bệnh nhân TỰ HỎI về chẩn đoán của CHÍNH MÌNH ("em có phải bị X không ạ?") mà
  không có nhận định y khoa nào xác nhận → KHÔNG gán X. NHƯNG nếu bác sĩ trả
  lời bằng ngôn từ dè dặt ("khả năng em bị X") → VẪN gán X — đây là nhận định
  chuyên môn, không phải câu hỏi.
- Thuốc/xét nghiệm mà bác sĩ nói SẼ quyết định ở tương lai ("sẽ so sánh nguy
  cơ, lợi ích rồi quyết định tiếp tục hay ngưng") → KHÔNG gán, khác với thuốc
  đã kê/đang dùng thật (luôn gán).
- Nếu 1 đoạn mô tả CÙNG 1 sự kiện cấp tính bằng nhiều câu chi tiết hoá liên
  tục (không phải các lần xuất hiện tách biệt) → chỉ giữ 1 câu cốt lõi, không
  tag riêng từng câu bổ sung.
"""

# HOSPITAL: structured clinical record, section headers, repeated mentions.
_HOSPITAL = """
LƯU Ý RIÊNG CHO BỆNH ÁN CÓ CẤU TRÚC:
- Câu định nghĩa "X là tình trạng Y" (mô tả cơ chế/sinh lý bệnh) → giữ tên
  bệnh X, bỏ phần giải thích Y — NHƯNG xét TỪNG occurrence: nếu cùng cụm đó
  xuất hiện lại ở chỗ khác như 1 finding thật của ca bệnh thì vẫn giữ ở đó.
- Gán TẤT CẢ các lần lặp lại hợp lệ của 1 khái niệm trong văn bản (mỗi vị trí
  1 object riêng) — đây là nguồn recall lớn nhất, đừng chỉ gán lần đầu rồi bỏ.
- Dấu hiệu sinh tồn (mạch, huyết áp, nhịp thở, spo2, nhiệt độ) kèm SỐ ĐO →
  TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM (không phải TRIỆU_CHỨNG).
"""

# FAQ: pure educational article, no dialogue, no patient case.
_FAQ = """
LƯU Ý RIÊNG CHO BÀI GIÁO KHOA/FAQ (không có hội thoại bệnh nhân-bác sĩ):
- Tên bệnh chủ đề của bài (kể cả generic, kể cả nhắc trong tiêu đề/định nghĩa)
  VẪN được gán CHẨN_ĐOÁN — đừng bỏ chỉ vì đây là bài giáo khoa.
- NHƯNG câu định nghĩa/giải thích cơ chế đi kèm ("X là tình trạng Y do...") →
  bỏ phần Y (giải thích), chỉ giữ tên bệnh X.
- Danh sách triệu chứng/tác dụng phụ liệt kê CHUNG cho cả loại bệnh (không
  gắn 1 ca bệnh cụ thể nào) → thường KHÔNG gán, trừ khi là tên bệnh cơ quan cụ
  thể trong danh sách biến chứng (tên bệnh cơ quan vẫn gán, chỉ bỏ từ chỉ kết
  cục/di chứng thuần như "tử vong", "bại não").
"""

# HYBRID_QA / HYBRID_CONSULT: EHR-like content precedes or mixes with a
# consult/dialogue section — combine both rule sets, plus the priority rule.
_HYBRID = """
LƯU Ý RIÊNG CHO VĂN BẢN LAI (bệnh án + phần hỏi-đáp/tư vấn):
- Nếu 1 sự kiện/triệu chứng được nhắc CẢ trong đoạn kể lể tự do (hỏi-đáp) LẪN
  trong 1 mục có tiêu đề rõ ràng (vd "Lý do nhập viện:", "Tiền sử bệnh") → ưu
  tiên giữ bản trong mục có tiêu đề, cân nhắc bỏ bản kể lể tự do trùng lặp.
- Áp dụng CẢ 2 bộ lưu ý: phần hội thoại xử lý như văn bản hỏi-đáp (bệnh nhân
  tự hỏi vs bác sĩ nhận định), phần có tiêu đề xử lý như bệnh án có cấu trúc
  (gán mọi lần lặp, cơ chế/định nghĩa theo occurrence).
"""

GENRE_HINTS: dict[str, str] = {
    "Q&A_FULL": _QA_FULL,
    "HOSPITAL": _HOSPITAL,
    "FAQ": _FAQ,
    "HYBRID_QA": _HYBRID,
    "HYBRID_CONSULT": _HYBRID,
}


def genre_hint(genre: str | None) -> str:
    """Return the prompt addendum for a genre, or "" if unknown/None (caller
    just gets the base instruction with no genre-specific supplement)."""
    if not genre:
        return ""
    return GENRE_HINTS.get(genre, "")
