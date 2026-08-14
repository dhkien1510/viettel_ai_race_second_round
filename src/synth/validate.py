"""Cổng kiểm duyệt BẰNG CODE cho note synth — chạy TRƯỚC mọi judge LLM.

Vì sao phải có, và phải chạy trước: 378 note synth đang nằm trong data/synth/ đã
qua judge LLM và qua cả một vòng refine bằng Claude Opus, vậy mà vẫn lệch:

    mật độ 31.13 khái niệm/1000 ký tự   (thật: 23)
    TRIỆU_CHỨNG 22.5%                   (thật: 40%)
    TÊN_XN + KQ_XN 45.7%                (thật: 33%)
    dính chữ 68% số note                (thật: 14%)
    trộn tiếng Anh 0%                   (thật: 5%)
    placeholder 0%                      (thật: 3%)

Không một lỗi nào ở trên là chuyện "ý kiến" — chúng đều là phép đếm. LLM judge
không bắt được vì nó đọc từng note một và không giữ thống kê toàn mẻ; hỏi nó
"note này có đúng phân bố không" là hỏi sai đối tượng. Code đếm được, miễn phí,
đúng 100%. Chỉ những gì code KHÔNG trả lời được mới nên đem hỏi LLM (ví dụ: nhãn
này có đúng lâm sàng không).

Hai tầng:
  · check_note()  -> HARD, loại thẳng 1 note (bất biến §7 guideline)
  · check_batch() -> SOFT, đo cả mẻ so với phân bố thật; lệch thì phải sinh bù /
                     chỉnh prompt, KHÔNG loại từng note một cách máy móc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..schema import (
    VALID_TYPES, VALID_ASSERTIONS, ASSERTABLE_TYPES, LINKABLE_TYPES, Entity,
)
from .markup import parse_markup, sanitize_markup

# --- mục tiêu, đo từ 100 file test thật (xem doc/annotation_guideline_v2.md §1, §8) ---
# Mục tiêu ở mức MẺ: 23 — bộ 20 file tinh chỉnh nhất đạt 23.1 (80 file tự gán chỉ
# 18.3, và chính chúng có WER 43.9 so với 33.7). Gán thưa làm WER xấu trực tiếp.
TARGET_DENSITY = 23.0                 # khái niệm / 1000 ký tự

# Dải HARD gate cho TỪNG note — đo từ 100 file thật, KHÔNG phải đoán.
# Phân bố thật: min 2.6 | p05 9.1 | trung vị 19.0 | p95 29.9 | max 32.7
# Dải [15, 30] ban đầu tôi tự đặt loại mất 31/100 FILE THẬT — một cổng đánh trượt
# được chính dữ liệu thật là cổng sai. [8, 35] chỉ loại 4/100, đúng vai trò lọc
# ngoại lai. Mật độ TỪNG note dao động rất rộng; chỉ tổng cả mẻ mới cần bám 23.
DENSITY_BAND = (8.0, 35.0)

TARGET_TYPE_MIX: Dict[str, float] = {
    "TRIỆU_CHỨNG": 0.40,
    "CHẨN_ĐOÁN": 0.18,
    "TÊN_XÉT_NGHIỆM": 0.17,
    "KẾT_QUẢ_XÉT_NGHIỆM": 0.16,
    "THUỐC": 0.09,
}
TYPE_MIX_TOL = 0.05                   # lệch quá 5 điểm % ở mức MẺ là phải sửa prompt

# tỉ lệ note THẬT mang từng đặc tính "bẩn" — synth phải bám theo, không nhiều hơn
TARGET_DIRTY: Dict[str, float] = {
    "glue_case": 0.09,     # đo được trên 100 file test thật (xem cảnh báo dưới)
    "english": 0.05,       # trộn Anh–Việt
    "abbrev": 0.12,        # viết tắt lâm sàng (danh sách guideline §8, xem dưới)
    "placeholder": 0.03,   # [Date] / [Name] / ngày DD MM
}
DIRTY_TOL = 0.10

TARGET_LEN = (1323, 400)              # (trung bình, độ lệch chấp nhận được) ký tự

_VN_LOWER = "a-zàáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
_VN_UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ"

# ⚠️ `glue_case` chỉ bắt được dính chữ CÓ ranh giới hoa/thường ("Theo dõiLoét").
# Dính chữ THẬT trong đề bài phần lớn là thường-với-thường và bộ dò này KHÔNG
# thấy: "atenololtrong", "cảm giáckhó chịu", "đau ngực tráikèm theo",
# "Dùngmethadonekéo dài". Bắt cho đủ thì cần tách từ tiếng Việt (src/wordseg.py
# + vncorenlp) — chưa làm. Nên đây là tín hiệu THAM KHẢO, cố tình để ở mức cảnh
# báo, KHÔNG dùng để đánh trượt cả mẻ.
#
# Đơn vị y khoa viết đúng chuẩn (mmHg, SpO2, PaCO2, ng/mL, pH, aVL) cũng có ranh
# giới hoa/thường — phải che đi trước, nếu không 100% note đều bị báo "dính chữ".
RE_UNIT_MASK = re.compile(
    r"\b(mmHg|mmhg|SpO2|SaO2|SvO2|PaO2|PaCO2|FiO2|HbA1c|pH|aVL|aVR|aVF|mEq|mmol|"
    r"mg/dL|ng/mL|mcg/dL|g/dL|mL|dL|IU|BiPAP|CPAP|McBurney|X-quang)\b")
_RE_GLUE_CASE = re.compile(f"[{_VN_LOWER}][{_VN_UPPER}]")


def has_glue_case(text: str) -> bool:
    """Dính chữ có ranh giới hoa/thường, sau khi che các đơn vị y khoa hợp lệ."""
    return bool(_RE_GLUE_CASE.search(RE_UNIT_MASK.sub(" ", text)))


RE_ENGLISH = re.compile(
    r"\b(nausea|diarrhea|abdominal pain|chest pain|shortness of breath|fever|"
    r"vomiting|intravenous fluids?|dizziness|headache)\b", re.I)
# ĐÚNG danh sách viết tắt "bẩn" ở guideline §8. Bản đầu tôi nhét thêm CT/MRI/ECG/
# CBC/COPD — đó là từ y khoa PHỔ THÔNG, không phải nhiễu; chúng kéo tỉ lệ đo được
# trên chính dữ liệu thật từ 12% vọt lên 21% và làm mục tiêu sai hẳn.
RE_ABBREV = re.compile(r"\b(LLQ|RLQ|NRB|PPM|PCP|FNA|DVT|ERCP)\b")
RE_PLACEHOLDER = re.compile(r"\[(Date|Name|Số|Tên|Ngày)\]|\bDD MM\b")


# --------------------------------------------------------------------------- #
# SỬA NHÃN — hai lỗi mà LLM judge KHÔNG bắt được (đã đo: tỉ lệ lỗi trong nhóm
# Qwen ACCEPT là 10.7%, trong nhóm Qwen REJECT là 9.4% — judge hoàn toàn MÙ).
# Phải sửa bằng code.
# --------------------------------------------------------------------------- #

# Động từ mở đầu một THỦ THUẬT ĐIỀU TRỊ. Guideline §6: KHÔNG GÁN.
# `(\s+bỏ)?` bắt buộc phải có: không thì "cắt bỏ u xơ tử cung" chỉ bị bóc chữ "cắt",
# phần dư thành "bỏ u xơ tử cung" — không khớp RE_DISEASE_HEAD nên bị BỎ OAN, mất
# mất khái niệm "u xơ tử cung" mà §2.2 bảo phải giữ.
RE_PROC = re.compile(
    r"^(đã\s+|ca\s+mổ\s+|mổ\s+)?(phẫu\s*thuật\s+)?"
    r"(cắt|thay\s+khớp|tán\s+sỏi|đặt\s+(stent|shunt)|ghép|nối|khâu|phẫu\s*thuật)"
    r"(\s+bỏ)?\b",
    re.I)

# Sau khi bỏ động từ thủ thuật, nếu phần còn lại BẮT ĐẦU bằng một trong các từ này
# thì đó là BỆNH LÝ — LÝ DO MỔ, và guideline §2.2 nói nó PHẢI được giữ làm
# CHẨN_ĐOÁN + isHistorical ("phẫu thuật cắt bỏ tuyến tiền liệt (u ác của tuyến
# tiền liệt)"). Không có từ nào khớp -> chỉ là thủ thuật thuần -> BỎ HẲN.
RE_DISEASE_HEAD = re.compile(
    r"^(u\b|ung\s*thư|sỏi|viêm|polyp|nang|áp\s*xe|thoát\s*vị|khối)", re.I)

# "cắt bỏ túi mật DO sỏi túi mật" -> phần sau chữ "do" chính là bệnh lý lý do mổ.
RE_REASON = re.compile(r"\bdo\s+(.+)$", re.I)

# Giá trị đo CÓ ĐƠN VỊ nằm trong span TRIỆU_CHỨNG.
RE_MEASURED = re.compile(
    r"\s*\d+([.,]\d+)?\s*(°C|độ\s*C|mmHg|%|/phút|nhịp/phút|lần/phút|mmol/l|mg/dl)",
    re.I)

# Chữ dẫn "tiền sử" bị NUỐT vào span. §4.1: cue đứng NGOÀI thẻ, thẻ chỉ chứa lõi
# lâm sàng. 'tiền sử tăng huyết áp' -> 'tăng huyết áp'.
RE_HIST_CUE = re.compile(r"^tiền\s+sử\s+(gia\s+đình\s+(có\s+)?)?", re.I)

# Tiêu đề mục bị gán thẻ (§6: KHÔNG GÁN).
RE_SECTION_HDR = re.compile(
    r"^(dấu\s+hiệu\s+sinh\s+tồn|triệu\s+chứng\s+hiện\s+tại|kết\s+quả\s+xét\s+nghiệm|"
    r"bệnh\s+sử\s+hiện\s+tại|tiền\s+sử\s+(bệnh|phẫu\s*thuật|gia\s+đình)|"
    r"khám\s+lâm\s+sàng|điều\s+trị(\s+theo\s+đợt)?)$", re.I)

# Sự kiện hành chính (§6: KHÔNG GÁN).
RE_ADMIN = re.compile(
    r"^(nhập\s+viện|ra\s+viện|xuất\s+viện|tái\s+khám|hội\s+chẩn|chuyển\s+(đến|viện)|"
    r"gọi\s+(ems|cấp\s+cứu)|đi\s+khám)", re.I)

# "Trạng thái chung chung" (§6: KHÔNG GÁN) — "Bệnh nhân tỉnh, tiếp xúc được",
# "cảm thấy khỏe", "Diễn biến ổn định". Đây KHÔNG phải dấu hiệu lâm sàng.
# ⚠️ Cố tình hẹp. Đừng mở rộng thành "mọi span có dấu phẩy": §4.3 phân biệt
# TÁCH DANH SÁCH ('tiểu buốt, tiểu rắt') với KHÔNG TÁCH CÂU MÔ TẢ
# ('da khô, nếp véo da mất chậm' — guideline nêu đích danh nó là TRIỆU_CHỨNG hợp
# lệ). Code không phân biệt nổi hai thứ đó; dữ liệu thật cũng có 2.2% span
# TRIỆU_CHỨNG chứa dấu phẩy. Sửa mò ở đây hại hơn lợi.
#
# ⚠️⚠️ VÀ PHẢI ĐÒI TRẠNG THÁI **BÌNH THƯỜNG**, không chỉ nhìn chữ "tỉnh".
# Bản đầu của tôi khớp `^tỉnh[,\s]` và BỎ MẤT 'tỉnh chậm, phản xạ kém' — thứ mà
# guideline §1.1 nêu ĐÍCH DANH là TRIỆU_CHỨNG hợp lệ (nó là dấu hiệu BẤT THƯỜNG).
# Ranh giới là bình thường/bất thường, không phải từ khoá:
#     'tỉnh táo, tiếp xúc tốt'   -> trạng thái BÌNH THƯỜNG  -> KHÔNG GÁN
#     'tỉnh chậm, phản xạ kém'   -> dấu hiệu BẤT THƯỜNG     -> TRIỆU_CHỨNG
RE_GENERIC_STATE = re.compile(
    r"^(bệnh\s+nhân\s+)?("
    r"tỉnh(\s+táo)?\s*,?\s*(và\s+)?tiếp\s+xúc\s+(tốt|được|rõ|bình\s+thường)|"
    r"tỉnh\s+táo$|"
    r"cảm\s+thấy\s+khỏe|diễn\s+biến\s+ổn\s+định|"
    r"tim\s+đều\s*,?\s*phổi\s+(trong|rõ|thông\s+khí\s+tốt)|"
    r"toàn\s+trạng\s+ổn|thể\s+trạng\s+(tốt|ổn))", re.I)

# DỊ ỨNG (thuốc/thức ăn/…) là TRIỆU_CHỨNG — h3+h4 (testing_theories_2.csv):
# gán CHẨN_ĐOÁN thì GIẢM điểm, gán TRIỆU_CHỨNG thì TĂNG. Generator không nhất
# quán: 20 span đúng TRIỆU_CHỨNG, 12 span sai thành CHẨN_ĐOÁN.
RE_ALLERGY = re.compile(r"^dị\s+ứng\b", re.I)

# HÀNH VI dùng chất gây nghiện là TRIỆU_CHỨNG — h1+h7: rượu/thuốc lá/cần sa phải
# gán (dù ở mục "Các yếu tố nguy cơ"), cà phê thì không.
# BẮT BUỘC bắt đầu bằng ĐỘNG TỪ HÀNH VI, nếu không sẽ nuốt nhầm BỆNH có tên chứa
# chữ "rượu": "Viêm tụy cấp do rượu" là CHẨN_ĐOÁN (bệnh/ICD), KHÔNG được đụng.
RE_SUBSTANCE_USE = re.compile(
    r"^(hút|uống|sử\s+dụng|dùng|nghiện|lạm\s+dụng)\s+"
    r".*(thuốc\s+lá|rượu|bia|cần\s+sa|ma\s+túy|chất\s+kích\s+thích)", re.I)

# TÊN sinh hiệu ĐỨNG TRẦN (không có mô tả định tính) bị gán TRIỆU_CHỨNG.
# §1.4: chúng là TÊN_XÉT_NGHIỆM. Nhưng CẨN THẬN — sinh hiệu MÔ TẢ ĐỊNH TÍNH
# ('nhịp thở nhanh', 'nhịp tim chậm', 'mạch nhanh') thì ĐÚNG là TRIỆU_CHỨNG và
# TUYỆT ĐỐI không được đụng vào: chúng chiếm 75+ ca trong mẻ này và guideline nêu
# đích danh chúng làm ví dụ. Chỉ khớp khi span là ĐÚNG cái tên, không gì thêm.
RE_VITAL_BARE = re.compile(
    r"^(spo2|sp02|mạch|huyết\s+áp|nhịp\s+thở|nhịp\s+tim|nhiệt\s+độ|hr|bp|rr|"
    r"độ\s+bão\s+hòa\s+oxy)$", re.I)


def repair_entities(clean: str, ents: List[Entity]) -> Tuple[List[Entity], Dict[str, int]]:
    """Sửa 2 lỗi nhãn hệ thống của generator. Trả về (entities đã sửa, thống kê).

    1. THỦ THUẬT ĐIỀU TRỊ bị gán CHẨN_ĐOÁN (10.7% số note):
         'Cắt ruột thừa'                    -> BỎ (thủ thuật thuần, §6)
         'cắt bỏ u xơ tử cung'              -> thu về 'u xơ tử cung'  (§2.2)
         'cắt bỏ túi mật do sỏi túi mật'    -> thu về 'sỏi túi mật'   (§2.2)

    2. TRIỆU_CHỨNG NUỐT GIÁ TRỊ ĐO (4% số note):
         'sốt nhẹ 37.8°C' -> BỎ.
       Căn cứ: §11/[GT29] chứng minh bằng số học trên lượt nộp thật rằng ground
       truth KHÔNG có khái niệm nào ở chỗ 'sốt' đi kèm số đo (chỉ 'sốt' đứng TRẦN
       mới có). Khớp với §1.4: *con số biến phép đo thành xét nghiệm*.
       ⚠️ Guideline TỰ MÂU THUẪN ở đây: bảng §4.3 lại bảo gán 'sốt'. Ta theo §11 vì
       đó là kết luận rút từ số liệu một lượt nộp thật, còn §4.3 chỉ là bảng tóm tắt.
    """
    out: List[Entity] = []
    stats = {"proc_dropped": 0, "proc_trimmed": 0, "measured_dropped": 0,
             "header_dropped": 0, "admin_dropped": 0, "hist_cue_trimmed": 0,
             "generic_dropped": 0, "substance_retyped": 0,
             "vital_retyped": 0}

    for e in ents:
        txt = e.text.strip()

        # --- tiêu đề mục / sự kiện hành chính: KHÔNG GÁN (§6) ---
        if RE_SECTION_HDR.match(txt):
            stats["header_dropped"] += 1
            continue
        if RE_ADMIN.match(txt):
            stats["admin_dropped"] += 1
            continue
        if e.type == "TRIỆU_CHỨNG" and RE_GENERIC_STATE.match(txt):
            stats["generic_dropped"] += 1
            continue

        # --- tên sinh hiệu ĐỨNG TRẦN gán nhầm TRIỆU_CHỨNG -> TÊN_XÉT_NGHIỆM (§1.4).
        # Sai type bị PHẠT KÉP nên đây là sửa đáng giá nhất trong nhóm này.
        if e.type == "TRIỆU_CHỨNG" and RE_VITAL_BARE.match(txt):
            out.append(Entity(e.text, e.start, e.end, "TÊN_XÉT_NGHIỆM",
                              assertions=[], candidates=[]))
            stats["vital_retyped"] += 1
            continue

        # --- dị ứng / hành vi dùng chất gây nghiện bị gán sai type -> TRIỆU_CHỨNG ---
        # (h1/h3/h4/h7). Chỉ đổi TYPE, giữ nguyên span+assertion.
        if e.type != "TRIỆU_CHỨNG" and (RE_ALLERGY.match(txt) or RE_SUBSTANCE_USE.match(txt)):
            out.append(Entity(e.text, e.start, e.end, "TRIỆU_CHỨNG",
                              assertions=list(e.assertions), candidates=[]))
            stats["substance_retyped"] += 1
            continue

        # --- span nuốt chữ dẫn "tiền sử" -> cắt bỏ, cue đứng NGOÀI thẻ (§4.1) ---
        m = RE_HIST_CUE.match(e.text)
        if m and e.type in ASSERTABLE_TYPES:
            keep = e.text[m.end():]
            if keep.strip():
                s = e.start + m.end()
                e = Entity(clean[s:e.end], s, e.end, e.type,
                           assertions=list(e.assertions), candidates=list(e.candidates))
                txt = e.text.strip()
                stats["hist_cue_trimmed"] += 1

        if e.type == "CHẨN_ĐOÁN" and RE_PROC.match(txt):
            m = RE_REASON.search(txt)
            keep = m.group(1).strip() if m else RE_PROC.sub("", txt).strip()
            if m or RE_DISEASE_HEAD.match(keep):
                off = e.text.find(keep)
                if off >= 0 and keep:
                    s = e.start + off
                    out.append(Entity(clean[s:s + len(keep)], s, s + len(keep),
                                      e.type, assertions=list(e.assertions),
                                      candidates=list(e.candidates)))
                    stats["proc_trimmed"] += 1
                    continue
            stats["proc_dropped"] += 1
            continue

        if e.type == "TRIỆU_CHỨNG" and RE_MEASURED.search(e.text):
            stats["measured_dropped"] += 1
            continue

        out.append(e)

    return out, stats


@dataclass
class NoteReport:
    note_id: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    n_entities: int = 0
    n_chars: int = 0
    density: float = 0.0
    type_counts: Dict[str, int] = field(default_factory=dict)
    dirty: Dict[str, bool] = field(default_factory=dict)


def check_note(note_id: str, marked: str) -> Tuple[NoteReport, str, List[Entity]]:
    """HARD gate trên note MARKUP (⟦TYPE|assert⟧…⟦/⟧).
    Trả về (report, clean_text, entities). Note hỏng -> ok=False."""
    try:
        clean, ents = parse_markup(sanitize_markup(marked))
    except ValueError as e:
        rep = NoteReport(note_id=note_id, ok=False)
        rep.errors.append(f"markup hỏng: {e}")
        return rep, "", []
    return check_parsed(note_id, clean, ents), clean, ents


def check_parsed(note_id: str, clean: str, ents: List[Entity]) -> NoteReport:
    """HARD gate trên cặp (text sạch, entities) đã parse sẵn — dùng để soi lại
    data/synth/notes/ + labels/ đã sinh từ trước mà không cần markup gốc."""
    rep = NoteReport(note_id=note_id, ok=True)
    rep.n_chars = len(clean)
    rep.n_entities = len(ents)

    if not ents:
        rep.ok = False
        rep.errors.append("0 khái niệm — note rỗng nhãn")
    if rep.n_chars < 200:
        rep.ok = False
        rep.errors.append(f"note quá ngắn ({rep.n_chars} ký tự)")

    # --- bất biến §7: offset khớp tuyệt đối ---
    for e in ents:
        if clean[e.start:e.end] != e.text:
            rep.ok = False
            rep.errors.append(f"offset lệch: {e.text!r} @ [{e.start},{e.end}]")

    # --- bất biến §7: KHÔNG span nào chồng lấn (0/2642 trên dữ liệu thật) ---
    ordered = sorted(ents, key=lambda e: (e.start, e.end))
    for a, b in zip(ordered, ordered[1:]):
        if b.start < a.end:
            rep.ok = False
            rep.errors.append(f"span chồng lấn: {a.text!r} và {b.text!r}")

    for e in ents:
        if e.type not in VALID_TYPES:
            rep.ok = False
            rep.errors.append(f"type lạ: {e.type!r}")
            continue
        # ĐỀ BÀI giới hạn assertions trong CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG.
        # XN/KQ mang assertion là vi phạm SPEC, không phải chuyện điểm số.
        if e.type not in ASSERTABLE_TYPES and e.assertions:
            rep.ok = False
            rep.errors.append(f"{e.type} mang assertion {e.assertions} — spec CẤM")
        if set(e.assertions) - VALID_ASSERTIONS:
            rep.ok = False
            rep.errors.append(f"assertion lạ: {e.assertions}")
        if e.type not in LINKABLE_TYPES and e.candidates:
            rep.ok = False
            rep.errors.append(f"{e.type} mang candidates — chỉ CHẨN_ĐOÁN/THUỐC mới có")

    # --- mật độ: gán thừa phình mẫu số N, đắt y hệt bỏ sót ---
    if rep.n_chars:
        rep.density = rep.n_entities / rep.n_chars * 1000
        lo, hi = DENSITY_BAND
        if not (lo <= rep.density <= hi):
            rep.ok = False
            rep.errors.append(
                f"mật độ {rep.density:.1f}/1000 ký tự ngoài dải [{lo}, {hi}] "
                f"(mục tiêu {TARGET_DENSITY})")

    for e in ents:
        rep.type_counts[e.type] = rep.type_counts.get(e.type, 0) + 1

    rep.dirty = {
        "glue_case": has_glue_case(clean),
        "english": bool(RE_ENGLISH.search(clean)),
        "abbrev": bool(RE_ABBREV.search(clean)),
        "placeholder": bool(RE_PLACEHOLDER.search(clean)),
    }
    return rep


def check_batch(reports: List[NoteReport]) -> Tuple[bool, List[str]]:
    """SOFT gate ở mức MẺ. Đây là tầng mà LLM judge KHÔNG THỂ thay thế: nó đọc
    từng note nên không bao giờ thấy được phân bố lệch."""
    good = [r for r in reports if r.ok]
    msgs: List[str] = []
    if not good:
        return False, ["không note nào qua được HARD gate"]

    total_ents = sum(r.n_entities for r in good)
    total_chars = sum(r.n_chars for r in good)
    ok = True

    density = total_ents / total_chars * 1000 if total_chars else 0.0
    msgs.append(f"mật độ mẻ      : {density:5.2f}/1000 ký tự   (mục tiêu {TARGET_DENSITY})")
    if abs(density - TARGET_DENSITY) > 3.0:
        ok = False
        msgs[-1] += "   ❌ LỆCH -> sửa prompt, đừng train"

    msgs.append("phân bố type   :")
    for t, tgt in TARGET_TYPE_MIX.items():
        got = sum(r.type_counts.get(t, 0) for r in good) / total_ents if total_ents else 0.0
        flag = ""
        if abs(got - tgt) > TYPE_MIX_TOL:
            ok = False
            flag = f"   ❌ lệch {(got-tgt)*100:+.1f} điểm %"
        msgs.append(f"    {t:22s} {got*100:5.1f}%   (mục tiêu {tgt*100:4.1f}%){flag}")

    msgs.append("đặc tính 'bẩn' :")
    for k, tgt in TARGET_DIRTY.items():
        got = sum(1 for r in good if r.dirty.get(k)) / len(good)
        flag = ""
        # Dung sai phải BIẾT CỠ MẪU. Đây là tỉ lệ nhị thức: với n=21 note và
        # p=0.12 thì độ lệch chuẩn đã là 7.1%, nên một ngưỡng cứng ±10 điểm %
        # chỉ bằng 1.4σ — nó la làng trên mẻ pilot nhỏ dù dữ liệu không sai gì
        # (đo được: abbrev nhảy 0% -> 17% -> 9% -> 24% qua 4 pilot 25 note).
        # Ở n=500 thì 2σ chỉ còn ~1.5%, nên ngưỡng cứng DIRTY_TOL lại thành cái
        # chặt hơn và vẫn có tác dụng. Lấy cái LỎNG HƠN trong hai cái.
        sigma = (tgt * (1 - tgt) / len(good)) ** 0.5
        tol = max(DIRTY_TOL, 2 * sigma)
        if abs(got - tgt) > tol:
            if k == "glue_case":
                # bộ dò chưa đầy đủ (xem chú thích ở RE_UNIT_MASK) -> chỉ cảnh
                # báo. Đánh trượt cả mẻ bằng một phép đo ta biết là thiếu sót
                # thì còn tệ hơn không đo.
                flag = f"   ⚠ lệch {(got-tgt)*100:+.0f} điểm % (bộ dò chỉ bắt được dính-chữ-hoa/thường)"
            else:
                ok = False
                flag = f"   ❌ lệch {(got-tgt)*100:+.0f} điểm %"
        msgs.append(f"    {k:22s} {got*100:5.1f}%   (mục tiêu {tgt*100:4.1f}%){flag}")

    avg_len = total_chars / len(good)
    tgt_len, tol = TARGET_LEN
    msgs.append(f"độ dài TB      : {avg_len:6.0f} ký tự      (mục tiêu {tgt_len}±{tol})")
    if abs(avg_len - tgt_len) > tol:
        ok = False
        msgs[-1] += "   ❌ LỆCH"

    return ok, msgs
