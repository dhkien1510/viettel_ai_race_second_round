"""Conflict resolver / dedup.

Rules (kept deliberately simple and observation-driven, not a guessed global
type priority):

1. Drop exact duplicates (same start, end, type).
2. Strip a trailing connective word ("... và", "... hoặc") a span over-extended
   into, and truncate a THUỐC span that swallowed its indication clause
   ("doxycycline cho viêm tuyến mồ hôi" -> just "doxycycline").
3. Drop entities whose FULL text (trimmed/normalized) is a known non-entity
   cue/meta word (see _BLACKLIST_TEXT) — model artifacts, not clinical content.
4. Harmonize type across repeated mentions of the same surface text within one
   document — the NER model decides type per local window/sentence, so the
   same clinical concept ("mệt mỏi", "rung nhĩ") can flip type from one
   occurrence to the next even though a single document should tag it
   consistently. Snap every occurrence to the majority type.
5. Longest-span-wins: drop any entity fully contained within a strictly longer
   entity. This resolves "nôn" inside "buồn nôn", "đau ngực" inside
   "đau ngực trái", etc.
6. For two entities on the EXACT same span but different type, keep one using a
   small, explainable preference (result value > test name > drug > diagnosis >
   symptom). Same-span cross-type collisions are rare in practice; overlaps are
   logged by scripts/inspect_predictions.py for review.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from ..schema import (
    Entity, TYPE_TEST_RESULT, TYPE_TEST_NAME, TYPE_DRUG, TYPE_DIAGNOSIS, TYPE_SYMPTOM,
)

# Exact-match blacklist: entities dropped ONLY when their entire (trimmed,
# lowercased, whitespace-collapsed) text equals one of these — never as a
# substring check, so a legitimate span like "không đau ngực" is untouched;
# only a degenerate entity whose ENTIRE text is just the cue/meta word itself
# gets dropped. None of these words are ever a valid standalone entity under
# any of the 5 types.
#
# Seeded two ways:
#   - section-header / meta words (đồng đội đề xuất): tiêu đề mục bệnh án hay
#     bị model học nhầm thành entity.
#   - cue words + template artifacts found by auditing the real
#     submission/0705/04 (vihealthbert) output across all 20 files: negation
#     cue "không" leaking out as its own CHẨN_ĐOÁN/TRIỆU_CHỨNG/THUỐC entity,
#     stray connectives ("còn", "của", "là", "có", "từ", "đây", "hơn", "bị",
#     "đi", "lần"), category-name-as-entity ("triệu chứng", "xét nghiệm",
#     "dấu hiệu", "chỉ số", "tình trạng", "lượng"), and "N/A" template
#     fragments mangled by tokenization ("n/a", "n/", "/a", "n").
#   - second audit round on output/xlmr_base_0607_1300 (20 eval files):
#     section-header leak ("thuốc" alone, from the "Thuốc trước khi nhập
#     viện" heading), negation-cue phrase leaking whole ("không có", distinct
#     from the already-blacklisted bare "không"), duration/connective
#     fragments ("khi", "dài", "giây", "tiếp", "hoặc", "và"), and a
#     psychosocial life-event mistaken for a clinical symptom ("mất việc
#     làm"). NOT blacklisted: any single Vietnamese word that IS a real
#     standalone symptom even at 2 chars ("ho" = cough) or a real short lab
#     token ("k" = kali/potassium) — a blind length cutoff would delete those
#     correct entities (verified: 8/17 "short" entities in that audit were
#     literally "ho"), so short entities are exact-matched here individually
#     instead of filtered by length.
_BLACKLIST_TEXT = {
    "tiền sử", "bệnh lý", "bệnh lý mãn tính", "kết quả", "ngày", "tháng",
    "không", "còn", "của", "là", "có", "từ", "đây", "lần", "đi", "bị", "hơn",
    "viên", "dấu hiệu", "triệu chứng", "chỉ số", "xét nghiệm", "thực sự",
    "tình trạng", "lượng", "thay thế", "cải thiện", "hôm nay", "có thể",
    "n/a", "n", "/a", "n/",
    "thuốc", "không có", "khi", "dài", "giây", "tiếp", "hoặc", "và",
    "mất việc làm",
    # sprint 2026-07-18 (S28): audit thật full 100 file so 01-cuong, 62/187
    # "extra TRIỆU_CHỨNG" là mảnh 1 từ — mỗi từ dưới đây đã kiểm tra thủ công
    # KHÔNG BAO GIỜ là entity triệu chứng độc lập hợp lệ (khác với "phù"/"đau"/
    # "nhói"/"ho" — những từ 1 âm tiết CÓ thể tự đứng làm triệu chứng thật, nên
    # CỐ Ý không đưa vào danh sách này).
    "khó", "sườn", "định", "khỏe", "phổi", "trở", "trái", "lan", "dội", "qua",
    "đầu", "mổ", "da", "cân", "nhẹ", "muộn", "tăng", "đào", "weak", "xuyên",
    "tay", "ăn", "dẫn", "thấy", "hộ", "theo", "tục",
    # cùng đợt audit (S29), bucket "extra TÊN_XÉT_NGHIỆM" — fragment tương tự,
    # KHÔNG đụng vào abbreviation/tên xét nghiệm thật dù ngắn (bun/ure/hco3-/
    # creatinine/magnesium/ntg đều bị loại khỏi danh sách này vì có thể là tên
    # xét nghiệm/thiết bị thật, chỉ là GT nhóm khác cách).
    "các", "tại", "lấy", "cứu", "chiều", "nghiệm", "bụng", "tiểu",
    # S37: "graft"/"stent"/"cabg"/"picc"/"turp"/"mask" là THIẾT BỊ/THỦ THUẬT
    # ĐIỀU TRỊ (quy tắc sinh data rule 11: thiết bị/thủ thuật điều trị KHÔNG
    # gán thẻ ở bất kỳ type nào — khác thăm dò CHẨN ĐOÁN như sinh thiết/nội
    # soi vẫn được gán). "tình cờ"/"tổng quát"/"mủ"/"máu" là fragment không
    # bao giờ tự đứng làm tên xét nghiệm.
    "tình cờ", "tổng quát", "mủ", "máu", "mask", "graft", "stent", "cabg",
    "picc", "turp",
    # S38: đợt audit "extra TRIỆU_CHỨNG" thứ 2 — mảnh 1 từ đã kiểm tra ngữ
    # cảnh thủ công (dữ<-"dữ dội", sâu<-"gập sâu", khuôn<-"thành khuôn",
    # đến<-"khi đến", đường<-"đường miệng", sữa<-"uống sữa", vị<-"Vị trí",
    # "-" là dấu gạch đầu dòng lẫn vào). LLQ/CPSOB/ngực là tên VỊ TRÍ giải
    # phẫu/viết tắt gộp nhiều triệu chứng, không phải bản thân 1 triệu chứng.
    "dữ", "sâu", "khuôn", "llq", "cpsob", "-", "đến", "đường", "sữa", "ngực",
    "vị",
}

# Trailing connective a span sometimes over-extends into ("rung nhĩ và" ->
# "rung nhĩ"). Matched as a literal suffix, so only stripped when the entity
# text ends with " và"/" hoặc"/etc. preceded by real content.
_TRAILING_CONNECTIVES = (" và", " hoặc", " của", " là", " thì", " mà")

# Đuôi tính-chất (mô tả diễn tiến/tần suất) không thuộc span — quy tắc đã
# VERIFY THẬT qua nộp thật (testing_theories_2.csv #34: bỏ đuôi này làm WER
# GIẢM, tức điểm tăng). KHÁC với định ngữ lâm sàng chuẩn như "về đêm"/"khi
# gắng sức" (testing_theories_2.csv #14: bỏ "về đêm" làm WER TĂNG, tức điểm
# giảm — PHẢI GIỮ những cụm đó). Chỉ liệt kê đúng cụm đã verify, không suy
# rộng ra các định ngữ khác chưa có bằng chứng.
_TRAILING_QUALITY_TAIL = (" ngày càng nặng", " liên tục", " tái phát",
                          " nhiều hơn", " kéo dài", " tương đối")

# Encoder backend (BIO tag per token) has no notion that a colon/comma/semicolon/
# period ends the concept — it regularly swallows the punctuation right after a
# span ("khó thở:" thay vì "khó thở", "buồn nôn," thay vì "buồn nôn", "đau ngực."
# thay vì "đau ngực"). Rule/dictionary spans don't have this problem (they stop
# at the known surface string), but the encoder model does it often enough
# (audit thật trên eval20: 141/1095 entity BARE bị dấu `:,;`, thêm ~85 entity
# nữa bị dấu `.` cuối câu) that it's worth a dedicated strip, separate from
# _TRAILING_CONNECTIVES (punctuation, not a word). "." an toàn để cắt vì nó chỉ
# xét KÝ TỰ CUỐI của span đã dự đoán — không đụng tới số thập phân giữa span
# (vd "3.7 cm" không đổi, chỉ "3.7." mới bị cắt về "3.7").
_TRAILING_PUNCT = ":,;."

# A THUỐC span that swallowed its indication clause ("doxycycline cho viêm
# tuyến mồ hôi") gets cut back to just the drug name/dose at the first
# occurrence of one of these markers. The indication half is DROPPED rather
# than re-tagged as CHẨN_ĐOÁN/TRIỆU_CHỨNG — guessing its type risks handing
# out a wrong-type entity, which scores 0 across all 3 metrics (see the
# scoring-metric memory), a worse outcome than just not extracting it.
_DRUG_INDICATION_MARKERS = (" cho ", " do ", " vì ")


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _drop_blacklisted(ents: List[Entity]) -> List[Entity]:
    return [e for e in ents if _normalize_text(e.text) not in _BLACKLIST_TEXT]


def _strip_trailing_suffixes(ents: List[Entity], suffixes: tuple) -> List[Entity]:
    for e in ents:
        for suf in suffixes:
            if e.text.endswith(suf) and len(e.text) > len(suf):
                e.end -= len(suf)
                e.text = e.text[: -len(suf)]
                break
    return ents


def _strip_trailing_connectives(ents: List[Entity]) -> List[Entity]:
    return _strip_trailing_suffixes(ents, _TRAILING_CONNECTIVES)


def _strip_trailing_quality_tail(ents: List[Entity]) -> List[Entity]:
    return _strip_trailing_suffixes(ents, _TRAILING_QUALITY_TAIL)


def _strip_trailing_punct(ents: List[Entity]) -> List[Entity]:
    for e in ents:
        # len>1 guard: một entity TOÀN dấu câu/khoảng trắng (vd model dự đoán
        # nhầm span chỉ có ":") không được strip sạch về rỗng (start==end) —
        # bug thật lộ ra khi đổi seed train (audit thật 2026-07-19): entity
        # rỗng làm validate_entities báo lỗi offset out of range. Giữ lại ký
        # tự cuối thay vì xoá hết; entity toàn rác kiểu này quá hiếm để cần
        # xử lý riêng, và giữ nguyên còn an toàn hơn xoá sạch.
        while len(e.text) > 1 and (e.text[-1] in _TRAILING_PUNCT or e.text[-1].isspace()):
            e.end -= 1
            e.text = e.text[:-1]
    return ents


# Encoder BIO-tag span đôi khi nuốt luôn dấu xuống dòng và trôi sang gạch đầu
# dòng/mục kế tiếp trong danh sách (vd "ck 58\n- alt 92" gộp 2 kết quả XN khác
# nhau; "iron\n\n2" nuốt luôn số thứ tự mục thuốc kế tiếp) — audit thật (sprint
# 2026-07-18) kiểm tra TOÀN BỘ 27 entity có "\n" trong span ở full 100 file:
# 100% là lỗi trôi mục, KHÔNG có case nào span hợp lệ cố ý chứa xuống dòng giữa
# nội dung. Cắt thẳng ở dấu "\n" ĐẦU TIÊN là an toàn tuyệt đối theo audit này.
def _truncate_at_newline(ents: List[Entity]) -> List[Entity]:
    for e in ents:
        idx = e.text.find("\n")
        if idx > 0:
            e.end = e.start + idx
            e.text = e.text[:idx]
    return ents


# Đuôi phủ định ("Không đánh trống ngực" -> "đánh trống ngực") KHÔNG thuộc
# span — assertion isNegated đã ghi nhận riêng, span chỉ giữ đúng tên khái
# niệm. Audit thật (sprint 2026-07-18, so 01-cuong) xác nhận nhất quán: GT
# luôn bỏ "Không "/"không " tiền tố dù pred (model) hay gộp câu phủ định
# nhiều triệu chứng đều giữ lại. Chỉ cắt CHỮ ĐẦU, không đụng "không" xuất
# hiện giữa/cuối span (vd "không đặc hiệu" vẫn giữ nguyên nếu đó là toàn bộ
# nội dung sau khi cắt).
_LEADING_NEGATION = ("Không ", "không ")

# ĐÃ THỬ (2026-07-19): strip tiền tố chung chung "triệu chứng "/"cảm giác "/
# "cơn " trước tên triệu chứng — THẤT BẠI, ĐÃ BỎ. Lý do: "cảm giác" không
# LUÔN LÀ meta-từ thừa — audit thật cho thấy GT GIỮ "cảm giác" khi theo sau
# là "như..." (mô tả ví von, vd "cảm giác như vòi nước", "cảm giác như bị
# xịt nước...") nhưng BỎ khi theo sau là tên triệu chứng trực tiếp (vd "cảm
# giác đánh trống ngực" -> "đánh trống ngực"). Ranh giới phụ thuộc NGỮ CẢNH
# theo sau, không phải bản thân từ "cảm giác" — quy tắc chung (strip mù) làm
# WER(sim) TỆ HƠN (35.34->35.40/35.42) dù một số case riêng lẻ đúng.


def _strip_leading_prefixes(ents: List[Entity], prefixes: tuple, types: tuple) -> List[Entity]:
    for e in ents:
        if e.type not in types:
            continue
        changed = True
        while changed:
            changed = False
            for pre in prefixes:
                if e.text.startswith(pre) and len(e.text) > len(pre):
                    e.start += len(pre)
                    e.text = e.text[len(pre):]
                    changed = True
                    break
            # nuốt khoảng trắng thừa lộ ra sau khi cắt tiền tố (vd nguồn có
            # 2 dấu cách dính "cơn  đau đầu" -> cắt "cơn " chỉ còn 1 dấu cách
            # đầu span) — lặp while ở trên để xử lý luôn nếu có tiền tố khác
            # lộ ra tiếp theo.
            while e.text[:1].isspace() and len(e.text) > 1:
                e.start += 1
                e.text = e.text[1:]
                changed = True
    return ents


def _strip_leading_negation(ents: List[Entity]) -> List[Entity]:
    return _strip_leading_prefixes(ents, _LEADING_NEGATION, (TYPE_SYMPTOM, TYPE_DIAGNOSIS))


# Kết quả xét nghiệm dạng đoạn văn (vd đọc ECG có nhiều finding trong 1 đoạn)
# phải GỘP thành 1 entity — đã VERIFY THẬT qua nộp thật (testing_theories.csv):
# tách "tim to"+"không có bất thường nào khác" ra 2 cụm làm GIẢM điểm; gộp
# "giãn đường mật, gợi ý tắc nghẽn đường mật" làm TĂNG điểm. Model (và cả
# data/synth/labels) có xu hướng tách theo từng finding riêng thay vì gộp cả
# đoạn — đây là hậu xử lý CHỦ Ý THẬN TRỌNG: chỉ gộp 2 entity CÙNG type
# KẾT_QUẢ_XÉT_NGHIỆM, khoảng cách <= _MAX_MERGE_GAP ký tự, và không cắt ngang
# gạch đầu dòng/đoạn mới (tránh gộp nhầm 2 kết quả không liên quan trong list).
# Verify trên eval20 (24 lần gộp): WER(sim) giảm 34.39 -> 33.95.
_MAX_MERGE_GAP = 30

# Gạch đầu dòng THỤT LỀ ("\n    - ", "\n\t- ") vẫn là ranh giới mục danh sách —
# check cũ chỉ bắt "\n-" DÍNH LIỀN (không có khoảng trắng ở giữa), bỏ lọt mọi
# bullet có thụt lề (rất phổ biến trong note thật, vd "- ast 92\n    - alt 58"),
# khiến 2 kết quả XN KHÁC NHAU trong 2 mục riêng biệt bị gộp nhầm thành 1 span
# tràn qua nhiều dòng (audit thật: 17/2803 entity vẫn còn "\n" giữa span dù đã
# có `_truncate_at_newline` chạy TRƯỚC bước gộp này trong `resolve()`).
_BULLET_BOUNDARY = re.compile(r"\n[ \t]*[-•*]|\n\s*\n")


def _merge_adjacent_test_results(ents: List[Entity], text: str) -> List[Entity]:
    ents = sorted(ents, key=lambda e: (e.start, e.end))
    out: List[Entity] = []
    i = 0
    while i < len(ents):
        cur = ents[i]
        j = i + 1
        while j < len(ents):
            nxt = ents[j]
            if cur.type != TYPE_TEST_RESULT or nxt.type != TYPE_TEST_RESULT:
                break
            gap = nxt.start - cur.end
            if gap <= 0 or gap > _MAX_MERGE_GAP:
                # gap<=0: 2 span dính liền/chồng lấn không có khoảng trắng —
                # KHÔNG phải câu văn liên tục (mọi ví dụ đã verify đều có ít
                # nhất 1 ký tự phân cách), mà là chữ GHÉP DÍNH (vd 1 note thật
                # lặp "bình thường" 4 lần liền không dấu cách) — gộp sẽ xoá
                # mất ranh giới giữa các khái niệm LẶP LẠI riêng biệt.
                break
            gap_text = text[cur.end:nxt.start]
            if _BULLET_BOUNDARY.search(gap_text):
                break
            cur.end = nxt.end
            cur.text = text[cur.start:cur.end]
            j += 1
        out.append(cur)
        i = j if j > i + 1 else i + 1
    return out


# Model hay gộp TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM thành MỘT span khi 2 cụm
# nối bằng " là " (vd "kali là 2.4", "ast (aspartate aminotransferase) là 319").
# Không dùng match rộng cho mọi span chứa " là " vì dễ bắt oan câu văn
# tường thuật dài. Bản này siết chặt: chỉ tách khi
#   (a) đúng MỘT " là " trong toàn bộ text,
#   (b) phần TRƯỚC "là" ngắn (<=50 ký tự) và KHÔNG chứa dấu phẩy/động từ tường
#       thuật (cho thấy/ghi nhận/phát hiện) — loại trừ câu văn dài,
#   (c) phần SAU "là" bắt đầu NGAY bằng chữ số (giá trị đo, không phải mô tả).
# Đo lại F1 SAU khi thêm để xác nhận không lặp lại thất bại của S12 trước khi
# giữ lại.
_NAME_VALUE_RE = re.compile(r"^(.{1,50}?)\s+là\s+(\d.*)$")
_NAME_VALUE_BAD_MARKERS = (",", "cho thấy", "ghi nhận", "phát hiện")


def _split_test_name_value(ents: List[Entity]) -> List[Entity]:
    out: List[Entity] = []
    for e in ents:
        if e.type != TYPE_TEST_RESULT:
            out.append(e)
            continue
        m = _NAME_VALUE_RE.match(e.text)
        if not m:
            out.append(e)
            continue
        name_part, val_part = m.group(1), m.group(2)
        if any(marker in name_part for marker in _NAME_VALUE_BAD_MARKERS):
            out.append(e)
            continue
        name_end = e.start + len(name_part)
        val_start = e.end - len(val_part)
        out.append(Entity(text=name_part, start=e.start, end=name_end,
                           type=TYPE_TEST_NAME, source=e.source))
        out.append(Entity(text=val_part, start=val_start, end=e.end,
                           type=TYPE_TEST_RESULT, source=e.source))
    return out


def _truncate_drug_indication(ents: List[Entity]) -> List[Entity]:
    for e in ents:
        if e.type != TYPE_DRUG:
            continue
        for marker in _DRUG_INDICATION_MARKERS:
            idx = e.text.find(marker)
            if idx > 0:
                e.end = e.start + idx
                e.text = e.text[:idx]
                break
    return ents


# On a true tie between a linkable type and a non-linkable one, prefer the
# linkable type: if the harmonized guess turns out right it also unlocks ICD-
# 10/RxNorm candidates (40% of the score, see scoring-metric memory); if wrong,
# the entity scores 0 either way, so there's no downside to picking it.
_TYPE_TIE_BREAK = (TYPE_DIAGNOSIS, TYPE_DRUG)


def _harmonize_types(ents: List[Entity]) -> List[Entity]:
    groups: dict = {}
    for e in ents:
        groups.setdefault(_normalize_text(e.text), []).append(e)
    for group in groups.values():
        if len({e.type for e in group}) <= 1:
            continue
        counts = Counter(e.type for e in group)
        best = max(counts.values())
        tied = [t for t, c in counts.items() if c == best]
        winner = tied[0]
        if len(tied) > 1:
            for t in _TYPE_TIE_BREAK:
                if t in tied:
                    winner = t
                    break
        for e in group:
            e.type = winner
    return ents


_SAME_SPAN_PRIORITY = {
    TYPE_TEST_RESULT: 0,
    TYPE_TEST_NAME: 1,
    TYPE_DRUG: 2,
    TYPE_DIAGNOSIS: 3,
    TYPE_SYMPTOM: 4,
}


def dedup_window_overlap(ents: List[Entity]) -> List[Entity]:
    """Chỉ loại entity TRÙNG CHÍNH XÁC (cùng start/end/type) — thứ duy nhất
    thật sự bắt buộc về mặt kỹ thuật khi chunk dài hơn max_len bị chia thành
    nhiều window CHỒNG LẤP (stride>0): 1 entity nằm trong vùng chồng lấp bị
    2 window độc lập dự đoán ra y hệt nhau, nếu không dedup sẽ bị xuất 2 lần
    trong JSON (tính thành 1 đúng + 1 THỪA khi chấm). KHÔNG đổi span/text/type
    của bất kỳ entity nào — không phải "mài dũa" theo nghĩa CLAUDE.md, chỉ là
    ghép kết quả nhiều window thành 1 danh sách. Dùng cho code path "raw" (zero
    post-processing) — xem `Pipeline(raw=True)` trong `src/infer.py`."""
    return _dedup_exact(ents)


def _dedup_exact(ents: List[Entity]) -> List[Entity]:
    seen = {}
    for e in ents:
        k = e.key()
        if k not in seen:
            seen[k] = e
        else:
            # merge assertions/candidates from duplicate producers
            prev = seen[k]
            prev.assertions = sorted(set(prev.assertions) | set(e.assertions))
            if not prev.candidates and e.candidates:
                prev.candidates = e.candidates
    return list(seen.values())


def _resolve_same_span(ents: List[Entity]) -> List[Entity]:
    by_span = {}
    for e in ents:
        by_span.setdefault((e.start, e.end), []).append(e)
    out = []
    for span, group in by_span.items():
        if len(group) == 1:
            out.append(group[0])
        else:
            group.sort(key=lambda x: _SAME_SPAN_PRIORITY.get(x.type, 99))
            out.append(group[0])
    return out


def _drop_nested(ents: List[Entity]) -> List[Entity]:
    # sort by length desc so containers come first
    ordered = sorted(ents, key=lambda e: e.length, reverse=True)
    kept: List[Entity] = []
    for e in ordered:
        contained = False
        for k in kept:
            if k.start <= e.start and e.end <= k.end and k.length > e.length:
                contained = True
                break
        if not contained:
            kept.append(e)
    return kept


# Cửa sổ trượt (max_len/stride) khiến CÙNG một cụm được dự đoán LẶP LẠI bởi
# nhiều window chồng lấp, mỗi lần biên hơi khác nhau (không nested tuyệt đối,
# _drop_nested bỏ lọt) — audit thật (sprint 2026-07-18, file 58) xác nhận:
# hàng chục candidate CÙNG type chồng lấn 1 phần quanh cùng vị trí ("chân phải,"
# / "không thể chịu lực ở chân phải," / "tỉnh chậm," / "tỉnh chậm, phản xạ
# kém." / "phản xạ kém." đều ở gần vị trí 2340-2422). Gộp: trong mỗi cụm
# CHỒNG LẤN (bất kỳ overlap>0) CÙNG type, chỉ giữ span DÀI NHẤT.
def _collapse_overlapping_same_type(ents: List[Entity]) -> List[Entity]:
    by_type: dict = {}
    for e in ents:
        by_type.setdefault(e.type, []).append(e)
    kept: List[Entity] = []
    for group in by_type.values():
        ordered = sorted(group, key=lambda e: e.length, reverse=True)
        chosen: List[Entity] = []
        for e in ordered:
            if any(overlap(e, k) > 0 for k in chosen):
                continue
            chosen.append(e)
        kept.extend(chosen)
    return kept


def overlap(a: Entity, b: Entity) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def resolve(ents: List[Entity], text: str = "") -> List[Entity]:
    ents = _dedup_exact(ents)
    ents = _collapse_overlapping_same_type(ents)
    ents = _truncate_at_newline(ents)
    # Punct trước, để "...liên tục." không bị dấu "." chặn khớp đuôi tính-chất/
    # connective bên dưới; rồi cắt lại 1 lần nữa cho khoảng trắng phát sinh.
    ents = _strip_trailing_punct(ents)
    ents = _strip_trailing_connectives(ents)
    ents = _strip_trailing_quality_tail(ents)
    ents = _strip_trailing_punct(ents)
    ents = _strip_leading_negation(ents)
    if text:
        ents = _merge_adjacent_test_results(ents, text)
        # Lưới an toàn thứ 2: nếu bước gộp trên vẫn để lọt "\n" (vd gộp qua
        # ranh giới không phải bullet nhưng vẫn chứa xuống dòng), cắt lại.
        ents = _truncate_at_newline(ents)
    ents = _split_test_name_value(ents)
    ents = _truncate_drug_indication(ents)
    ents = _drop_blacklisted(ents)
    ents = _harmonize_types(ents)
    ents = _resolve_same_span(ents)
    ents = _drop_nested(ents)
    if text:
        from .veto_filters import apply_veto_filters
        ents = apply_veto_filters(ents, text)
    ents.sort(key=lambda e: (e.start, e.end))
    return ents


def find_overlaps(ents: List[Entity]) -> List[tuple]:
    """Return partially-overlapping (not nested, not identical) pairs — for
    diagnostics only."""
    out = []
    s = sorted(ents, key=lambda e: (e.start, e.end))
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = s[i], s[j]
            if b.start >= a.end:
                break
            nested = (a.start <= b.start and b.end <= a.end) or (b.start <= a.start and a.end <= b.end)
            if not nested:
                out.append((a, b))
    return out
