"""End-to-end orchestration for one note.

Pipeline:
    raw text
      -> section annotation (metadata only; text never cut/normalized)
      -> candidate extraction (labs, drugs, symptoms, diagnoses) [+ optional NER model]
      -> conflict resolution / dedup / longest-span-wins
      -> assertion detection (isNegated / isFamily / isHistorical)
      -> entity linking (ICD-10 / RxNorm)  [gated by EMIT_CANDIDATES]
      -> validate (hard offset + enum invariants)
      -> list[dict]

The rule pipeline needs no GPU/model. The NER model is opt-in via Pipeline(
model_dir=...).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from .schema import Entity, TYPE_DIAGNOSIS, TYPE_DRUG, TYPE_SYMPTOM, TYPE_TEST_RESULT
from .offsets import read_text
from .sections import SectionCues, annotate
from .backends.base import SpanBackend
from .backends.rules import RuleBackend
from .assertions.detect import AssertionDetector
from .postprocess.merge import resolve, dedup_window_overlap
from .postprocess.validate import validate_entities
from .linking.icd10 import ICD10Linker
from .linking.rxnorm import RxNormLinker

# Default for whether CHẨN_ĐOÁN/THUỐC candidates are emitted. When False they
# come out as [] (linking left off until the real ICD-10-CM / RxNorm DBs are
# wired). This is only the DEFAULT — callers override it without touching code:
#   * Pipeline(emit_candidates=True)
#   * scripts:  --emit true / --emit false
#   * env var:  EMIT_CANDIDATES=1  (handy in a Kaggle/Colab notebook cell)
# so a read-only clone (e.g. Kaggle) never has to edit this file.
EMIT_CANDIDATES = False

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def parse_emit(value) -> Optional[bool]:
    """Normalize a CLI/env value to a bool, or None if unset (use default)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"--emit expects true/false (got {value!r})")


def _default_emit() -> bool:
    """Module default, overridable by the EMIT_CANDIDATES env var."""
    env = parse_emit(os.environ.get("EMIT_CANDIDATES"))
    return EMIT_CANDIDATES if env is None else env


# Tên loạn nhịp/dẫn truyền — khi ĐỌC ĐIỆN TÂM ĐỒ, đây là KẾT_QUẢ_XÉT_NGHIỆM
# (xem _reclassify_ecg_findings). FULLMATCH nên chỉ áp cho span ĐÚNG một finding
# trần, không đụng câu dài hơn ("suy tim do rung nhĩ" giữ nguyên CHẨN_ĐOÁN).
_ECG_FINDING = re.compile(
    r"(nhịp\s+xoang(\s+(chiếm\s+ưu\s+thế|đều|không\s+đều))?|"
    r"nhịp\s+(nhanh|chậm)\s+xoang(\s+tần\s+số[^,.;]*)?|"
    r"ngoại\s+tâm\s+thu\s+(nhĩ|thất)([^,.;]*)?|"
    r"rung\s+nhĩ([^,.;]*)?|cuồng\s+nhĩ([^,.;]*)?|"
    r"block\s+nhĩ\s+thất\s+độ\s+(i{1,3}|1|2|3)([^,.;]*)?|"
    r"block\s+nhánh\s+(trái|phải)([^,.;]*)?|"
    r"sóng\s+t\s+(đảo|dẹt)([^,.;]*)?|"
    r"st\s+chênh\s+(lên|xuống)([^,.;]*)?|"
    r"trục\s+điện\s+tim\s+lệch\s+(trái|phải)|"
    r"qt\s+kéo\s+dài([^,.;]*)?|phì\s+đại\s+thất\s+trái)",
    re.IGNORECASE)
_ECG_CUE = re.compile(
    r"(điện\s+tâm\s+đồ|điện\s+tim\b|đtđ|\becg\b|holter|monitor\s+(theo\s+dõi|tim)?|"
    r"ekg)",
    re.IGNORECASE)

# Cụm mô tả triệu chứng CHUNG (không phải tên bệnh riêng) mà model hay gán
# nhầm CHẨN_ĐOÁN — audit thật (sprint 2026-07-18, so 01-cuong) xác nhận GT
# LUÔN gán TRIỆU_CHỨNG cho các cụm CHÍNH XÁC này. Khác với dictionary
# `diagnoses_vi_icd.yaml` (tên bệnh CHÍNH THỨC luôn là CHẨN_ĐOÁN dù model đoán
# TRIỆU_CHỨNG) — đây là chiều NGƯỢC LẠI: mô tả triệu chứng chung bị model đoán
# nhầm thành tên bệnh. Match CASEFOLD CHÍNH XÁC (không phải substring) để
# tránh đụng những cụm dài hơn hợp lệ là CHẨN_ĐOÁN thật (vd "hội chứng nút
# xoang bệnh lý" có thể chứa "nhịp tim chậm" nhưng không match ở đây vì không
# đúng TOÀN BỘ span).
_SYMPTOM_NOT_DIAGNOSIS = {
    "đau đầu gối", "rối loạn dáng đi", "cơn rối loạn ý thức thoáng qua",
    "cơn co giật", "co giật", "suy giảm huyết động", "đau hố chậu",
    "nhịp tim chậm", "cơn nhịp tim chậm nặng", "nhịp tim chậm nặng",
}

# ĐÃ THỬ (2026-07-19, S48-50): reclassify KẾT_QUẢ_XÉT_NGHIỆM mở đầu bằng cue
# phủ định ("không phát hiện X"...) sang CHẨN_ĐOÁN theo "luật vàng ngoại lệ"
# (src/synth/prompts.py rule 8). Local-sim (F1 so 01-cuong) cải thiện rõ sau
# khi thu hẹp phạm vi (loại trừ mô tả chung chung "gì bất thường"...), NHƯNG
# **NỘP THẬT xác nhận NGƯỢC LẠI: WER thật TĂNG** (49.3925->49.5103) dù local-
# sim dự đoán giảm — bằng chứng RÕ NHẤT từ trước giờ cho "01-cuong là silver,
# không phải gold" (xem memory 01-cuong-is-silver-not-gold): rule làm pred
# GIỐNG 01-cuong hơn nhưng GIỐNG GOLD THẬT ÍT hơn, tức 01-cuong tự nó xử lý
# case "phủ định tên bệnh cụ thể" khác gold thật. ĐÃ REVERT HẲN — không dùng
# nữa, dù local-sim từng cho thấy cải thiện.

# "LUẬT VÀNG" (xem src/synth/prompts.py): đứng sau động từ tường thuật kết quả
# xét nghiệm thì là KẾT_QUẢ_XÉT_NGHIỆM dù mang tên bệnh. Dùng làm lưới an toàn
# thứ 2 CHỈ cho danh sách imaging_finding đã flag (xem _reclassify_imaging_findings).
_NARRATIVE_RESULT_CUE = re.compile(
    r"(cho\s+thấy|ghi\s+nhận|phát\s+hiện)", re.IGNORECASE)


class Pipeline:
    """Rule backend (always on) + an OPTIONAL extra backend (encoder / LLM),
    OR-merged then reconciled.

    `model` may be: None (rule-only), a registry key from configs/models.yaml,
    a checkpoint directory, or a ready SpanBackend instance.
    """

    def __init__(self, model=None, emit_candidates: Optional[bool] = None,
                 max_len: int = 256, stride: int = 64, rules_on: bool = True,
                 raw: bool = False):
        # rules_on=False -> BỎ toàn bộ heuristic domain (rule span backend +
        # reclassify imaging + assertion detector). Dùng để đo model backend
        # (GLiNER) MỘT MÌNH, không bị rule bồi/chỉnh. Linker (candidates) vẫn theo
        # emit_candidates vì nó không đổi span/type.
        #
        # raw=True -> ZERO post-processing (CLAUDE.md): bỏ luôn `resolve()`
        # (dedup/strip punct/quality-tail/merge KQXN/...), chỉ giữ đúng MỘT
        # bước dedup_window_overlap() (loại entity trùng CHÍNH XÁC do 2 window
        # chồng lấp dự đoán lại — bắt buộc về kỹ thuật, không đổi span/type).
        # Ép rules_on=False (không rule nào được chạy khi đo model thuần) và bỏ
        # qua reclassify/assertion/linking hoàn toàn, bất kể emit_candidates.
        self.raw = raw
        self.rules_on = False if raw else rules_on
        self.cues = SectionCues.load()
        self.rules = RuleBackend()
        self.assertions = AssertionDetector.load()
        self.icd = ICD10Linker.load()
        self.rx = RxNormLinker.load()

        # None -> fall back to module default / EMIT_CANDIDATES env var
        self.emit_candidates = (
            _default_emit() if emit_candidates is None else emit_candidates
        )

        self.extra: Optional[SpanBackend] = None
        if isinstance(model, SpanBackend):
            self.extra = model
        elif model:
            from .registry import build_backend
            self.extra = build_backend(model, max_len=max_len, stride=stride)

    def extract_entities(self, text: str) -> List[Entity]:
        if self.raw:
            # ZERO post-processing: model tự thân, chỉ dedup đúng-1-thứ bắt
            # buộc kỹ thuật (window chồng lấp), không rule/reclassify/
            # assertion/linking/strip/merge nào khác.
            cands = self.extra.predict(text) if self.extra is not None else []
            ents = dedup_window_overlap(cands)
            for e in ents:
                e.candidates = []
            validate_entities(ents, text)
            return ents

        cands: List[Entity] = self.rules.predict(text) if self.rules_on else []
        if self.extra is not None:
            cands += self.extra.predict(text)
        return self._postprocess(cands, text)

    def extract_entities_batch(self, texts: List[str], batch_size: int = 8) -> List[List[Entity]]:
        """Batched counterpart of `extract_entities` — same rule pipeline and
        post-processing per file (fast, CPU, so left as a plain loop), but the
        LLM backend call is done ONCE per chunk via `predict_batch` (if the
        backend supports it) instead of once per file. Ignores `self.raw`
        (batching is a throughput optimization for the normal hybrid path;
        the raw/model-only measurement path stays single-file, see
        `extract_entities`)."""
        rule_cands = [self.rules.predict(t) if self.rules_on else [] for t in texts]
        if self.extra is not None:
            if hasattr(self.extra, "predict_batch"):
                llm_cands = self.extra.predict_batch(texts, batch_size=batch_size)
            else:
                llm_cands = [self.extra.predict(t) for t in texts]
            for rc, lc in zip(rule_cands, llm_cands):
                rc.extend(lc)
        return [self._postprocess(cands, text) for cands, text in zip(rule_cands, texts)]

    def _postprocess(self, cands: List[Entity], text: str) -> List[Entity]:
        smap = annotate(text, self.cues)
        ents = resolve(cands, text)
        if self.rules_on:
            self._reclassify_imaging_findings(ents, smap, text)
            self._reclassify_ecg_findings(ents, text)
            self._reclassify_symptom_phrases(ents)
            self.assertions.apply(ents, text, smap)
        self._link(ents)

        # clear candidates on non-linkable / when disabled
        if not self.emit_candidates:
            for e in ents:
                e.candidates = []

        validate_entities(ents, text)
        return ents

    def _reclassify_imaging_findings(self, ents: List[Entity], smap, text: str) -> None:
        """A diagnosis surface flagged `imaging_finding: true` (data/diagnoses_
        vi_icd.yaml) is a plain lab/imaging FINDING, not a physician diagnosis,
        when it occurs inside a LAB/IMAGING section -> demote to
        KẾT_QUẢ_XÉT_NGHIỆM so it doesn't compete with real CHẨN_ĐOÁN entries.

        Lưới an toàn thứ 2 (chỉ cho danh sách FLAGGED này, KHÔNG áp dụng cho
        mọi CHẨN_ĐOÁN): nếu SectionMap không nhận
        LAB/IMAGING (note thuật lại kết quả trong đoạn văn tường thuật, không
        có heading riêng) nhưng có động từ tường thuật kết quả ("cho thấy",
        "ghi nhận", "phát hiện") NGAY TRƯỚC span cùng dòng, vẫn hạ xuống
        KẾT_QUẢ_XÉT_NGHIỆM — cùng cơ chế _reclassify_ecg_findings."""
        flagged = self.rules.diagnoses.imaging_finding_surfaces
        if not flagged:
            return
        for e in ents:
            # match theo canonical (rule backend) HOẶC theo text thô (model
            # backend không có canonical) -- cùng 1 danh sách flagged surfaces.
            if e.type != TYPE_DIAGNOSIS or (e.canonical not in flagged and e.text.casefold() not in flagged):
                continue
            if smap.category_at(e.start) in ("LAB", "IMAGING"):
                e.type = TYPE_TEST_RESULT
                continue
            line_start = text.rfind("\n", 0, e.start) + 1
            window = text[line_start:e.start]
            if _NARRATIVE_RESULT_CUE.search(window):
                e.type = TYPE_TEST_RESULT

    def _reclassify_ecg_findings(self, ents: List[Entity], text: str) -> None:
        """Tên loạn nhịp/dẫn truyền khi ĐỌC ĐIỆN TÂM ĐỒ là KẾT_QUẢ_XÉT_NGHIỆM,
        không phải CHẨN_ĐOÁN/TRIỆU_CHỨNG — cùng logic với reclassify imaging,
        nhưng bắt theo CỤM CỤC BỘ (cửa sổ ký tự trước span) thay vì SectionMap,
        vì câu đọc ECG hay nằm LỒNG trong mục tường thuật ("...monitor holter
        cho thấy Nhịp xoang...") chứ không phải dưới heading Cận lâm sàng riêng.

        Lưới an toàn cho model: 516+100 note train có ví dụ các finding này
        LUÔN kèm định ngữ ("ngoại tâm thu nhĩ rải rác"); câu thật hay để TRẦN
        ("ngoại tâm thu nhĩ và ngoại tâm thu thất xuất hiện thường xuyên") nên
        model chưa tổng quát hoá hết — rule này vá trực tiếp, không cần train
        lại."""
        for e in ents:
            if e.type not in (TYPE_DIAGNOSIS, TYPE_SYMPTOM):
                continue
            if not _ECG_FINDING.fullmatch(e.text.strip()):
                continue
            # cửa sổ = DÒNG hiện tại (không rộng hơn) — khớp cấu trúc gạch đầu
            # dòng của note thật ("- monitor holter cho thấy Nhịp xoang... Ghi
            # nhận ngoại tâm thu..." nhiều CÂU trong CÙNG MỘT dòng, câu sau
            # không tự lặp "holter" nhưng vẫn thuộc cùng phát hiện). Chỉ chặn ở
            # xuống dòng để không dính oan chẩn đoán tiền sử ở DÒNG/bullet khác.
            line_start = text.rfind("\n", 0, e.start) + 1
            window = text[line_start:e.start]
            if _ECG_CUE.search(window):
                e.type = TYPE_TEST_RESULT
                e.candidates = []
                e.candidates = []

    def _reclassify_symptom_phrases(self, ents: List[Entity]) -> None:
        """Downgrade CHẨN_ĐOÁN -> TRIỆU_CHỨNG cho các cụm mô tả triệu chứng
        chung đã xác nhận qua audit (xem _SYMPTOM_NOT_DIAGNOSIS ở trên) —
        chiều ngược với _reclassify_imaging_findings/_reclassify_ecg_findings
        (những hàm đó fix tên BỆNH bị đoán nhầm; hàm này fix mô tả TRIỆU
        CHỨNG bị đoán nhầm thành tên bệnh, kể cả khi trùng khớp
        _SAME_SPAN_PRIORITY ưu tiên sai chiều cho case này)."""
        for e in ents:
            if e.type == TYPE_DIAGNOSIS and e.text.casefold() in _SYMPTOM_NOT_DIAGNOSIS:
                e.type = TYPE_SYMPTOM

    def _link(self, ents: List[Entity]) -> None:
        for e in ents:
            if e.type == TYPE_DIAGNOSIS:
                e.candidates = self.icd.link(e)
            elif e.type == TYPE_DRUG:
                e.candidates = self.rx.link(e)
            else:
                e.candidates = []

    def process_text(self, text: str) -> List[dict]:
        return [e.to_dict() for e in self.extract_entities(text)]

    def process_file(self, path) -> List[dict]:
        return self.process_text(read_text(path))
