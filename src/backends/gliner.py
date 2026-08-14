"""GLiNER span backend — urchade/gliner_multi-v2.1 fine-tune.

GLiNER gán span TRÊN TOKEN của input rồi trả CHAR-OFFSET trực tiếp, nên offset
khớp raw về mặt cấu trúc (không cần relocate() như LLMBackend). Nó chỉ lo
span+type; assertion do rule src/assertions/detect.py gắn sau trong Pipeline.

Note thật dài tới ~1476 token, vượt cửa sổ GLiNER (~384). Ta CHUNK theo ký tự có
overlap rồi dời offset về gốc và khử trùng — y như encoder dùng sliding window.
Nếu bỏ chunk, GLiNER lặng lẽ cắt phần đuôi note dài (giống bug max_length của Qwen).
"""

from __future__ import annotations

from typing import List

from .base import SpanBackend
from ..schema import Entity, VALID_TYPES

# Nhãn đưa cho GLiNER = đúng tên type trong schema (đã train với tên này). Giữ
# nguyên để inference map thẳng label -> type, không qua bảng trung gian dễ lệch.
LABELS = sorted(VALID_TYPES)

# 1100 ký tự ≈ 343 token < GIỚI HẠN CỨNG 384 của GLiNER processor (1400c cũ ra
# tới 409 token -> processor lặng lẽ cắt còn 384 -> entity ở ĐUÔI chunk bị bỏ sót,
# tụt recall). Overlap 400 ký tự > KẾT_QUẢ_XÉT_NGHIỆM dài nhất (57 từ ≈ 350 ký tự)
# nên khái niệm vắt qua ranh giới vẫn nằm TRỌN trong chunk kế. PHẢI khớp
# scripts/build_gliner_dataset.py để phân phối train/infer trùng nhau.
CHUNK_CHARS = 1100
OVERLAP_CHARS = 400


def _chunks(text: str, size: int, overlap: int):
    """(đoạn, offset_gốc). Cắt tại khoảng trắng gần nhất để không xé giữa từ."""
    if len(text) <= size:
        yield text, 0
        return
    step = size - overlap
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            sp = text.rfind(" ", i + step, end)
            if sp != -1:
                end = sp
        yield text[i:end], i
        if end >= n:
            break
        i = end


class GLiNERBackend(SpanBackend):
    name = "gliner"

    def __init__(self, model_id: str, threshold: float = None, device: str = None):
        # Núm ĐỘ NHẠY: hạ threshold -> GLiNER phun nhiều span hơn (recall lên,
        # precision xuống). Chỉnh nhanh không cần sửa code:  GLINER_THRESHOLD=0.35
        # LƯU Ý: nhạy hơn = nhiều rác (header/span cụt) -> resolve()/rule phải dọn.
        import os
        if threshold is None:
            threshold = float(os.environ.get("GLINER_THRESHOLD", "0.5"))
        try:
            from gliner import GLiNER
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "Thiếu gliner. Cài:  pip install gliner\n"
                f"(lỗi gốc: {exc})"
            )
        print(f"[gliner] threshold={threshold}")
        self.model = GLiNER.from_pretrained(model_id)
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.model = self.model.to(device)
        self.model.eval()
        self.threshold = threshold

    def predict(self, text: str) -> List[Entity]:
        seen = set()
        out: List[Entity] = []
        for chunk, base in _chunks(text, CHUNK_CHARS, OVERLAP_CHARS):
            for r in self.model.predict_entities(chunk, LABELS, threshold=self.threshold):
                typ = r["label"]
                if typ not in VALID_TYPES:
                    continue
                s = base + r["start"]
                e = base + r["end"]
                # HÀNG RÀO: offset phải trỏ đúng chuỗi trong raw. GLiNER trả char
                # offset trên CHUNK; sau khi dời về gốc, xác nhận lại — chunk cắt
                # ở khoảng trắng nên không lệch, nhưng kiểm cho chắc.
                if text[s:e] != r["text"]:
                    continue
                key = (s, e, typ)
                if key in seen:              # khử trùng vùng overlap giữa 2 chunk
                    continue
                seen.add(key)
                out.append(Entity(text[s:e], s, e, typ,
                                  assertions=[], candidates=[], source="gliner"))
        return out
