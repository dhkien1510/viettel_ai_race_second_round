"""Generative LLM backend (group C: Qwen2.5, zero-shot — no training).

The model reads the note and emits a JSON list of concepts (text + type +
assertions). It does NOT emit positions; we relocate each surface onto the raw
text (src/relocate.py) so offsets stay exact. Requires torch + transformers;
loaded lazily so the rule pipeline never imports them.
"""

from __future__ import annotations

import json
import re
from typing import List

from ..schema import (
    Entity, VALID_TYPES,
    TYPE_SYMPTOM, TYPE_TEST_NAME, TYPE_TEST_RESULT, TYPE_DIAGNOSIS, TYPE_DRUG,
)
from ..relocate import relocate
from .base import SpanBackend

_SYSTEM = (
    "Bạn là hệ thống trích xuất khái niệm y khoa từ bệnh án/hỏi-đáp y tế tiếng Việt. "
    "Chỉ trả về JSON, không giải thích."
)

_INSTRUCTION = """Trích xuất MỌI khái niệm y tế trong đoạn văn dưới đây.

Mỗi khái niệm là 1 object JSON gồm:
- "text": cụm từ XUẤT HIỆN NGUYÊN VĂN trong đoạn. Copy y hệt: giữ nguyên chính
  tả, dấu, viết hoa. KHÔNG sửa lỗi, KHÔNG dịch, KHÔNG viết lại, KHÔNG chuẩn hóa.
- "type": một trong {TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC}.
- "assertions": tập con của ["isNegated","isFamily","isHistorical"], mặc định [].
    isNegated  = bị phủ định ("không sốt").
    isFamily   = của người nhà, không phải bệnh nhân.
    isHistorical = tiền sử / bệnh nền / thuốc dùng trước khi nhập viện.
    (chỉ áp dụng cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC)

NGUYÊN TẮC CỐT LÕI: chỉ giữ 1 cụm khi nó hoạt động như 1 KHÁI NIỆM LÂM SÀNG ĐỘC LẬP
trong CHÍNH câu đó (triệu chứng/dấu hiệu cụ thể của ca bệnh, tên bệnh cụ thể, tên xét
nghiệm/thủ thuật cụ thể, 1 giá trị/finding cụ thể, hoặc tên thuốc/nhóm thuốc đang
dùng thật) — không gán chỉ vì cụm từ nghe "có vẻ y khoa".

KHÔNG GÁN (loại bỏ dù nghe có vẻ y khoa):
- Mệnh đề giải thích CƠ CHẾ/SINH LÝ BỆNH thuần giáo khoa trong câu định nghĩa "X là
  tình trạng Y" (giữ tên bệnh X, bỏ phần giải thích cơ chế Y) — NHƯNG xét theo TỪNG
  CHỖ xuất hiện: nếu cùng cụm đó ở chỗ KHÁC lại mô tả 1 finding THẬT của ca bệnh thì
  vẫn giữ ở chỗ đó.
- Trạng thái sinh lý bình thường: "có thai", "mang thai", "mãn kinh" (dù có mã ICD).
- Bệnh nhân TỰ HỎI về chẩn đoán CỦA CHÍNH MÌNH ("em có phải bị X không ạ?") khi
  không có nhận định y khoa nào xác nhận. ⚠️ CHỈ áp dụng khi CHÍNH BỆNH NHÂN tự hỏi —
  nếu bác sĩ nói "khả năng bị X" (dù dè dặt), hoặc đây là tiêu đề FAQ chung ("Bệnh X
  có lây không?"), hoặc câu cảnh báo hậu quả ("dẫn đến X") thì VẪN GÁN X.
- Danh sách tác dụng phụ/biến chứng chỉ CÓ THỂ xảy ra trong tương lai (không phải
  hiện tại của ca bệnh), và quyết định điều trị còn ĐỂ NGỎ chưa chốt ("bác sĩ sẽ
  quyết định tiếp tục hay ngưng thuốc X") — khác với thuốc/xét nghiệm ĐÃ/ĐANG thực
  hiện thật (luôn giữ, kể cả khi mới chỉ được đề nghị/khuyên làm).
- Từ chỉ KẾT CỤC/DI CHỨNG thuần: "tử vong", "bại não", "chậm phát triển trí tuệ" khi
  đứng riêng trong 1 danh sách biến chứng giả định — NHƯNG tên bệnh cơ quan thật
  trong CÙNG danh sách đó ("viêm tim", "suy tim", "nhồi máu cơ tim") vẫn GÁN.
- Cụm mô tả CHỨC NĂNG/KHẢ NĂNG chung chung (vẫn hoạt động bình thường, không phải 1
  finding cụ thể): "phản xạ được âm thanh", "cấu trúc X vẫn phát triển tốt".
- Câu bổ sung/chi tiết hoá LIÊN TỤC cho CÙNG 1 sự kiện đã có câu cốt lõi khác ngay
  trước/sau (không phải occurrence tách biệt ở thời điểm khác) — chỉ giữ câu cốt lõi.
- Nếu 1 sự kiện được nhắc cả trong đoạn kể lể tự do LẪN trong 1 mục có tiêu đề rõ
  ràng ("Lý do nhập viện:", "Bệnh sử hiện tại") → ưu tiên giữ bản trong mục có tiêu
  đề, bỏ bản kể lể tự do trùng lặp.
- Chuỗi bị che `***`/`xxx` (không đoán tên thuốc bị ẩn).
- Heading, số thứ tự mục, tên lĩnh vực, từ đơn quá chung khi câu đã có cụm cụ thể hơn.

VẪN PHẢI GÁN (dù trực giác muốn bỏ):
- Bệnh CẦN PHÒNG NGỪA dù bệnh nhân chưa mắc ("thuốc phòng ngừa tiền sản giật").
- Xét nghiệm/thủ thuật mới được ĐỀ NGHỊ/KHUYÊN LÀM, chưa thực hiện, chưa có kết quả.
- Tên bệnh thật trong nội dung LẠC ĐỀ/boilerplate giáo khoa (gán theo text thô tại
  chỗ, không xét có khớp chủ đề câu hỏi hay không).
- Mọi lần lặp lại hợp lệ của 1 khái niệm trong CÙNG đoạn văn (mỗi occurrence 1 object
  riêng) — trừ trường hợp là câu bổ sung liên tục cho cùng 1 sự kiện (xem trên).

QUY TẮC CẮT SPAN (rất quan trọng — chấm điểm theo WER, thừa/thiếu 1 từ đều bị trừ):
- Chỉ lấy ĐÚNG cụm tên khái niệm, NGẮN GỌN NHẤT nhưng ĐẦY ĐỦ. Với CHẨN_ĐOÁN và
  TRIỆU_CHỨNG, TUYỆT ĐỐI KHÔNG kèm:
    · động từ / mệnh đề dẫn: "cho thấy", "ghi nhận", "cảm thấy", "than phiền",
      "phát hiện", "được chẩn đoán", "xuất hiện", "tái phát"
    · nhãn mục: "Vị trí:", "Chẩn đoán:", "Lý do:", "Kết quả khám:"
    · mệnh đề nguyên nhân / diễn giải mức độ-thời gian-tiến triển: "... do ...",
      "... vì ...", "... ngày càng nặng", "... kéo dài", "... liên quan đến ..."
    · số đo / kích thước / mốc thời gian đi kèm (vd bỏ "kích thước 8mm", "trong 3 ngày").
  Ví dụ: lấy "Bệnh phổi kẽ", KHÔNG lấy "Bệnh phổi kẽ do dùng corticoid liều cao kéo dài".
- NHƯNG giữ nguyên đuôi định danh là 1 phần TÊN bệnh (không phải bổ ngữ): "típ 2",
  "mạn tính", "giai đoạn IV", "không đặc hiệu", "do rượu" — đây là tên bệnh, không cắt.
- Cặp "Tên A (hay/là Tên B)" (tên khoa học kèm chú thích trong ngoặc) → giữ NGUYÊN
  CẢ CỤM làm 1 span, KHÔNG tách thành 2 entity, KHÔNG chỉ lấy phần trong ngoặc.
- Nếu mô tả bằng lời thường ("X bé không có Y", "X không phát triển đều") NHƯNG có 1
  TÊN CHẨN ĐOÁN CHUẨN cụ thể tương ứng → ưu tiên gán tên chuẩn đó thay vì mô tả gộp.
- Nhiều khái niệm ngăn nhau bằng dấu phẩy / "và" / "hoặc" → TÁCH thành nhiều object
  riêng, KHÔNG gộp thành 1 span dài — NHƯNG đừng tách 1 câu MÔ TẢ liên tục (không
  phải danh sách) thành nhiều mảnh.
- THUỐC: giữ NGUYÊN cả liều + đường dùng + tần suất như trong đoạn
  (vd "aspirin 81mg po qd", "Cefuroxime 500mg PO BID") — đây KHÔNG phải cụm tối thiểu.
- TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM là HAI object riêng: tách tên xét nghiệm
  khỏi giá trị (vd "WBC" và "12.3"; "Troponin" và "0,8 ng/mL").
- Dấu hiệu sinh tồn (mạch, huyết áp, nhịp thở, spo2, nhiệt độ) KÈM SỐ ĐO → TÊN_XÉT_NGHIỆM
  + KẾT_QUẢ_XÉT_NGHIỆM (không phải TRIỆU_CHỨNG). Không kèm số (mô tả định tính như
  "nhịp tim chậm") → TRIỆU_CHỨNG.
- "text" phải khớp nguyên văn đoạn gốc để định vị lại được.
Chỉ in ra một mảng JSON, không giải thích."""

_FEWSHOT_IN = ("Bệnh nhân có tiền sử tăng huyết áp, đái tháo đường típ 2. "
               "Hiện không sốt, không ho. Chẩn đoán: suy tim do tăng huyết áp. "
               "WBC 12.3. Được cho aspirin 81mg po qd.")
_FEWSHOT_OUT = json.dumps([
    {"text": "tăng huyết áp", "type": TYPE_DIAGNOSIS, "assertions": ["isHistorical"]},
    {"text": "đái tháo đường típ 2", "type": TYPE_DIAGNOSIS, "assertions": ["isHistorical"]},
    {"text": "sốt", "type": TYPE_SYMPTOM, "assertions": ["isNegated"]},
    {"text": "ho", "type": TYPE_SYMPTOM, "assertions": ["isNegated"]},
    {"text": "suy tim", "type": TYPE_DIAGNOSIS, "assertions": []},
    {"text": "WBC", "type": TYPE_TEST_NAME, "assertions": []},
    {"text": "12.3", "type": TYPE_TEST_RESULT, "assertions": []},
    {"text": "aspirin 81mg po qd", "type": TYPE_DRUG, "assertions": []},
], ensure_ascii=False)


def _fold_system_into_user(messages: List[dict]) -> List[dict]:
    """Merge any `system` turn into the first `user` turn.

    Some chat templates (Gemma family) reject the `system` role outright.
    Folding its content into the first user message preserves the instruction
    without relying on a role the template doesn't support."""
    out: List[dict] = []
    sys_txt = ""
    for m in messages:
        if m.get("role") == "system":
            sys_txt = (sys_txt + "\n\n" + m["content"]).strip() if sys_txt else m["content"]
            continue
        if sys_txt and m.get("role") == "user":
            m = {"role": "user", "content": sys_txt + "\n\n" + m["content"]}
            sys_txt = ""
        out.append(m)
    if sys_txt:  # no user turn to fold into -> prepend as a user turn
        out.insert(0, {"role": "user", "content": sys_txt})
    return out


def build_messages(text: str, include_fewshot: bool = True, genre: str | None = None) -> List[dict]:
    """The single source of truth for the LLM prompt — used by BOTH zero-shot
    inference (below) and fine-tuning (src/model/train_llm.py), so a fine-tuned
    model always sees exactly the prompt it was trained on.

    `genre` (one of GenreType, or None) appends a short, genre-targeted rule
    supplement from `genre_prompts.py` after the base instruction — see that
    module's docstring for why this is split out instead of one mega-prompt."""
    from .genre_prompts import genre_hint
    instruction = _INSTRUCTION + genre_hint(genre)
    msgs = [{"role": "system", "content": _SYSTEM}]
    if include_fewshot:
        msgs += [
            {"role": "user", "content": instruction + "\n\nĐoạn:\n" + _FEWSHOT_IN},
            {"role": "assistant", "content": _FEWSHOT_OUT},
            {"role": "user", "content": "Đoạn:\n" + text},
        ]
    else:
        msgs += [{"role": "user", "content": instruction + "\n\nĐoạn:\n" + text}]
    return msgs


def items_to_target(items: List[dict]) -> str:
    """Serialize gold/silver items to the JSON the model should generate:
    only {text, type, assertions} (positions are relocated later; candidates
    come from the linker, not the LLM)."""
    slim = [
        {"text": it["text"], "type": it["type"],
         "assertions": list(it.get("assertions", []))}
        for it in items
        if it.get("type") in VALID_TYPES and it.get("text")
    ]
    return json.dumps(slim, ensure_ascii=False)


class LLMBackend(SpanBackend):
    name = "llm"

    def __init__(self, model_id: str, max_new_tokens: int = 3072, device: str = "auto",
                 include_fewshot: bool = None):
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "LLM backend needs torch/transformers. Install the optional block "
                f"in requirements.txt. (import error: {exc})"
            )
        # A fine-tuned checkpoint (local dir with backend_meta.json) has already
        # learned the task, so it doesn't need the few-shot example in-prompt.
        # A zero-shot base model does. Auto-detect from the checkpoint meta.
        import os
        fine_tuned = False
        if os.path.isdir(model_id):
            meta_path = os.path.join(model_id, "backend_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    fine_tuned = bool(json.load(f).get("fine_tuned", True))
        self.include_fewshot = (not fine_tuned) if include_fewshot is None else include_fewshot

        self._torch = __import__("torch")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype="auto", device_map=device,
            )
        except ValueError:
            # Multimodal families (e.g. Gemma 3n / the *-E4B "effective param"
            # models) aren't registered under AutoModelForCausalLM; they still
            # generate from text-only inputs via the image-text-to-text class.
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype="auto", device_map=device,
            )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _build_messages(self, text: str, genre: str | None = None):
        return build_messages(text, include_fewshot=self.include_fewshot, genre=genre)

    def _apply_template(self, messages: List[dict]) -> str:
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Template rejects the `system` role (Gemma etc.) -> fold it in.
            return self.tok.apply_chat_template(
                _fold_system_into_user(messages), tokenize=False,
                add_generation_prompt=True)

    def generate(self, text: str, genre: str | None = None) -> str:
        torch = self._torch
        prompt = self._apply_template(self._build_messages(text, genre=genre))
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)

    def generate_batch(self, texts: List[str], genres: List[str | None] | None = None) -> List[str]:
        """Batched generation — N prompts in ONE forward pass instead of N
        sequential `.generate()` calls. Autoregressive decoding is latency-
        bound, not compute-bound, at batch size 1 (a single-sequence call
        barely uses the GPU — see the ~20% utilization this was written to
        fix); batching lets the same wall-clock decode step advance every
        sequence in the batch at once, so throughput scales close to
        linearly with batch size until the GPU actually saturates.

        Left-padding is required for decoder-only batched generation (all
        prompts padded to the same length on the LEFT so the "next token"
        position lines up for every sequence); output is right-sliced back
        per-sample using each sample's own attention mask so padding never
        leaks into the parsed JSON."""
        torch = self._torch
        if genres is None:
            genres = [None] * len(texts)
        prompts = [
            self._apply_template(self._build_messages(t, genre=g))
            for t, g in zip(texts, genres)
        ]
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        old_side = self.tok.padding_side
        self.tok.padding_side = "left"
        try:
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        finally:
            self.tok.padding_side = old_side
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, pad_token_id=self.tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]  # uniform thanks to left-padding
        gen = out[:, prompt_len:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    @staticmethod
    def parse_items(resp: str) -> List[dict]:
        # pull the first JSON array out of the response (strip code fences etc.)
        m = re.search(r"\[.*\]", resp, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
        out = []
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and it.get("type") in VALID_TYPES and it.get("text"):
                    out.append(it)
        return out

    def predict(self, text: str) -> List[Entity]:
        from ..pipeline.genre_router import classify_text
        genre = classify_text(text)
        items = self.parse_items(self.generate(text, genre=genre))
        return relocate(text, items)

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[List[Entity]]:
        """Batch version of `predict` — chunks `texts` into groups of
        `batch_size` and runs one `generate_batch` call per chunk. Tune
        `batch_size` down if you hit an OOM (prompt length varies a lot
        note-to-note, so the worst-case padded batch can be large)."""
        from ..pipeline.genre_router import classify_text
        genres = [classify_text(t) for t in texts]
        results: List[List[Entity]] = []
        for i in range(0, len(texts), batch_size):
            chunk_texts = texts[i:i + batch_size]
            chunk_genres = genres[i:i + batch_size]
            responses = self.generate_batch(chunk_texts, chunk_genres)
            for t, resp in zip(chunk_texts, responses):
                items = self.parse_items(resp)
                results.append(relocate(t, items))
        return results
