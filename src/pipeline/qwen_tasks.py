"""Task-specific zero/few-shot prompts for text/type and assertions."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable

from ..schema import ASSERTABLE_TYPES, VALID_ASSERTIONS, VALID_TYPES
from .context_router import SegmentKind


_NER_POLICY: dict[SegmentKind, str] = {
    "FAQ_EDUCATIONAL": (
        "Đây là nội dung FAQ/giáo dục. Giữ tên bệnh độc lập và tên khái niệm đang "
        "được nói tới trực tiếp; bỏ cơ chế sinh lý bệnh, kết cục chung, mô tả chức "
        "năng và danh sách giả định không gắn với một ca bệnh cụ thể."
    ),
    "QA_USER": (
        "Đây là lời của người hỏi. Giữ triệu chứng thật, thuốc đã dùng và xét "
        "nghiệm đã làm. Không gán tên bệnh mà người hỏi chỉ tự suy đoán trong câu "
        "'có phải tôi bị X không' nếu chưa có nhận định y khoa xác nhận."
    ),
    "QA_CLINICIAN": (
        "Đây là lời bác sĩ. Nhận định như 'có thể', 'khả năng', 'nghĩ nhiều' vẫn "
        "có thể là CHẨN_ĐOÁN. Xét nghiệm được khuyên làm và thuốc đã được chỉ định "
        "vẫn được gán."
    ),
    "EHR_HISTORY": (
        "Đây là mục tiền sử, bệnh nền hoặc thuốc trước nhập viện. Trích xuất entity "
        "bình thường; trạng thái lịch sử sẽ do module assertion xử lý."
    ),
    "EHR_CURRENT": (
        "Đây là đợt bệnh hiện tại. Không coi một sự kiện là tiền sử chỉ vì câu dùng "
        "thì quá khứ hoặc có từ 'đã'."
    ),
    "EHR_LAB": (
        "Đây là xét nghiệm/chẩn đoán hình ảnh. Tách tên phép đo và kết quả. Finding "
        "sau 'cho thấy', 'ghi nhận', 'phát hiện' thường là KẾT_QUẢ_XÉT_NGHIỆM, "
        "kể cả khi finding mang hình thức tên bệnh."
    ),
    "EHR_DIAGNOSIS": (
        "Đây là mục chẩn đoán/đánh giá. Ưu tiên tên bệnh, hội chứng hoặc chấn thương "
        "độc lập là CHẨN_ĐOÁN."
    ),
    "EHR_OTHER": "Đây là mục EHR khác; quyết định theo occurrence và động từ dẫn.",
    "FREE_TEXT": "Không có section đủ tin cậy; quyết định theo occurrence và câu lân cận.",
}

_NER_SYSTEM = """Bạn là module NER_TEXT trong hệ thống gán nhãn y khoa tiếng Việt.

NHIỆM VỤ DUY NHẤT: phát hiện occurrence của khái niệm y tế và trả về một mảng JSON.
Mỗi object chỉ có đúng hai trường:
{"text":"cụm nguyên văn","type":"MỘT_NHÃN"}

Nhãn hợp lệ:
- TRIỆU_CHỨNG: biểu hiện chủ quan hoặc dấu hiệu lâm sàng cụ thể.
- CHẨN_ĐOÁN: tên bệnh, hội chứng, chấn thương hoặc tình trạng bệnh lý độc lập.
- THUỐC: tên thuốc, hoạt chất, chế phẩm hoặc nhóm thuốc cụ thể.
- TÊN_XÉT_NGHIỆM: tên phép đo, xét nghiệm, chẩn đoán hình ảnh hoặc thăm dò.
- KẾT_QUẢ_XÉT_NGHIỆM: giá trị, kết luận hoặc finding do phép đo/thăm dò tạo ra.

QUY TẮC CỐT LÕI:
1. Chỉ giữ occurrence đang hoạt động như một khái niệm lâm sàng độc lập trong
   chính ngữ cảnh đó; không gán chỉ vì cụm từ nghe có vẻ y khoa.
2. Copy text y hệt văn bản: không sửa lỗi, không dịch, không chuẩn hóa.
3. Lấy span ngắn gọn nhất nhưng đầy đủ tên khái niệm. Bỏ heading, động từ dẫn,
   thời gian, mức độ và diễn biến nếu chúng không thuộc tên.
4. Giữ qualifier định danh như "típ 2", "mạn tính", "giai đoạn IV", "không đặc
   hiệu", "do rượu" khi nó thuộc tên bệnh.
5. Tách các khái niệm độc lập trong danh sách. Không gộp tên xét nghiệm với kết quả.
6. Với thuốc, giữ liều, đường dùng và tần suất nếu được viết liền trong cụm thuốc.
7. Sinh hiệu kèm số đo: tên sinh hiệu là TÊN_XÉT_NGHIỆM, số đo là
   KẾT_QUẢ_XÉT_NGHIỆM. Mô tả định tính không có số như "nhịp tim chậm" là
   TRIỆU_CHỨNG.
8. Không gán trạng thái sinh lý bình thường, cơ chế/định nghĩa thuần giáo khoa,
   kết cục chung, heading, từ quá chung, chuỗi thuốc bị che bằng *** hoặc xxx.
9. Mỗi occurrence hợp lệ là một object riêng. Không tự động chuyển nhãn theo từ
   điển hoặc theo occurrence ở file khác.
10. Không sinh position, assertions, candidates hay lời giải thích. Chỉ trả JSON."""

_ASSERTION_SYSTEM = """Bạn là module ASSERTION_LINKER trong hệ thống gán nhãn y khoa.

Entity đã được module NER cố định. Bạn KHÔNG được thêm, xóa, sửa text hoặc type.
Trường position là offset cục bộ chỉ để phân biệt các occurrence trùng text;
không được sao chép position vào output hoặc thay đổi entity.
Với mỗi entity, chỉ trả về:
{"id":ID,"assertions":[]}

Nhãn hợp lệ:
- isNegated: entity bị phủ định trong đúng scope của nó.
- isFamily: entity thuộc người nhà/họ hàng, không thuộc bệnh nhân.
- isHistorical: tiền sử, bệnh nền, thuốc dùng trước đó hoặc sự kiện đã kết thúc
  được trình bày như tiền sử; không phải diễn biến của đợt hiện tại.

QUY TẮC:
1. Chỉ TRIỆU_CHỨNG, CHẨN_ĐOÁN và THUỐC có thể mang assertion.
2. TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM luôn có assertions [].
3. "không đặc hiệu", "không điển hình", "không tế bào nhỏ", "không cản quang"
   có thể là qualifier/đặc tả kỹ thuật, không phải phủ định.
4. "Tiền sử bệnh hiện tại" và "Bệnh sử hiện tại" là đợt hiện tại, không tự động
   là isHistorical. Thì quá khứ hoặc từ "đã" một mình cũng không đủ.
5. Người nhà chỉ kể lại tình trạng của bệnh nhân không tạo isFamily.
6. Một entity có thể có nhiều assertion nếu ngữ cảnh thật sự thỏa mãn.
7. Khi không đủ bằng chứng, dùng []. Trả đủ mọi ID và chỉ trả mảng JSON."""


def _ner_user(text: str, kind: SegmentKind) -> str:
    policy = _NER_POLICY.get(kind, _NER_POLICY["FREE_TEXT"])
    return f"LOẠI ĐOẠN: {kind}\nCHÍNH SÁCH: {policy}\n\nVĂN BẢN:\n{text}"


def _ner_fewshot(kind: SegmentKind) -> tuple[str, list[dict]]:
    examples: dict[SegmentKind, tuple[str, list[dict]]] = {
        "FAQ_EDUCATIONAL": (
            "Thiếu men G6PD là một bệnh di truyền. Khi thiếu men này, hồng cầu dễ bị phá hủy.",
            [{"text": "Thiếu men G6PD", "type": "CHẨN_ĐOÁN"}],
        ),
        "QA_USER": (
            "Hỏi: Tôi đau đầu hai ngày nay và đã uống paracetamol. Tôi có phải bị u não không?",
            [
                {"text": "đau đầu", "type": "TRIỆU_CHỨNG"},
                {"text": "paracetamol", "type": "THUỐC"},
            ],
        ),
        "QA_CLINICIAN": (
            "Trả lời: Có thể bạn bị đau nửa đầu. Bạn nên chụp MRI sọ não.",
            [
                {"text": "đau nửa đầu", "type": "CHẨN_ĐOÁN"},
                {"text": "chụp MRI sọ não", "type": "TÊN_XÉT_NGHIỆM"},
            ],
        ),
        "EHR_HISTORY": (
            "Tiền sử: tăng huyết áp, từng dùng aspirin 81 mg mỗi ngày.",
            [
                {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"},
                {"text": "aspirin 81 mg mỗi ngày", "type": "THUỐC"},
            ],
        ),
        "EHR_CURRENT": (
            "Triệu chứng hiện tại: sốt, ho đờm xanh và không đau ngực.",
            [
                {"text": "sốt", "type": "TRIỆU_CHỨNG"},
                {"text": "ho đờm xanh", "type": "TRIỆU_CHỨNG"},
                {"text": "đau ngực", "type": "TRIỆU_CHỨNG"},
            ],
        ),
        "EHR_LAB": (
            "CT ngực cho thấy tràn dịch màng phổi. WBC: 14,3.",
            [
                {"text": "CT ngực", "type": "TÊN_XÉT_NGHIỆM"},
                {"text": "tràn dịch màng phổi", "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
                {"text": "WBC", "type": "TÊN_XÉT_NGHIỆM"},
                {"text": "14,3", "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
            ],
        ),
        "EHR_DIAGNOSIS": (
            "Chẩn đoán: viêm phổi cộng đồng.",
            [{"text": "viêm phổi cộng đồng", "type": "CHẨN_ĐOÁN"}],
        ),
    }
    return examples.get(kind, (
        "Bệnh nhân khó thở. Huyết áp: 150/90 mmHg.",
        [
            {"text": "khó thở", "type": "TRIỆU_CHỨNG"},
            {"text": "Huyết áp", "type": "TÊN_XÉT_NGHIỆM"},
            {"text": "150/90 mmHg", "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
        ],
    ))


def build_ner_messages(
    text: str,
    kind: SegmentKind = "FREE_TEXT",
    include_fewshot: bool = False,
) -> list[dict]:
    messages = [{"role": "system", "content": _NER_SYSTEM}]
    if include_fewshot:
        example_text, example_output = _ner_fewshot(kind)
        messages.extend([
            {"role": "user", "content": _ner_user(example_text, kind)},
            {
                "role": "assistant",
                "content": json.dumps(example_output, ensure_ascii=False),
            },
        ])
    messages.append({"role": "user", "content": _ner_user(text, kind)})
    return messages


def _assertion_user(
    text: str,
    entities: list[dict],
    kind: SegmentKind = "FREE_TEXT",
) -> str:
    policy = {
        "FAQ_EDUCATIONAL": (
            "Nội dung giáo dục/FAQ không phải tiền sử của một bệnh nhân. "
            "Mặc định mọi assertion là []; không gán isHistorical cho bệnh, "
            "biểu hiện hoặc biến chứng được mô tả chung."
        ),
        "EHR_HISTORY": (
            "Đây là mục tiền sử/bệnh nền/thuốc trước đây. Gán isHistorical cho "
            "entity của bệnh nhân; vẫn phân biệt entity thuộc người nhà."
        ),
        "EHR_CURRENT": (
            "Đây là đợt hiện tại. Không gán isHistorical chỉ vì có từ 'đã', "
            "'trước nhập viện' hoặc động từ quá khứ trong diễn biến hiện tại."
        ),
        "QA_USER": (
            "Đây là lời người hỏi về ca hiện tại. Thuốc đã dùng hoặc triệu chứng "
            "đã xuất hiện không tự động là isHistorical."
        ),
        "QA_CLINICIAN": (
            "Đây là lời tư vấn của bác sĩ; nhận định và khuyến nghị hiện tại "
            "không phải tiền sử."
        ),
    }.get(kind, "Chỉ gán assertion khi context có bằng chứng trực tiếp.")
    return (
        f"LOẠI ĐOẠN: {kind}\nCHÍNH SÁCH: {policy}\n\nCONTEXT:\n" +
        text + "\n\nENTITIES:\n" +
        json.dumps(entities, ensure_ascii=False)
    )


def build_assertion_messages(
    text: str,
    entities: Iterable[dict],
    kind: SegmentKind = "FREE_TEXT",
    include_fewshot: bool = False,
) -> list[dict]:
    slim = []
    for index, entity in enumerate(entities):
        typ = entity.get("type")
        if typ in VALID_TYPES and entity.get("text"):
            item = {"id": index, "text": entity["text"], "type": typ}
            position = entity.get("position")
            if (
                isinstance(position, list)
                and len(position) == 2
                and all(isinstance(value, int) for value in position)
            ):
                item["position"] = position
            slim.append(item)

    messages = [{"role": "system", "content": _ASSERTION_SYSTEM}]
    if include_fewshot:
        example_text = (
            "Tiền sử: tăng huyết áp. Hiện không sốt. Mẹ bệnh nhân bị đái tháo đường. "
            "WBC: 12,3."
        )
        example_entities = [
            {"id": 0, "text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"},
            {"id": 1, "text": "sốt", "type": "TRIỆU_CHỨNG"},
            {"id": 2, "text": "đái tháo đường", "type": "CHẨN_ĐOÁN"},
            {"id": 3, "text": "WBC", "type": "TÊN_XÉT_NGHIỆM"},
            {"id": 4, "text": "12,3", "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
        ]
        example_output = [
            {"id": 0, "assertions": ["isHistorical"]},
            {"id": 1, "assertions": ["isNegated"]},
            {"id": 2, "assertions": ["isFamily"]},
            {"id": 3, "assertions": []},
            {"id": 4, "assertions": []},
        ]
        messages.extend([
            {
                "role": "user",
                "content": _assertion_user(
                    example_text, example_entities, "FREE_TEXT"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(example_output, ensure_ascii=False),
            },
        ])
    messages.append({
        "role": "user",
        "content": _assertion_user(text, slim, kind),
    })
    return messages


def _json_array(response: str):
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except (TypeError, json.JSONDecodeError):
            pass

    # Small models sometimes emit several one-object arrays or repeat until
    # max_new_tokens cuts the final array. Recover only complete flat objects;
    # schema/type filtering and exact relocation still run afterward.
    recovered = []
    for raw_object in re.findall(r"\{[^{}]*\}", response, re.DOTALL):
        try:
            item = json.loads(raw_object)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            recovered.append(item)
    return recovered or None


def _canonical_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFD", value.upper().strip())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = normalized.replace("Đ", "D")
    normalized = re.sub(r"[^A-Z]+", "_", normalized).strip("_")
    return {
        "TRIEU_CHUNG": "TRIỆU_CHỨNG",
        "CHAN_DOAN": "CHẨN_ĐOÁN",
        "THUOC": "THUỐC",
        "TEN_XET_NGHIEM": "TÊN_XÉT_NGHIỆM",
        "KET_QUA_XET_NGHIEM": "KẾT_QUẢ_XÉT_NGHIỆM",
    }.get(normalized)


def parse_ner_response(response: str) -> list[dict]:
    """Parse strict text/type output and discard fields owned by other stages."""
    data = _json_array(response)
    if data is None:
        return []
    output = []
    for item in data:
        typ = _canonical_type(item.get("type")) if isinstance(item, dict) else None
        if (
            isinstance(item, dict)
            and typ is not None
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ):
            output.append({"text": item["text"], "type": typ})
    return output


def parse_assertion_response(response: str, entities: list[dict]) -> dict[int, list[str]]:
    """Parse ID assertions without allowing any entity mutation."""
    data = _json_array(response)
    if data is None:
        return {}
    parsed: dict[int, list[str]] = {}
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            continue
        entity_id = item["id"]
        if not 0 <= entity_id < len(entities):
            continue
        typ = entities[entity_id].get("type")
        raw = item.get("assertions")
        if typ not in ASSERTABLE_TYPES or not isinstance(raw, list):
            parsed[entity_id] = []
            continue
        parsed[entity_id] = list(dict.fromkeys(
            value for value in raw if isinstance(value, str) and value in VALID_ASSERTIONS
        ))
    return parsed


def apply_assertion_response(response: str, entities: list[dict]) -> list[dict]:
    assertions = parse_assertion_response(response, entities)
    output = []
    for index, entity in enumerate(entities):
        copied = dict(entity)
        copied["assertions"] = assertions.get(index, [])
        output.append(copied)
    return output
