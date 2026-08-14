"""Build confidence-aware occurrence-verification examples for Qwen.

Unlike flat NER SFT, this dataset never assumes that every unannotated medical
phrase is negative. Only reference agreement and narrow scorer-backed vetoes
receive full loss weight; disputed pseudo-labels remain visible for review.
"""

from __future__ import annotations

import re
import json
import random
from dataclasses import dataclass, asdict

from ..offsets import at_word_boundary
from ..pipeline.context_router import ContextRouter


@dataclass(frozen=True)
class VerifierExample:
    file_id: str
    genre: str
    segment_kind: str
    context: str
    context_start: int
    candidate: dict
    action: str
    target: dict | None
    confidence: str
    loss_weight: float
    evidence: str

    def as_dict(self) -> dict:
        return asdict(self)


VERIFIER_SYSTEM = """Bạn là module OCCURRENCE_VERIFIER cho NER y khoa tiếng Việt.

Bạn nhận một candidate đã có offset và context. Không được phát hiện thêm entity.
Chỉ quyết định một trong ba hành động:
- KEEP: giữ nguyên candidate.
- DROP: candidate không phải entity hợp lệ trong occurrence này.
- ADJUST: candidate đúng ý nhưng cần sửa boundary hoặc type theo target.

Không suy diễn từ surface ở file khác. Genre và segment chỉ là prior; context của
occurrence mới là bằng chứng quyết định. Chỉ trả một object JSON, không giải thích."""


def verifier_messages(row: dict) -> tuple[list[dict], str]:
    candidate = dict(row["candidate"])
    context_start = row["context_start"]
    candidate["position"] = [
        candidate["position"][0] - context_start,
        candidate["position"][1] - context_start,
    ]
    user = {
        "genre": row["genre"],
        "segment_kind": row["segment_kind"],
        "context": row["context"],
        "candidate": candidate,
    }
    action = row["action"]
    target = {"action": action}
    if action == "ADJUST" and row.get("target"):
        target["text"] = row["target"]["text"]
        target["type"] = row["target"]["type"]
    return (
        [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        json.dumps(target, ensure_ascii=False),
    )


def balance_verifier_rows(
    rows: list[dict], class_cap: int = 800, seed: int = 3407
) -> list[dict]:
    """Match the action-balanced sampling policy used for SFT and evaluation."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("loss_weight", 0) > 0:
            grouped.setdefault(row["action"], []).append(row)
    rng = random.Random(seed)
    balanced = []
    for action, action_rows in sorted(grouped.items()):
        rng.shuffle(action_rows)
        action_rows.sort(key=lambda row: row["loss_weight"], reverse=True)
        if action == "DROP":
            target = min(max(len(action_rows), class_cap // 2), class_cap)
            balanced.extend(
                action_rows[index % len(action_rows)] for index in range(target)
            )
        else:
            balanced.extend(action_rows[:class_cap])
    return balanced


_PHYSIOLOGICAL = re.compile(r"^(?:có thai|mang thai|mãn kinh)$", re.I)
_GENERIC_TEST = {
    "xét nghiệm",
    "khám chuyên khoa",
    "khám lâm sàng",
    "chẩn đoán hình ảnh",
    "nhìn bên ngoài",
}


def key(item: dict) -> tuple:
    return (*item["position"], item["type"], item["text"])


def overlap(left: dict, right: dict) -> int:
    if left["type"] != right["type"]:
        return 0
    return max(
        0,
        min(left["position"][1], right["position"][1])
        - max(left["position"][0], right["position"][0]),
    )


def confirmed_drop_reason(item: dict, text: str, segment_kind: str) -> str | None:
    surface = " ".join(item["text"].casefold().split())
    stripped = item["text"].strip()
    stars = stripped.count("*")
    if stripped and stars / len(stripped) >= 0.5:
        return "masked-name-veto"
    if item["type"] == "CHẨN_ĐOÁN" and _PHYSIOLOGICAL.fullmatch(surface):
        return "physiological-state-veto"
    if item["type"] == "TÊN_XÉT_NGHIỆM" and surface in _GENERIC_TEST:
        # Legacy structured EHR headings are a separate annotation profile.
        if not (surface == "chẩn đoán hình ảnh" and segment_kind == "EHR_LAB"):
            return "bare-generic-test-veto"
    start, end = item["position"]
    if len(stripped) <= 2 and not at_word_boundary(text, start, end):
        return "short-span-inside-token-veto"
    return None


def _segment_for(router: ContextRouter, text: str, start: int):
    routed = router.route(text)
    for segment in routed.segments:
        if segment.start <= start < segment.end:
            return routed.genre, segment
    return routed.genre, routed.segments[0]


def _make_example(
    file_id: str,
    text: str,
    item: dict,
    action: str,
    target: dict | None,
    confidence: str,
    loss_weight: float,
    evidence: str,
    router: ContextRouter,
) -> VerifierExample:
    genre, segment = _segment_for(router, text, item["position"][0])
    start = max(segment.start, item["position"][0] - 220)
    end = min(segment.end, item["position"][1] + 220)
    candidate = {
        "text": item["text"],
        "position": item["position"],
        "type": item["type"],
    }
    slim_target = None if target is None else {
        "text": target["text"],
        "position": target["position"],
        "type": target["type"],
    }
    return VerifierExample(
        file_id=file_id,
        genre=genre,
        segment_kind=segment.kind,
        context=text[start:end],
        context_start=start,
        candidate=candidate,
        action=action,
        target=slim_target,
        confidence=confidence,
        loss_weight=loss_weight,
        evidence=evidence,
    )


def build_inference_row(
    file_id: str,
    text: str,
    item: dict,
    evidence: str,
    router: ContextRouter | None = None,
) -> dict:
    """Build the same occurrence context used during verifier training."""
    router = router or ContextRouter()
    return _make_example(
        file_id=file_id,
        text=text,
        item=item,
        action="REVIEW",
        target=None,
        confidence="unknown",
        loss_weight=0.0,
        evidence=evidence,
        router=router,
    ).as_dict()


def build_verifier_examples(
    file_id: str,
    text: str,
    reference_100: list[dict],
    reference_146: list[dict],
    model_predictions: list[dict],
    router: ContextRouter | None = None,
) -> list[VerifierExample]:
    router = router or ContextRouter()
    left = {key(item): item for item in reference_100}
    right = {key(item): item for item in reference_146}
    model = {key(item): item for item in model_predictions}
    common = left.keys() & right.keys()
    examples = [
        _make_example(
            file_id, text, right[item_key], "KEEP", right[item_key],
            "high", 1.0, "exact-reference-agreement", router,
        )
        for item_key in sorted(common)
    ]

    # Boundary corruption supplies diverse ADJUST examples without declaring
    # naturally unannotated phrases to be negatives. The full target must be
    # present in the same context, so a standalone unseen term remains KEEP-able.
    for item_key in sorted(common):
        item = right[item_key]
        token_matches = list(re.finditer(r"\S+", item["text"]))
        if len(token_matches) < 3:
            continue
        variants = [
            (
                item["position"][0] + token_matches[1].start(),
                item["position"][1],
            ),
            (
                item["position"][0],
                item["position"][0] + token_matches[-1].start(),
            ),
        ]
        for start, end in variants:
            surface = text[start:end].strip()
            start += len(text[start:end]) - len(text[start:end].lstrip())
            end = start + len(surface)
            variant = {
                "text": surface,
                "position": [start, end],
                "type": item["type"],
            }
            if not surface or key(variant) in left or key(variant) in right:
                continue
            examples.append(_make_example(
                file_id, text, variant, "ADJUST", item,
                "low", 0.25, "synthetic-boundary-corruption", router,
            ))

    left_only = [left[item_key] for item_key in left.keys() - common]
    right_only = [right[item_key] for item_key in right.keys() - common]
    used_right = set()
    for item in left_only:
        candidates = [
            (overlap(item, other), index, other)
            for index, other in enumerate(right_only)
            if index not in used_right and overlap(item, other) > 0
        ]
        if candidates:
            _, index, target = max(candidates)
            used_right.add(index)
            examples.append(_make_example(
                file_id, text, item, "ADJUST", target,
                "medium", 0.5, "reference-boundary-disagreement", router,
            ))
        else:
            examples.append(_make_example(
                file_id, text, item, "DROP", None,
                "medium", 0.5, "removed-in-refinement", router,
            ))
    for index, item in enumerate(right_only):
        if index not in used_right:
            examples.append(_make_example(
                file_id, text, item, "KEEP", item,
                "medium", 0.5, "added-in-refinement", router,
            ))

    known = left.keys() | right.keys()
    for item_key in model.keys() - known:
        item = model[item_key]
        _, segment = _segment_for(router, text, item["position"][0])
        reason = confirmed_drop_reason(item, text, segment.kind)
        if reason:
            examples.append(_make_example(
                file_id, text, item, "DROP", None,
                "high", 1.0, reason, router,
            ))
        else:
            examples.append(_make_example(
                file_id, text, item, "REVIEW", None,
                "unknown", 0.0, "model-only-unlabeled-not-negative", router,
            ))
    return examples
