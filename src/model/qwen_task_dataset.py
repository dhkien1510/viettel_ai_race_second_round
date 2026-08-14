"""Build segmented NER/assertion chat examples from document-level labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ..pipeline.context_router import ContextRouter
from ..pipeline.qwen_tasks import build_assertion_messages, build_ner_messages
from ..schema import ASSERTABLE_TYPES, VALID_ASSERTIONS, VALID_TYPES

TaskName = Literal["ner", "assertion", "multitask"]


@dataclass(frozen=True)
class ChatExample:
    task: Literal["ner", "assertion"]
    messages: list[dict]
    target: str


def _inside_segment(items: list[dict], start: int, end: int) -> list[dict]:
    selected = []
    for item in items:
        position = item.get("position")
        if (
            item.get("type") not in VALID_TYPES
            or not item.get("text")
            or not isinstance(position, list)
            or len(position) != 2
        ):
            continue
        entity_start, entity_end = position
        if start <= entity_start and entity_end <= end:
            selected.append(item)
    return sorted(selected, key=lambda item: (item["position"][0], item["position"][1]))


def _ner_target(items: list[dict]) -> str:
    return json.dumps(
        [{"text": item["text"], "type": item["type"]} for item in items],
        ensure_ascii=False,
    )


def _assertion_target(items: list[dict]) -> str:
    def labels(item: dict) -> list[str]:
        if item["type"] not in ASSERTABLE_TYPES:
            return []
        raw = item.get("assertions") or []
        return list(dict.fromkeys(
            value for value in raw
            if isinstance(value, str) and value in VALID_ASSERTIONS
        ))

    return json.dumps(
        [
            {
                "id": index,
                "assertions": labels(item),
            }
            for index, item in enumerate(items)
        ],
        ensure_ascii=False,
    )


def build_task_examples(
    text: str,
    items: list[dict],
    task: TaskName,
    router: ContextRouter | None = None,
) -> list[ChatExample]:
    """Convert one labeled document into section-aware training examples."""
    if task not in ("ner", "assertion", "multitask"):
        raise ValueError(f"unsupported task: {task}")
    router = router or ContextRouter()
    routed = router.route(text)
    examples: list[ChatExample] = []

    for segment in routed.segments:
        segment_items = _inside_segment(items, segment.start, segment.end)
        if task in ("ner", "multitask"):
            examples.append(ChatExample(
                "ner",
                build_ner_messages(segment.text, segment.kind),
                _ner_target(segment_items),
            ))
        if task in ("assertion", "multitask") and segment_items:
            local_items = []
            for item in segment_items:
                copied = dict(item)
                copied["position"] = [
                    item["position"][0] - segment.start,
                    item["position"][1] - segment.start,
                ]
                local_items.append(copied)
            examples.append(ChatExample(
                "assertion",
                build_assertion_messages(
                    segment.text, local_items, segment.kind
                ),
                _assertion_target(local_items),
            ))
    return examples
