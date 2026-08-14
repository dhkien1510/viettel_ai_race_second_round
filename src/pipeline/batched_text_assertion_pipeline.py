"""Batched two-pass Qwen orchestration for multiple documents."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from ..assertions.detect import AssertionDetector
from ..postprocess.validate import validate_entities
from ..relocate import relocate
from ..schema import Entity
from ..sections import annotate
from .context_router import ContextRouter, ContextSegment
from .qwen_tasks import (
    apply_assertion_response,
    build_assertion_messages,
    build_ner_messages,
    parse_ner_response,
)

BatchChatGenerator = Callable[[list[list[dict]]], list[str]]


@dataclass
class _SegmentState:
    document_id: str
    segment: ContextSegment
    entities: list[Entity]
    serialized: list[dict]


class BatchedTextAssertionPipeline:
    """Run all NER jobs first, then assertion jobs, in GPU-friendly batches."""

    def __init__(
        self,
        ner_generate_batch: BatchChatGenerator,
        assertion_generate_batch: BatchChatGenerator,
        router: ContextRouter | None = None,
        include_fewshot: bool = False,
        assertion_strategy: Literal["qwen", "rule"] = "qwen",
        ner_batch_size: int = 4,
        assertion_batch_size: int = 4,
    ):
        if ner_batch_size < 1 or assertion_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        self.ner_generate_batch = ner_generate_batch
        self.assertion_generate_batch = assertion_generate_batch
        self.router = router or ContextRouter()
        self.include_fewshot = include_fewshot
        self.assertion_strategy = assertion_strategy
        self.ner_batch_size = ner_batch_size
        self.assertion_batch_size = assertion_batch_size
        self.assertion_detector = AssertionDetector.load()

    @staticmethod
    def _chunks(items: list, size: int):
        for start in range(0, len(items), size):
            yield items[start:start + size]

    def process_documents(self, documents: Mapping[str, str]) -> dict[str, list[dict]]:
        segment_jobs: list[tuple[str, ContextSegment]] = []
        for document_id, text in documents.items():
            routed = self.router.route(text)
            segment_jobs.extend((document_id, segment) for segment in routed.segments)
        segment_jobs.sort(key=lambda job: len(job[1].text))

        states: list[_SegmentState] = []
        for chunk in self._chunks(segment_jobs, self.ner_batch_size):
            messages = [
                build_ner_messages(
                    segment.text,
                    segment.kind,
                    include_fewshot=self.include_fewshot,
                )
                for _, segment in chunk
            ]
            responses = self.ner_generate_batch(messages)
            if len(responses) != len(chunk):
                raise RuntimeError("NER batch generator returned the wrong response count")
            for (document_id, segment), response in zip(chunk, responses):
                entities = relocate(
                    segment.text,
                    parse_ner_response(response),
                    require_boundary=True,
                    exact_case_for_short=True,
                )
                if not entities:
                    continue
                states.append(_SegmentState(
                    document_id=document_id,
                    segment=segment,
                    entities=entities,
                    serialized=[entity.to_dict() for entity in entities],
                ))

        assertion_states = [
            state for state in states
            if self.assertion_strategy == "qwen"
            and state.segment.kind != "FAQ_EDUCATIONAL"
        ]
        assertion_states.sort(
            key=lambda state: len(state.segment.text) + len(str(state.serialized))
        )
        for chunk in self._chunks(assertion_states, self.assertion_batch_size):
            messages = [
                build_assertion_messages(
                    state.segment.text,
                    state.serialized,
                    state.segment.kind,
                    include_fewshot=self.include_fewshot,
                )
                for state in chunk
            ]
            responses = self.assertion_generate_batch(messages)
            if len(responses) != len(chunk):
                raise RuntimeError(
                    "assertion batch generator returned the wrong response count"
                )
            for state, response in zip(chunk, responses):
                state.serialized = apply_assertion_response(
                    response, state.serialized
                )

        by_document: dict[str, list[Entity]] = defaultdict(list)
        for state in states:
            for entity, updated in zip(state.entities, state.serialized):
                entity.start += state.segment.start
                entity.end += state.segment.start
                raw = documents[state.document_id]
                entity.text = raw[entity.start:entity.end]
                entity.assertions = updated.get("assertions", [])
                entity.candidates = []
                entity.source = f"qwen-batched:{state.segment.kind}"
                by_document[state.document_id].append(entity)

        output: dict[str, list[dict]] = {}
        for document_id, text in documents.items():
            entities = []
            seen = set()
            for entity in sorted(
                by_document.get(document_id, []),
                key=lambda item: (item.start, item.end, item.type),
            ):
                if entity.key() in seen:
                    continue
                seen.add(entity.key())
                entities.append(entity)

            if self.assertion_strategy == "rule":
                self.assertion_detector.apply(entities, text, annotate(text))
                for entity in entities:
                    if entity.source.endswith(":FAQ_EDUCATIONAL"):
                        entity.assertions = []

            validate_entities(entities, text)
            output[document_id] = [entity.to_dict() for entity in entities]
        return output
