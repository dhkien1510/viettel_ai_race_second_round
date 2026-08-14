"""Model-agnostic orchestration for segmented Qwen NER and assertions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ..assertions.detect import AssertionDetector
from ..postprocess.validate import validate_entities
from ..relocate import relocate
from ..schema import Entity
from ..sections import annotate
from .context_router import ContextRouter
from .qwen_tasks import (
    apply_assertion_response,
    build_assertion_messages,
    build_ner_messages,
    parse_ner_response,
)

ChatGenerator = Callable[[list[dict]], str]


class TextAssertionPipeline:
    """Run two task-specific generation passes without candidate linking.

    Generators may wrap local Transformers, vLLM, an OpenAI-compatible server,
    or two separately loaded LoRA adapters. They receive chat messages and must
    return the assistant response as a string.
    """

    def __init__(
        self,
        ner_generate: ChatGenerator,
        assertion_generate: ChatGenerator,
        router: ContextRouter | None = None,
        include_fewshot: bool = False,
        assertion_strategy: Literal["qwen", "rule"] = "qwen",
    ):
        self.ner_generate = ner_generate
        self.assertion_generate = assertion_generate
        self.router = router or ContextRouter()
        self.include_fewshot = include_fewshot
        self.assertion_strategy = assertion_strategy
        self.assertion_detector = AssertionDetector.load()

    def extract_entities(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        routed = self.router.route(text)

        for segment in routed.segments:
            ner_response = self.ner_generate(
                build_ner_messages(
                    segment.text,
                    segment.kind,
                    include_fewshot=self.include_fewshot,
                )
            )
            local_entities = relocate(
                segment.text,
                parse_ner_response(ner_response),
                require_boundary=True,
                exact_case_for_short=True,
            )
            if not local_entities:
                continue

            serialized = [entity.to_dict() for entity in local_entities]
            if self.assertion_strategy == "rule" or segment.kind == "FAQ_EDUCATIONAL":
                # Generic educational content has no patient timeline or subject.
                with_assertions = serialized
            else:
                assertion_response = self.assertion_generate(
                    build_assertion_messages(
                        segment.text,
                        serialized,
                        segment.kind,
                        include_fewshot=self.include_fewshot,
                    )
                )
                with_assertions = apply_assertion_response(
                    assertion_response, serialized
                )

            for entity, updated in zip(local_entities, with_assertions):
                entity.start += segment.start
                entity.end += segment.start
                entity.text = text[entity.start:entity.end]
                entity.assertions = updated["assertions"]
                entity.candidates = []
                entity.source = f"qwen-two-pass:{segment.kind}"
                entities.append(entity)

        # Segment/role boundaries do not overlap today, but exact dedup keeps
        # this safe when overlap chunking is added later.
        deduplicated = []
        seen = set()
        for entity in sorted(entities, key=lambda e: (e.start, e.end, e.type)):
            if entity.key() in seen:
                continue
            seen.add(entity.key())
            deduplicated.append(entity)

        if self.assertion_strategy == "rule":
            self.assertion_detector.apply(deduplicated, text, annotate(text))
            for entity in deduplicated:
                if entity.source.endswith(":FAQ_EDUCATIONAL"):
                    entity.assertions = []

        validate_entities(deduplicated, text)
        return deduplicated

    def process_text(self, text: str) -> list[dict]:
        return [entity.to_dict() for entity in self.extract_entities(text)]
