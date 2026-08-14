"""Final validation: hard offset + enum invariants. Raises on any violation so
a bad submission can never be written."""

from __future__ import annotations

from typing import List

from ..schema import Entity, validate_entity, validate_prediction


def validate_entities(ents: List[Entity], raw_text: str) -> None:
    for e in ents:
        validate_entity(e, raw_text)


def validate_items(items: List[dict], raw_text: str) -> None:
    validate_prediction(items, raw_text)
