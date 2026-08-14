"""Backend interface: a span producer over raw text."""

from __future__ import annotations

from typing import List

from ..schema import Entity


class SpanBackend:
    name: str = "base"

    def predict(self, text: str) -> List[Entity]:
        raise NotImplementedError
