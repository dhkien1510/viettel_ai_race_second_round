"""Encoder NER backend (groups A & B).

Wraps a trained token-classification checkpoint. For word-level backbones
(PhoBERT/ViHealthBERT, `word_segment=True`) the text is segmented with
underthesea, the model predicts on the segmented string, and spans are mapped
back to raw character offsets — so offsets stay exact regardless of backbone.

Reads `backend_meta.json` (written by train_ner) from the checkpoint dir to
know whether the model was trained word-level.
"""

from __future__ import annotations

import json
import os
from typing import List

from ..schema import Entity
from ..model.infer_ner import NERModel
from .. import wordseg
from .base import SpanBackend


class EncoderBackend(SpanBackend):
    name = "encoder"

    def __init__(self, model_dir: str, word_segment: bool = None,
                 max_len: int = 256, stride: int = 64):
        meta_path = os.path.join(model_dir, "backend_meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        self.word_segment = meta.get("word_segment", False) if word_segment is None else word_segment
        self.model = NERModel(model_dir, max_len=max_len, stride=stride)

    def predict(self, text: str) -> List[Entity]:
        if not self.word_segment:
            return self.model.predict(text)
        # word-level: segment, predict on segmented text, map spans back to raw
        seg, seg2raw = wordseg.segment_with_offsets(text)
        spans = self.model.predict(seg)
        out: List[Entity] = []
        for e in spans:
            raw_span = wordseg.seg_span_to_raw(seg2raw, e.start, e.end)
            if raw_span is None:
                continue
            s, en = raw_span
            out.append(Entity(text[s:en], s, en, e.type, source="encoder:wordseg"))
        return out
