"""Run a fine-tuned NER model over raw text and return Entity spans with exact
character offsets. Fast tokenizers (xlmr, mdeberta, ...) use the offset
mapping directly; slow tokenizers (PhoBERT/ViHealthBERT, no fast variant) go
through the word-level path in `wordlevel_encode.py` instead (word granularity
offsets, matching how they were trained).

SCAFFOLDING. Requires torch + transformers. Designed to be OR-merged with the
dictionary extractors in infer.py (the conflict resolver dedups/keeps longest).
"""

from __future__ import annotations

import json
import os
from typing import List

from ..schema import Entity, VALID_TYPES
from . import wordlevel_encode as wl
from .word_pool import collapse_to_word_starts, group_word_tags_into_entities


class NERModel:
    def __init__(self, model_dir: str, max_len: int = 256, stride: int = 64):
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "NER inference needs torch/transformers. Install the optional "
                f"block in requirements.txt. (import error: {exc})"
            )
        # a trained checkpoint is a LOCAL folder; guard against accidentally
        # hitting the HF hub for a path that just doesn't exist yet
        if not os.path.isdir(model_dir):
            raise SystemExit(
                f"NER checkpoint folder not found: {model_dir!r}\n"
                f"Train one first:  python -m src.model.train_ner --out {model_dir}\n"
                f"(or omit --model to run the rule-only pipeline)."
            )
        self._torch = __import__("torch")
        torch = self._torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        meta_path = os.path.join(model_dir, "backend_meta.json")
        self.use_crf = False
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.use_crf = bool(json.load(f).get("crf", False))

        if self.use_crf:
            from .crf_model import EncoderCRFForTokenClassification
            self.model = EncoderCRFForTokenClassification.from_pretrained(model_dir)
            self.id2label = self.model.id2label
            model_config = self.model.encoder.config
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            self.model.eval()
            self.id2label = self.model.config.id2label
            model_config = self.model.config
        self.model = self.model.to(self.device)

        # clamp window to the backbone's position limit (roberta reserves 2)
        model_max = getattr(model_config, "max_position_embeddings", 512) or 512
        self.max_len = max(8, min(max_len, model_max - 2))
        self.stride = min(stride, self.max_len // 2)

    def predict(self, text: str) -> List[Entity]:
        if self.use_crf:
            return self._predict_crf(text)
        if not self.tok.is_fast:
            return self._predict_wordlevel(text)
        return self._predict_fast(text)

    def _predict_crf(self, text: str) -> List[Entity]:
        torch = self._torch
        if self.tok.is_fast:
            enc = self.tok(
                text, return_offsets_mapping=True, truncation=True,
                max_length=self.max_len, stride=self.stride,
                return_overflowing_tokens=True, return_tensors="pt", padding=True,
            )
            offsets = enc.pop("offset_mapping")
            n_windows = offsets.shape[0]
            wids_batch, spans_per_window = [], []
            for wi in range(n_windows):
                wids_out, first_pos = collapse_to_word_starts(enc.word_ids(wi))
                spans_per_window.append(
                    [(int(offsets[wi][t][0]), int(offsets[wi][t][1])) for t in first_pos]
                )
                wids_batch.append(wids_out)
            enc.pop("overflow_to_sample_mapping", None)
            word_ids_tensor = torch.tensor(wids_batch).to(self.device)
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            decoded, _ = self.model.decode(input_ids, attn, word_ids_tensor)
            ents: List[Entity] = []
            for wi in range(n_windows):
                for typ, s, e in group_word_tags_into_entities(
                        self.id2label, text, spans_per_window[wi], decoded[wi]):
                    ents.append(Entity(text[s:e], s, e, typ, source="model"))
            return ents

        # word-level (slow tokenizer) path
        spans = wl.word_spans(text)
        words = [text[s:e] for s, e in spans]
        if not words:
            return []
        windows = wl.encode_words(self.tok, words, self.max_len, self.stride)
        ents: List[Entity] = []
        for w in windows:
            wids_out, first_pos = collapse_to_word_starts(w["word_ids"])
            wspans = [spans[w["word_ids"][t]] for t in first_pos]
            input_ids = torch.tensor([w["input_ids"]]).to(self.device)
            attn = torch.tensor([w["attention_mask"]]).to(self.device)
            word_ids_tensor = torch.tensor([wids_out]).to(self.device)
            decoded, _ = self.model.decode(input_ids, attn, word_ids_tensor)
            for typ, s, e in group_word_tags_into_entities(self.id2label, text, wspans, decoded[0]):
                ents.append(Entity(text[s:e], s, e, typ, source="model"))
        return ents

    def _predict_wordlevel(self, text: str) -> List[Entity]:
        torch = self._torch
        spans = wl.word_spans(text)
        words = [text[s:e] for s, e in spans]
        if not words:
            return []
        windows = wl.encode_words(self.tok, words, self.max_len, self.stride)
        ents: List[Entity] = []
        for w in windows:
            input_ids = torch.tensor([w["input_ids"]]).to(self.device)
            attn = torch.tensor([w["attention_mask"]]).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            pred = logits.argmax(-1)[0]
            cur = None  # (type, start, end)
            for t, wid in enumerate(w["word_ids"]):
                if wid is None:                 # continuation subtoken / special / pad
                    continue
                label = self.id2label[int(pred[t])]
                if label == "O":
                    if cur:
                        ents.append(self._mk(text, cur))
                        cur = None
                    continue
                bio, typ = label.split("-", 1)
                if typ not in VALID_TYPES:
                    continue
                s, e = spans[wid]
                # a newline between the current entity and this word means the
                # model tagged an unrelated bullet/line as a same-type
                # continuation -> close the old entity instead of merging
                # across lines (see xlmr_base_0607_1300 audit: "13.9\n-
                # 80 neutrophil\n-...").
                gap_has_newline = cur is not None and "\n" in text[cur[2]:s]
                if bio == "B" or cur is None or cur[0] != typ or gap_has_newline:
                    if cur:
                        ents.append(self._mk(text, cur))
                    cur = [typ, s, e]
                else:
                    cur[2] = e
            if cur:
                ents.append(self._mk(text, cur))
        return ents

    def _predict_fast(self, text: str) -> List[Entity]:
        torch = self._torch
        enc = self.tok(
            text, return_offsets_mapping=True, truncation=True,
            max_length=self.max_len, stride=self.stride,
            return_overflowing_tokens=True, return_tensors="pt", padding=True,
        )
        offsets = enc.pop("offset_mapping")
        n_windows = offsets.shape[0]
        # a word can split into several BPE subwords ("Viêm" -> "Vi"+"êm",
        # "PICC" -> "PI"+"CC"); only the FIRST subword's predicted label is
        # trusted, later subwords just extend the span — otherwise a
        # continuation subword predicting a different (or "O") label truncates
        # or re-types the word mid-way (verified on output/xlmr_base_0607_1300:
        # "Viêm" -> entity text "Vi", "PICC" -> two entities "PI"+"CC" of
        # different types).
        word_ids_per_window = [enc.word_ids(w) for w in range(n_windows)]
        enc.pop("overflow_to_sample_mapping", None)
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        pred = logits.argmax(-1)

        ents: List[Entity] = []
        for w in range(pred.shape[0]):
            cur = None  # (type, start, end)
            last_word_id = None
            word_ids = word_ids_per_window[w]
            for t in range(pred.shape[1]):
                a, b = int(offsets[w][t][0]), int(offsets[w][t][1])
                if a == b:
                    continue
                wid = word_ids[t]
                if wid == last_word_id:
                    if cur:
                        cur[2] = b
                    continue
                last_word_id = wid
                label = self.id2label[int(pred[w][t])]
                if label == "O":
                    if cur:
                        ents.append(self._mk(text, cur))
                        cur = None
                    continue
                bio, typ = label.split("-", 1)
                if typ not in VALID_TYPES:
                    continue
                # a newline between the current entity and this word means the
                # model tagged an unrelated bullet/line as a same-type
                # continuation -> close the old entity instead of merging
                # across lines.
                gap_has_newline = cur is not None and "\n" in text[cur[2]:a]
                if bio == "B" or cur is None or cur[0] != typ or gap_has_newline:
                    if cur:
                        ents.append(self._mk(text, cur))
                    cur = [typ, a, b]
                else:
                    cur[2] = b
            if cur:
                ents.append(self._mk(text, cur))
        return ents

    @staticmethod
    def _mk(text: str, cur) -> Entity:
        typ, s, e = cur
        return Entity(text[s:e], s, e, typ, source="model")
