"""Encoder + word-level CRF token classifier.

Replaces AutoModelForTokenClassification's linear-head + independent-softmax-
per-token setup with: encoder -> pool each word's FIRST subword hidden state
-> linear head -> LinearChainCRF (crf.py). The CRF learns a transition matrix
between BIO labels, so at inference it can no longer produce an ungrammatical
sequence like a bare I-CHẨN_ĐOÁN with no preceding B- (the boundary/split bugs
this was written to fix — see `git log` around 2026-07-06 for the concrete
audit that motivated it).

Word-level pooling (not subword-level) is required because the CRF's forward
algorithm/Viterbi need a clean, right-padded mask — see word_pool.py's
docstring for why subword-level label arrays don't have that shape.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, Trainer
from transformers.modeling_outputs import TokenClassifierOutput

from .crf import LinearChainCRF

_STATE_FILE = "crf_model.pt"
_META_FILE = "crf_labels.json"


class EncoderCRFForTokenClassification(nn.Module):
    def __init__(self, model_id: str, num_labels: int, id2label: dict, label2id: dict):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = LinearChainCRF(num_labels)
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def _pool_words(self, hidden: torch.Tensor, word_ids: torch.Tensor):
        """hidden: (B, T, H). word_ids: (B, T), -1 at non-word-start positions
        (see word_pool.collapse_to_word_starts). Returns word_hidden (B, T, H)
        zero-padded and word_mask (B, T) bool, always padded out to T (=
        hidden.size(1), the FIXED subword max_len — an upper bound on word
        count) rather than the max real word count actually seen in this
        batch. Padding to a batch-local dynamic width breaks multi-GPU
        DataParallel: each GPU gets a different sub-batch with a different
        real word-count max, so replicas return differently-shaped emissions
        and DataParallel's gather() crashes trying to concatenate them
        (`RuntimeError: Input tensor at index 1 has invalid shape [...]`,
        hit on a real 2-GPU Kaggle run). Padding to the fixed T sidesteps this
        — every replica always returns the same (sub_batch, T, H) shape."""
        seq_len = hidden.size(1)
        per_example = []
        for b in range(hidden.size(0)):
            sel = word_ids[b] != -1
            per_example.append(hidden[b][sel])
        lengths = torch.tensor([t.size(0) for t in per_example], device=hidden.device)
        word_hidden = pad_sequence(per_example, batch_first=True)  # (B, W_local<=seq_len, H)
        pad_amount = seq_len - word_hidden.size(1)
        if pad_amount > 0:
            word_hidden = torch.nn.functional.pad(word_hidden, (0, 0, 0, pad_amount))
        word_mask = torch.arange(seq_len, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
        return word_hidden, word_mask

    def forward(self, input_ids=None, attention_mask=None, word_ids=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)
        word_hidden, word_mask = self._pool_words(hidden, word_ids)
        emissions = self.classifier(word_hidden)  # (B, W, C)

        loss = None
        if labels is not None:
            w_max = emissions.size(1)
            word_labels = labels[:, :w_max].clone()
            word_labels[word_labels < 0] = 0  # dummy valid index at padded
                                               # slots; excluded via word_mask
            loss = self.crf(emissions, word_labels, word_mask)

        return TokenClassifierOutput(loss=loss, logits=emissions)

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, word_ids):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        word_hidden, word_mask = self._pool_words(out.last_hidden_state, word_ids)
        emissions = self.classifier(word_hidden)
        return self.crf.decode(emissions, word_mask), word_mask

    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, _STATE_FILE))
        with open(os.path.join(out_dir, _META_FILE), "w", encoding="utf-8") as f:
            json.dump({
                "base_model": self.encoder.config._name_or_path,
                "num_labels": self.num_labels,
                "id2label": self.id2label,
                "label2id": self.label2id,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, out_dir: str) -> "EncoderCRFForTokenClassification":
        with open(os.path.join(out_dir, _META_FILE), "r", encoding="utf-8") as f:
            meta = json.load(f)
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        model = cls(meta["base_model"], meta["num_labels"], id2label, meta["label2id"])
        state = torch.load(os.path.join(out_dir, _STATE_FILE), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model


class CRFTrainer(Trainer):
    """Overrides eval-time prediction to decode via CRF Viterbi instead of
    per-token argmax (what the base Trainer would do with raw emissions —
    exactly the independent-per-token decision this whole layer exists to
    avoid). Runs the encoder a second time during eval to get the decode;
    eval batches are small and infrequent, so the extra forward pass is not
    worth threading through the loss-computation path just to save it."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        # under multi-GPU, Trainer passes the DataParallel-wrapped model here.
        # DataParallel only dispatches forward()/__call__ across GPUs; a direct
        # .decode() call has to go through the underlying module (.module) or
        # it errors with "'DataParallel' object has no attribute 'decode'"
        # (hit on a real 2-GPU Kaggle run) — runs single-GPU, fine since eval
        # is infrequent relative to training.
        real_model = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                             word_ids=inputs["word_ids"], labels=labels)
            loss = outputs.loss.detach() if outputs.loss is not None else None
            if prediction_loss_only:
                return (loss, None, None)
            decoded, word_mask = real_model.decode(inputs["input_ids"], inputs["attention_mask"],
                                                    inputs["word_ids"])
        w_max = word_mask.size(1)
        preds = torch.full((len(decoded), w_max), -100, dtype=torch.long)
        for i, seq in enumerate(decoded):
            preds[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        label_slice = labels[:, :w_max].detach().cpu() if labels is not None else None
        return (loss, preds, label_slice)
