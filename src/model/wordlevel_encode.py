"""Manual word-level subword alignment for SLOW tokenizers (no fast variant).

PhoBERT (`vinai/phobert-base`) and ViHealthBERT (`demdecuong/vihealthbert-base-word`)
use `PhobertTokenizer`, which has no Fast counterpart in `transformers` — so
`return_offsets_mapping=True` (the mechanism the rest of the pipeline relies on
for exact character offsets) cannot be requested for them. This module builds
the equivalent alignment by hand, at WORD granularity instead of character
granularity: each whitespace-separated word (already underscore-joined by
underthesea for compounds) is tokenized on its own, its first subtoken carries
the real label, and continuation subtokens are marked `word_id=None` so they
are ignored both for loss (-100) and at inference time. This is the standard
approach for token classification with slow tokenizers.

Used by both `train_ner.py` (encode side) and `infer_ner.py` (predict side) so
the two stay in lockstep.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def word_spans(text: str) -> List[Tuple[int, int]]:
    """Character [start, end) span of each whitespace-separated word."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def words_and_tags_from_char_tags(text: str, char_tags: List[str]):
    """Reduce per-char BIO tags to one tag per whitespace-separated word (the
    tag of the word's first character — sufficient since these backbones only
    ever predict at word granularity anyway)."""
    spans = word_spans(text)
    words = [text[s:e] for s, e in spans]
    tags = [char_tags[s] if s < len(char_tags) else "O" for s, _e in spans]
    return words, tags, spans


def encode_words(tok, words: List[str], max_len: int, stride: int):
    """Tokenize whitespace-separated words with a slow tokenizer, producing
    overlapping windows analogous to `return_overflowing_tokens=True` +
    `stride` on a fast tokenizer.

    Returns a list of windows; each is {input_ids, attention_mask, word_ids}
    where word_ids[t] is the index into `words` for the FIRST subtoken of a
    word, and None for continuation subtokens / special tokens / padding.
    """
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    flat_ids: List[int] = []
    flat_word: List[Optional[int]] = []
    for wi, w in enumerate(words):
        sub_ids = tok(w, add_special_tokens=False)["input_ids"]
        if not sub_ids:
            continue
        flat_ids.append(sub_ids[0])
        flat_word.append(wi)
        for sid in sub_ids[1:]:
            flat_ids.append(sid)
            flat_word.append(None)

    if not flat_ids:
        return []

    body_len = max(1, max_len - 2)  # room for bos/eos
    win_step = max(1, body_len - stride)
    windows = []
    i = 0
    n = len(flat_ids)
    while i < n:
        chunk_ids = flat_ids[i:i + body_len]
        chunk_word = flat_word[i:i + body_len]
        ids = [bos] + chunk_ids + [eos]
        wids: List[Optional[int]] = [None] + chunk_word + [None]
        attn = [1] * len(ids)
        pad_n = max_len - len(ids)
        ids = ids + [pad] * pad_n
        wids = wids + [None] * pad_n
        attn = attn + [0] * pad_n
        windows.append({"input_ids": ids, "attention_mask": attn, "word_ids": wids})
        if i + body_len >= n:
            break
        i += win_step
    return windows
