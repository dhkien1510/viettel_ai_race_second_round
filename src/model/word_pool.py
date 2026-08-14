"""Shared helpers for the CRF path — pooling subword predictions/labels down
to WORD granularity, used by both train_ner.py (build word-level gold labels)
and infer_ner.py (build word-level spans for CRF decode), for BOTH tokenizer
families (fast subword tokenizers via `word_ids()`, and the slow-tokenizer
word-level path via `wordlevel_encode.encode_words`' own `word_ids`).

Why pooling is needed at all: a linear-chain CRF's forward algorithm/Viterbi
decode assume a clean RIGHT-padded mask (real content first, padding trails).
Subword-level label arrays don't have that shape — continuation subwords get
a filler value (None/-100) INTERSPERSED throughout the sequence, not just at
the end. Pooling to one position per word (using each word's first subword)
gives a naturally clean, contiguous sequence the CRF can operate on.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..schema import VALID_TYPES


def collapse_to_word_starts(word_ids_raw: List[Optional[int]]) -> Tuple[List[int], List[int]]:
    """word_ids_raw: per-token tokenizer word index (int) or None (special/
    padding/continuation-of-a-word-already-seen — wordlevel_encode already
    marks continuation subwords with the SAME wid as their word's first
    subword, so `wid in seen` catches those too).

    Returns:
      word_ids_out: same length as word_ids_raw; -1 everywhere except each
        word's first-subword position, which gets a running 0-based counter.
      first_pos: token index of each word's first subword, in word order —
        use it to look up that word's offset/label from whatever array is
        indexed by the ORIGINAL word id (word_ids_raw[t] for t in first_pos).
    """
    word_ids_out = [-1] * len(word_ids_raw)
    first_pos: List[int] = []
    seen = set()
    counter = 0
    for t, wid in enumerate(word_ids_raw):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        word_ids_out[t] = counter
        first_pos.append(t)
        counter += 1
    return word_ids_out, first_pos


def group_word_tags_into_entities(id2label: dict, text: str,
                                   word_spans: List[Tuple[int, int]],
                                   tag_ids: List[int]) -> List[Tuple[str, int, int]]:
    """Walk a word-level BIO tag sequence (already Viterbi-decoded, so B-/I-
    transitions are grammatically consistent by construction) into (type,
    start, end) spans. Still guards against merging across a newline the CRF
    doesn't know about (a bullet/line boundary is a real span limit even if
    the model tags both sides the same type)."""
    groups: List[Tuple[str, int, int]] = []
    cur = None  # [type, start, end]
    for (s, e), tid in zip(word_spans, tag_ids):
        label = id2label[int(tid)]
        if label == "O":
            if cur:
                groups.append(tuple(cur))
                cur = None
            continue
        bio, typ = label.split("-", 1)
        if typ not in VALID_TYPES:
            continue
        gap_has_newline = cur is not None and "\n" in text[cur[2]:s]
        if bio == "B" or cur is None or cur[0] != typ or gap_has_newline:
            if cur:
                groups.append(tuple(cur))
            cur = [typ, s, e]
        else:
            cur[2] = e
    if cur:
        groups.append(tuple(cur))
    return groups
