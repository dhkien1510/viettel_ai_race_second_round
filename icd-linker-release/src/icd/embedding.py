"""Embedding cache and retrieval for ICD-10 TT06 names.

Supports BAAI/bge-m3 (via FlagEmbedding) and general transformer models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_FILENAME = "icd_tt06_vectors.npy"
META_FILENAME = "icd_tt06_vector_meta.json"


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------------------------------------------------------------------
#  BGE-M3 helpers (FlagEmbedding)
# ---------------------------------------------------------------------------

def _is_bge_m3(model_name: str) -> bool:
    return "bge-m3" in model_name.lower()


def _load_bgem3(model_name: str, device: str = "cpu") -> Any:
    """Load BGE-M3 model via FlagEmbedding."""
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(
        model_name,
        use_fp16=(device != "cpu"),
        device=device,
    )


def _encode_bgem3(model: Any, texts: list[str],
                  batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """Encode texts with BGE-M3 (dense only)."""
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = output["dense_vecs"].astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _encode_bgem3_query(model: Any, text: str,
                        max_length: int = 256) -> np.ndarray:
    """Encode a single query with BGE-M3."""
    return _encode_bgem3(model, [text], max_length=max_length)[0]


# ---------------------------------------------------------------------------
#  Generic transformer helpers (fallback for non-BGE-M3 models)
# ---------------------------------------------------------------------------

def _load_transformers(model_name: str, device: str = "cpu") -> tuple[Any, Any]:
    """Load model + tokenizer using transformers."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.unpad_inputs = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, config=config, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


def _mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.shape).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _encode_transformers_documents(
    model_tokenizer: tuple[Any, Any],
    texts: list[str],
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    import torch
    model, tokenizer = model_tokenizer
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            return_tensors="pt").to(device)
        seq_len = encoded["input_ids"].shape[1]
        bs = encoded["input_ids"].shape[0]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                position_ids=pos_ids,
            )
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vectors.append(pooled.cpu().numpy())
    return _normalize_rows(np.concatenate(all_vectors, axis=0))


def _encode_transformers_query(
    model_tokenizer: tuple[Any, Any],
    text: str,
    device: str = "cpu",
) -> np.ndarray:
    import torch
    model, tokenizer = model_tokenizer
    encoded = tokenizer([text], padding=True, truncation=True,
                        return_tensors="pt").to(device)
    seq_len = encoded["input_ids"].shape[1]
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(1, -1)
    with torch.no_grad():
        outputs = model(**encoded, position_ids=pos_ids)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return _normalize_rows(pooled.cpu().numpy())


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> Any:
    """Load embedding model. Returns a model object (BGE-M3 or transformers)."""
    if _is_bge_m3(model_name):
        return _load_bgem3(model_name, device)
    else:
        return _load_transformers(model_name, device)


def encode_documents(
    model: Any,
    texts: list[str],
    batch_size: int = 128,
    device: str = "cpu",
    model_name: str = "",
) -> np.ndarray:
    """Encode texts into normalized embedding vectors."""
    if _is_bge_m3(model_name or getattr(model, '__class__', '')) or \
       hasattr(model, 'encode'):
        return _encode_bgem3(model, texts, batch_size)
    else:
        return _encode_transformers_documents(model, texts, batch_size, device)


def encode_query(
    model: Any,
    text: str,
    device: str = "cpu",
    model_name: str = "",
) -> np.ndarray:
    """Encode a single query text into a normalized embedding vector."""
    if _is_bge_m3(model_name or getattr(model, '__class__', '')) or \
       hasattr(model, 'encode'):
        return _encode_bgem3_query(model, text)
    else:
        return _encode_transformers_query(model, text, device)


# ---------------------------------------------------------------------------
#  Cache builder
# ---------------------------------------------------------------------------

def _embedding_entries(index_dir: str | Path,
                       levels: set[str] | None = None) -> list[dict[str, Any]]:
    """Build entries from the ICD index with enriched text."""
    from .index import ICDIndex
    index = ICDIndex(index_dir)
    levels = levels or {"category", "disease"}
    entries = []
    for node in index.nodes:
        if node.get("level") not in levels:
            continue
        code = node.get("code")
        name = node.get("name_vi")
        if not code or not name:
            continue
        # enriched text (matching what BGE-M3 pipeline generates)
        parts = [f"Mã ICD: {code}", f"Tên bệnh: {name}"]
        parent_code = node.get("parent_code", "")
        if parent_code:
            pn = index.get_code(parent_code)
            if pn:
                parts.append(f"Nhóm cha: {pn.get('name_vi', '')}")
        entries.append({
            "code": code,
            "name_vi": name,
            "level": node.get("level"),
            "text": ". ".join(parts),
        })
    entries.sort(key=lambda item: item["code"])
    return entries


def build_embedding_cache(
    index_dir: str | Path = "data/processed",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model: Any = None,
    batch_size: int = 128,
    levels: set[str] | None = None,
) -> dict[str, Any]:
    """Build embedding cache: encode ICD entries and save vectors + meta."""
    index_dir = Path(index_dir)
    levels = levels or {"category", "disease"}
    entries = _embedding_entries(index_dir, levels)
    texts = [e["text"] for e in entries]

    if model is None:
        model = load_embedding_model(model_name)

    if _is_bge_m3(model_name) or hasattr(model, "encode"):
        vectors = _encode_bgem3(model, texts, batch_size)
    else:
        vectors = _encode_transformers_documents(model, texts, batch_size)

    np.save(index_dir / VECTOR_FILENAME, vectors.astype(np.float32))
    meta = {
        "model_name": model_name,
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "levels": sorted(levels),
        "entries": entries,
    }
    (index_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
#  Retrieval index
# ---------------------------------------------------------------------------

class ICDEmbeddingIndex:
    """Load cached embedding vectors and retrieve candidates by cosine similarity."""

    def __init__(
        self,
        index_dir: str | Path = "data/processed",
        model_name: str | None = None,
        model: Any = None,
        vectors_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.vectors_path = Path(vectors_path) if vectors_path else self.index_dir / VECTOR_FILENAME
        self.meta_path = Path(meta_path) if meta_path else self.index_dir / META_FILENAME
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.model_name = model_name or self.meta.get("model_name") or DEFAULT_EMBEDDING_MODEL
        if self.meta.get("model_name") and self.meta["model_name"] != self.model_name:
            raise ValueError(
                f"Embedding cache was built with {self.meta['model_name']!r}, "
                f"not {self.model_name!r}."
            )
        self.vectors = _normalize_rows(np.load(self.vectors_path))
        self.entries: list[dict[str, Any]] = self.meta.get("entries", [])
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            self._model = load_embedding_model(self.model_name)
        return self._model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_codes: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.entries:
            return []

        model = self._load_model()
        query_vector = encode_query(model, query, model_name=self.model_name)
        scores = (query_vector @ self.vectors.T).reshape(-1)

        candidate_set = {code.upper() for code in candidate_codes or []}
        if candidate_set:
            indices = [
                idx for idx, entry in enumerate(self.entries)
                if str(entry.get("code", "")).upper() in candidate_set
            ]
        else:
            indices = list(range(len(self.entries)))
        if not indices:
            return []

        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
        return [
            {
                "code": self.entries[idx]["code"],
                "name_vi": self.entries[idx]["name_vi"],
                "score": float(scores[idx]),
            }
            for idx in ranked
        ]
