"""Model registry + backend factory.

Resolve a --model argument (a registry key, or a checkpoint directory path)
into a SpanBackend. Rule pipeline stays model-free; this only builds the
OPTIONAL extra backend that gets OR-merged with the rules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .schema import REPO_ROOT
from .backends.base import SpanBackend

CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
MODELS_DIR = REPO_ROOT / "models"


def load_registry(path=None) -> dict:
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def checkpoint_dir(name: str) -> Path:
    """Where train_ner writes / infer reads an encoder checkpoint for a key."""
    return MODELS_DIR / name


def resolve_model_id(name: str) -> str:
    """Registry key -> HF model id (for training). Pass-through if not a key."""
    reg = load_registry()
    if name in reg:
        return reg[name]["model_id"]
    return name


def _meta_backend(model_dir: str) -> Optional[str]:
    """Read backend_meta.json to learn whether a checkpoint dir is an encoder or
    a fine-tuned llm (train_ner / train_llm both write this field)."""
    import json
    meta_path = os.path.join(model_dir, "backend_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f).get("backend")
    return None


def build_backend(spec: Optional[str], max_len: int = 256, stride: int = 64) -> Optional[SpanBackend]:
    """spec may be: None, a registry key, or a path to a trained checkpoint.

    `max_len`/`stride` only affect the encoder backend (sliding-window size
    over long notes) — ignored for rules/llm, which don't window."""
    if not spec:
        return None

    # 1) an existing directory -> a trained checkpoint. backend_meta.json says
    #    whether it's an encoder or a fine-tuned LLM.
    if os.path.isdir(spec):
        mb = _meta_backend(spec)
        if mb == "llm":
            from .backends.llm import LLMBackend
            return LLMBackend(spec)
        if mb == "gliner":
            from .backends.gliner import GLiNERBackend
            return GLiNERBackend(spec)
        from .backends.encoder import EncoderBackend
        return EncoderBackend(spec, max_len=max_len, stride=stride)

    # 2) a registry key
    reg = load_registry()
    if spec in reg:
        entry = reg[spec]
        backend = entry.get("backend")
        if backend == "rules":
            # branch 0: rule-only. RuleBackend is always on in Pipeline, so the
            # extra backend is simply None.
            return None
        if backend == "llm":
            from .backends.llm import LLMBackend
            # prefer a fine-tuned checkpoint at models/<key>/ if present;
            # otherwise fall back to the base HF model (zero-shot).
            ckpt = checkpoint_dir(spec)
            if ckpt.is_dir():
                return LLMBackend(str(ckpt))
            return LLMBackend(entry["model_id"])
        if backend == "gliner":
            from .backends.gliner import GLiNERBackend
            # ưu tiên checkpoint fine-tune ở models/<key>/, không có thì dùng base
            # HF (zero-shot) — giống nhánh llm. Fallback phải HÉT LÊN: đã từng có
            # submission zero-shot đội lốt fine-tune vì thư mục checkpoint vắng mặt.
            ckpt = checkpoint_dir(spec)
            if ckpt.is_dir():
                print(f"[registry] gliner: dùng checkpoint fine-tune {ckpt}")
                return GLiNERBackend(str(ckpt))
            print(f"⚠⚠ [registry] {ckpt} KHÔNG tồn tại -> dùng BASE ZERO-SHOT "
                  f"({entry['model_id']}). Nếu bạn định chạy bản fine-tune, DỪNG và "
                  f"kiểm tra --out lúc train có khớp models/{spec} không.")
            return GLiNERBackend(entry["model_id"])
        if backend == "encoder":
            ckpt = checkpoint_dir(spec)
            if not ckpt.is_dir():
                raise SystemExit(
                    f"Encoder '{spec}' chưa được train.\n"
                    f"Train trước:  python -m src.model.train_ner --model {spec}\n"
                    f"(checkpoint sẽ nằm ở {ckpt})"
                )
            from .backends.encoder import EncoderBackend
            return EncoderBackend(str(ckpt), word_segment=entry.get("word_segment", False),
                                   max_len=max_len, stride=stride)
        raise SystemExit(f"Registry '{spec}': backend không hợp lệ: {backend!r}")

    raise SystemExit(
        f"--model {spec!r} không phải key trong configs/models.yaml và cũng không "
        f"phải thư mục checkpoint. Các key hợp lệ: {', '.join(load_registry())}"
    )
