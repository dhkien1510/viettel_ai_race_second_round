"""Patch assertion LoRA adapter to load the local Qwen base model on Vast."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    adapter = Path("models/superbest_fullfit_qwen_v1_localmodel/assertion/adapter_config.json")
    data = json.loads(adapter.read_text(encoding="utf-8"))
    data["base_model_name_or_path"] = "/workspace/models/Qwen2.5-3B-Instruct-local"
    adapter.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched {adapter}: {data['base_model_name_or_path']}")


if __name__ == "__main__":
    main()
