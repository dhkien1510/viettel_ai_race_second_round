"""Minimal .env loader (stdlib, no dependency).

Reads KEY=VALUE lines from the repo-root .env and puts them into os.environ.
Real environment variables always win (we never override an already-set var),
so `export OPENAI_API_KEY=...` still takes precedence over .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..schema import REPO_ROOT


def load_dotenv(path=None, override: bool = False) -> bool:
    p = Path(path) if path else (REPO_ROOT / ".env")
    if not p.is_file():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
    return True
