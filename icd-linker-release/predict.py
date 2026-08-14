"""icd-linker-release: predict an ICD-10 (TT06) code for a Vietnamese
diagnosis mention, given the surrounding note context.

Usage (CLI):
    python predict.py --text "béo phì" \\
        --before "...Các bệnh mãn tính\\n- Tiểu đường loại 1\\n- " \\
        --after "\\n- Nhiễm trùng đường tiết niệu..."

    # or, simplest form: paste the whole note and mark the span with [[ ]]
    python predict.py --context "...tiền sử [[tăng huyết áp]] và đái tháo đường..."

Usage (Python API):
    from predict import ICDLinker
    linker = ICDLinker()  # loads once, reuse across calls
    code = linker.predict(context="...tiền sử [[tăng huyết áp]]...")
    # or
    code = linker.predict(text="tăng huyết áp", before="...tiền sử ", after=" và đái tháo đường...")

Returns an ICD-10 TT06 code string (e.g. "I10"), or "NONE" if the model
decides no code should be assigned (e.g. the mention doesn't describe an
active, codeable diagnosis in this context).

How it decides: two stages, exactly like at evaluation time --
  1. Candidate retrieval: dictionary/alias exact match, fuzzy string
     matching, and dense BGE-M3 embedding similarity all vote on a pool of
     plausible codes for the diagnosis text alone (no context yet).
  2. Context reranking: a fine-tuned XLM-RoBERTa cross-encoder reads the
     FULL context (not just the isolated diagnosis text) together with each
     candidate's official name and scores how well it fits; the highest-
     scoring candidate (or the built-in "no code" option) is returned.
Stage 1 gives recall (finds plausible codes even for typos/synonyms/
abbreviations); stage 2 gives precision (uses context to pick the right one
among several plausible codes, e.g. distinguishing "tăng huyết áp" alone
from "tăng huyết áp kèm biến chứng thận").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from icd.index import ICDIndex          # noqa: E402
from icd.matcher import ICDMatcher      # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

CASCADE_TOPK = 8
DENSE_TOPK = 30
CAP = 40


class _Reranker(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.score_head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.score_head(cls).squeeze(-1)


class ICDLinker:
    """Load once (index + dense vectors + BGE-M3 + reranker), then call
    .predict(...) as many times as needed."""

    def __init__(self, data_dir: str | Path = ROOT / "data" / "processed",
                 model_dir: str | Path = ROOT / "models" / "reranker",
                 device: str | None = None):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.index = ICDIndex(str(self.data_dir))
        self.matcher = ICDMatcher(index=self.index)
        self._cascade_cache: dict[str, dict] = {}

        vectors = np.load(self.data_dir / "icd_tt06_vectors.npy")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_vectors = (vectors / norms).astype(np.float32)
        self.dense_entries = json.loads(
            (self.data_dir / "icd_tt06_vector_meta.json").read_text(encoding="utf-8"))["entries"]

        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer("BAAI/bge-m3", device=self.device)

        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.reranker = _Reranker(config["base_model"]).to(self.device)
        self.reranker.load_state_dict(
            torch.load(self.model_dir / "best.pt", map_location=self.device, weights_only=True))
        self.reranker.eval()

    def _cached_cascade(self, span: str) -> dict:
        key = span.strip().lower()
        if key not in self._cascade_cache:
            self._cascade_cache[key] = self.matcher.match(span)
        return self._cascade_cache[key]

    def _dense_retrieve(self, span: str, top_k: int = DENSE_TOPK) -> list[dict]:
        qv = self.embed_model.encode([span], normalize_embeddings=True)[0]
        scores = self.dense_vectors @ qv
        idx = np.argsort(-scores)[:top_k]
        return [{"code": self.dense_entries[i]["code"], "name_vi": self.dense_entries[i]["name_vi"]}
                for i in idx]

    def candidates(self, text: str) -> list[tuple[str, str]]:
        """Stage 1 only: return the (code, name) candidate pool for a bare
        diagnosis text, before context reranking. Useful for debugging."""
        cascade = self._cached_cascade(text)
        pool: dict[str, str] = {}
        for c in (cascade.get("candidates") or [])[:CASCADE_TOPK]:
            code = c.get("code")
            if code and code not in pool:
                pool[code] = c.get("name_vi", "")
        for c in self._dense_retrieve(text):
            if c["code"] not in pool:
                pool[c["code"]] = c["name_vi"]
        return list(pool.items())[: CAP - 1]

    def predict(self, context: str | None = None, text: str | None = None,
                before: str = "", after: str = "") -> str:
        """Provide EITHER `context` (a string with the diagnosis mention
        marked as [[...]]) OR `text` + `before`/`after` (assembled into that
        same marked form). Returns an ICD-10 code, or "NONE"."""
        if context is not None:
            if "[[" not in context or "]]" not in context:
                raise ValueError("`context` must mark the diagnosis mention with [[ ]]")
            span = context.split("[[", 1)[1].split("]]", 1)[0]
        elif text is not None:
            span = text
            context = f"{before}[[{text}]]{after}"
        else:
            raise ValueError("Provide either `context` (with [[ ]] marker) or `text`.")

        pool = self.candidates(span)
        options = [("NONE", "(không gán mã ICD)")] + pool
        contexts = [context] * len(options)
        cands = [f"{code}: {name}" for code, name in options]
        enc = self.tokenizer(contexts, cands, truncation=True, max_length=self.max_length,
                              padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.reranker(**enc)
        best_idx = int(torch.argmax(logits).item())
        return options[best_idx][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", help="Full context string with the diagnosis marked as [[...]]")
    ap.add_argument("--text", help="Just the diagnosis mention (use with --before/--after)")
    ap.add_argument("--before", default="")
    ap.add_argument("--after", default="")
    ap.add_argument("--data-dir", default=ROOT / "data" / "processed", type=Path)
    ap.add_argument("--model-dir", default=ROOT / "models" / "reranker", type=Path,
                     help="Thư mục checkpoint đã train (output của train.py --out)")
    args = ap.parse_args()

    linker = ICDLinker(data_dir=args.data_dir, model_dir=args.model_dir)
    code = linker.predict(context=args.context, text=args.text, before=args.before, after=args.after)
    print(code)


if __name__ == "__main__":
    main()
