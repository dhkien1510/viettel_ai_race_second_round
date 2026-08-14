"""Bước 3: Train context reranker.

Input: 1 hoặc nhiều file JSONL sinh ra từ build_training_data.py (data thật)
và/hoặc generate_synonyms.py build-pairs (data tổng hợp từ LLM). Mỗi dòng là
1 hàng (entity, candidate, label) — xem docstring 2 script trên để biết
schema.

Cách hoạt động: mỗi entity là 1 "nhóm" gồm tất cả candidate của nó (kể cả
"NONE"); model encode cả nhóm, chọn ra candidate đúng bằng softmax cross-
entropy (bài toán "chọn 1 trong N", không phải phân loại nhị phân từng
dòng). Val split theo DOCUMENT (không theo entity) để không rò rỉ 2 entity
cùng 1 bệnh án vào cả train và val — entity có doc_id không phải số (vd
"synth0", "synth1" từ data tổng hợp) luôn nằm ở train, không dùng để validate.

Chạy:
    python train.py \\
        --pairs data/training_pairs.jsonl data/synthetic_pairs.jsonl \\
        --out models/reranker \\
        --epochs 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Reranker(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size
        self.score_head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.score_head(cls).squeeze(-1)


def load_groups(paths: list[Path]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = (r["doc_id"], tuple(r["position"]), r["entity_text"], r["gold_code"])
            groups.setdefault(key, []).append(r)
    return groups


def cap_group(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    if len(rows) <= cap:
        return rows
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(neg)
    return pos + neg[: max(0, cap - len(pos))]


def pretokenize(groups: dict, keys: list, tokenizer, cap: int, max_length: int, rng: random.Random):
    out = {}
    for key in keys:
        rows = cap_group(groups[key], cap, rng)
        pos_idx = next(i for i, r in enumerate(rows) if r["label"] == 1)
        contexts = [r["context"] for r in rows]
        cands = [f"{r['candidate_code']}: {r['candidate_name']}" for r in rows]
        enc = tokenizer(contexts, cands, truncation=True, max_length=max_length,
                         padding="max_length", return_tensors="pt")
        n = len(rows)
        input_ids = torch.zeros(cap, max_length, dtype=torch.long)
        attn = torch.zeros(cap, max_length, dtype=torch.long)
        cand_mask = torch.zeros(cap, dtype=torch.bool)
        input_ids[:n] = enc["input_ids"]
        attn[:n] = enc["attention_mask"]
        cand_mask[:n] = True
        out[key] = (input_ids, attn, cand_mask, pos_idx)
    return out


def run_epoch(model, cache: dict, keys: list, device: str, rng: random.Random, train: bool,
              optimizer=None, meta_batch: int = 8, cap: int = 40):
    model.train(train)
    total_loss = 0.0
    correct = 0
    if train:
        rng.shuffle(keys)
        optimizer.zero_grad()
    for start in range(0, len(keys), meta_batch):
        batch_keys = keys[start:start + meta_batch]
        B = len(batch_keys)
        input_ids = torch.stack([cache[k][0] for k in batch_keys]).to(device)
        attn = torch.stack([cache[k][1] for k in batch_keys]).to(device)
        cand_mask = torch.stack([cache[k][2] for k in batch_keys]).to(device)
        pos_idx = torch.tensor([cache[k][3] for k in batch_keys], device=device)
        L = input_ids.shape[-1]
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                             enabled=(device == "cuda")):
            # Do not send all-padding candidate rows through the encoder. XLM-R
            # attention over an entirely masked row can produce NaNs which then
            # contaminate gradients even after the row is masked from the loss.
            flat_mask = cand_mask.view(-1)
            valid_logits = model(
                input_ids.view(B * cap, L)[flat_mask],
                attn.view(B * cap, L)[flat_mask],
            )
            flat_logits = torch.full(
                (B * cap,), float("-inf"), device=device, dtype=valid_logits.dtype
            )
            flat_logits = flat_logits.masked_scatter(flat_mask, valid_logits)
            logits = flat_logits.view(B, cap)
            loss = F.cross_entropy(logits, pos_idx)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at entity batch offset {start}; "
                "check candidate masks and model precision."
            )
        if train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * B
        correct += (torch.argmax(logits, dim=-1) == pos_idx).sum().item()
    return total_loss / len(keys), correct / len(keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cap", type=int, default=40, help="số candidate tối đa/entity (padding/cắt về mức này)")
    ap.add_argument("--max-length", dest="max_length", type=int, default=192)
    ap.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.15)
    ap.add_argument("--meta-batch", dest="meta_batch", type=int, default=8,
                     help="số entity xử lý cùng lúc trên GPU — giảm nếu bị OOM")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    groups = load_groups(args.pairs)
    real_doc_ids = sorted({k[0] for k in groups if k[0].isdigit()}, key=int)
    rng.shuffle(real_doc_ids)
    n_val = max(1, round(len(real_doc_ids) * args.val_ratio))
    val_docs = set(real_doc_ids[:n_val])
    train_keys = [k for k in groups if k[0] not in val_docs]
    val_keys = [k for k in groups if k[0] in val_docs]
    n_synth = sum(1 for k in groups if not k[0].isdigit())
    print(f"train entities={len(train_keys)} (gồm {n_synth} tổng hợp)  "
          f"val entities={len(val_keys)} (toàn bộ là data thật)  val docs={len(val_docs)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Đang tokenize (1 lần duy nhất)...", flush=True)
    cache = {**pretokenize(groups, train_keys, tokenizer, args.cap, args.max_length, rng),
             **pretokenize(groups, val_keys, tokenizer, args.cap, args.max_length, rng)}

    model = Reranker(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, cache, train_keys, device, rng, train=True,
                                           optimizer=optimizer, meta_batch=args.meta_batch, cap=args.cap)
        val_loss, val_acc = run_epoch(model, cache, val_keys, device, rng, train=False,
                                       meta_batch=args.meta_batch, cap=args.cap)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            marker = " * BEST"
            torch.save(model.state_dict(), out_dir / "best.pt")
        print(f"epoch {epoch:>3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}", flush=True)

    tokenizer.save_pretrained(out_dir)
    (out_dir / "config.json").write_text(json.dumps({"base_model": args.model, "cap": args.cap,
                                                        "max_length": args.max_length}), encoding="utf-8")
    (out_dir / "val_docs.json").write_text(json.dumps(sorted(val_docs, key=int)), encoding="utf-8")
    print(f"\nBest val entity accuracy: {best_acc:.4f}")
    print(f"Checkpoint tốt nhất -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
