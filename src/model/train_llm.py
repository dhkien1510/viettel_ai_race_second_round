"""Fine-tune a Qwen2.5 LLM backend with Unsloth (2x faster QLoRA).

Trains the generative backend (group C) on our own note→JSON pairs so the model
stops relying on the in-prompt few-shot and learns the task directly. The prompt
format is IMPORTED from src/backends/llm.py (build_messages / items_to_target),
so training matches inference exactly — no drift.

Data: the same (note, labels) pairs the encoder uses — silver by default
(data/silver + data/input), or synthetic (data/synth/labels + data/synth/notes),
or hand gold. Only {text, type, assertions} is used as the target (positions are
relocated at inference; candidates come from the linker).

Output: a MERGED 16-bit checkpoint at models/<key>/ + backend_meta.json. The
existing LLMBackend loads it with plain transformers (no unsloth needed at
inference — important for the self-host submission), and auto-drops the few-shot
because the meta marks it fine-tuned.

Typical runs (Colab/Kaggle T4 is enough for 1.5B and 3B QLoRA):
    python -m src.model.train_llm --model qwen25_1_5b
    python -m src.model.train_llm --model qwen25_3b --epochs 3
    python -m src.model.train_llm --model qwen25_1_5b \
        --train data/synth/labels --input data/synth/notes

Then inference picks the checkpoint up automatically (models/<key>/ beats base):
    python scripts/make_submission.py --model qwen25_1_5b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_deps():
    try:
        import unsloth  # noqa: F401
        import torch  # noqa: F401
        import trl  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "LLM fine-tuning needs unsloth + trl + torch.\n"
            "Install (Colab/Kaggle GPU):\n"
            '    pip install "unsloth[colab-new]" trl\n'
            f"(import error: {exc})"
        )


def unsloth_base(model_id: str, load_in_4bit: bool) -> str:
    """Map a registry HF id to Unsloth's mirror for fast 4bit download.
    'Qwen/Qwen2.5-1.5B-Instruct' -> 'unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit'.
    Falls back to the plain id if it is already an unsloth/local model."""
    if model_id.startswith("unsloth/") or "/" not in model_id:
        return model_id
    tail = model_id.split("/", 1)[1]
    return f"unsloth/{tail}-bnb-4bit" if load_in_4bit else f"unsloth/{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25_1_5b",
                    help="registry key (configs/models.yaml): qwen25_1_5b / qwen25_3b / qwen25_7b")
    ap.add_argument("--base", default=None,
                    help="override base model id (else derived from the registry key)")
    ap.add_argument("--train", default="data/silver",
                    help="dir of <id>.json label files (silver / synth / gold)")
    ap.add_argument("--input", default="data/input",
                    help="dir of matching <id>.txt notes")
    ap.add_argument(
        "--verifier-data",
        default="data/round02_qwen_verifier.jsonl",
        help="confidence-aware JSONL used when --task verifier",
    )
    ap.add_argument(
        "--verifier-class-cap",
        type=int,
        default=800,
        help="maximum sampled rows per verifier action before confidence repeats",
    )
    ap.add_argument("--out", default=None, help="checkpoint dir (default models/<key>)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 to cap steps (quick test)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    # 4096 phủ 100% mẫu (đo: max 3696 token gồm system+instruction+note+target).
    # 2048 chỉ phủ 73% -> 27% mẫu mất đuôi JSON. Nếu T4 OOM ở 4096, hạ --batch về 1
    # (đừng hạ --max-seq-len xuống dưới 3072, sẽ cắt marker của note dài lần nữa).
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-4bit", action="store_true", help="load in 16bit instead of 4bit QLoRA")
    ap.add_argument("--save", choices=["merged_16bit", "lora"], default="merged_16bit",
                    help="merged_16bit = self-contained (inference needs only transformers); "
                         "lora = tiny adapter (inference needs the base + peft)")
    ap.add_argument(
        "--task",
        choices=["joint", "ner", "assertion", "multitask", "verifier"],
        default="joint",
        help=(
            "joint = legacy full-note text/type/assertions target; ner/assertion = "
            "new segmented two-pass tasks; multitask = both new task schemas; "
            "verifier = KEEP/DROP/ADJUST occurrence classifier"
        ),
    )
    args = ap.parse_args()

    _require_deps()

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments, set_seed
    from datasets import Dataset

    # local imports (repo root is on sys.path when run as a module)
    from ..registry import load_registry, resolve_model_id, checkpoint_dir
    from ..backends.llm import build_messages, items_to_target
    from ..model.ner_dataset import load_gold_pairs
    from ..model.qwen_task_dataset import build_task_examples
    from ..model.qwen_verifier_dataset import balance_verifier_rows, verifier_messages

    set_seed(args.seed)

    reg = load_registry()
    key = args.model
    if key in reg and reg[key].get("backend") != "llm":
        raise SystemExit(
            f"'{key}' là backend '{reg[key].get('backend')}', không dùng train_llm.\n"
            f"  - encoder (xlmr_base, ...) -> python -m src.model.train_ner --model {key}\n"
            f"  - llm (qwen25_1_5b/3b/7b)  -> python -m src.model.train_llm --model {key}"
        )
    load_in_4bit = not args.no_4bit
    base = args.base or unsloth_base(resolve_model_id(key), load_in_4bit)
    out_dir = args.out or str(checkpoint_dir(key))
    print(f"base={base}  4bit={load_in_4bit}  out={out_dir}")

    # ---- data: (note, items) -> chat text (prompt + assistant JSON) ---------
    root = Path(__file__).resolve().parents[2]
    train_dir = (root / args.train) if not Path(args.train).is_absolute() else Path(args.train)
    input_dir = (root / args.input) if not Path(args.input).is_absolute() else Path(args.input)
    pairs = [] if args.task == "verifier" else load_gold_pairs(input_dir, train_dir)
    if args.task != "verifier" and not pairs:
        raise SystemExit(
            f"No training pairs (input={input_dir}, labels={train_dir}).\n"
            f"Bootstrap silver first:  python scripts/make_silver.py"
        )
    if args.task == "verifier":
        verifier_path = (
            root / args.verifier_data
            if not Path(args.verifier_data).is_absolute()
            else Path(args.verifier_data)
        )
        verifier_rows = [
            json.loads(line)
            for line in verifier_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        verifier_rows = balance_verifier_rows(
            verifier_rows, args.verifier_class_cap, args.seed
        )
        print(f"Loaded {len(verifier_rows)} weighted verifier rows from {verifier_path}")
    else:
        verifier_rows = []
        print(f"Loaded {len(pairs)} labeled notes from {train_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=args.max_seq_len,
        dtype=None, load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    # fine-tuned model is trained WITHOUT the few-shot (include_fewshot=False):
    # it learns the task, so the prompt stays short — matches LLMBackend's
    # auto-drop of the few-shot for fine-tuned checkpoints.
    def render(convo, target: str):
        convo = convo + [{"role": "assistant", "content": target}]
        return tokenizer.apply_chat_template(convo, tokenize=False,
                                             add_generation_prompt=False)

    texts = []
    task_counts = {}
    if args.task == "verifier":
        for row in verifier_rows:
            messages, target = verifier_messages(row)
            texts.append(render(messages, target))
            label = f"verifier_{row['action'].lower()}"
            task_counts[label] = task_counts.get(label, 0) + 1
    for text, items, _tags in pairs:
        if args.task == "joint":
            convo = build_messages(text, include_fewshot=False)
            texts.append(render(convo, items_to_target(items)))
            task_counts["joint"] = task_counts.get("joint", 0) + 1
            continue
        for example in build_task_examples(text, items, args.task):
            texts.append(render(example.messages, example.target))
            task_counts[example.task] = task_counts.get(example.task, 0) + 1

    rows = {"text": texts}
    ds = Dataset.from_dict(rows)
    print(f"Built {len(ds)} training examples: {task_counts}")

    ta_kwargs = dict(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5, learning_rate=args.lr,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=args.seed,
        output_dir=str(Path(out_dir) / "_trainer"), report_to="none",
        # Disable the Trainer's own mid-run checkpointing. On the bleeding-edge
        # Unsloth/TRL stack its `torch.save(self.args)` crashes because Unsloth
        # re-patches SFTConfig into a different class object
        # (PicklingError: ... not the same object as trl...SFTConfig). We don't
        # need those checkpoints anyway — the single merged 16-bit save below
        # (Unsloth's own saver, no torch.save of the args) is the deliverable.
        save_strategy="no",
    )
    if args.max_steps and args.max_steps > 0:
        ta_kwargs["max_steps"] = args.max_steps
    else:
        ta_kwargs["num_train_epochs"] = args.epochs

    # ĐỘ DÀI CHUỖI — sửa bug im lặng version-drift TRL/Transformers.
    #
    # TRL mới (khớp Transformers 5.x) ĐÃ BỎ tham số `max_seq_length` của SFTTrainer;
    # nó đọc `max_length` từ SFTConfig, MẶC ĐỊNH 1024. Nên đặt --max-seq-len 2048 mà
    # vẫn bị cắt ở 1024: prompt của ta = system + instruction (~654 token cố định)
    # + note, rồi mới tới marker `<|im_start|>assistant`. Marker vượt 1024 ở 93%
    # số mẫu -> train_on_responses_only mask sạch nhãn thành -100 -> Unsloth loại
    # 479/516 mẫu -> crash "'int' object has no attribute 'mean'".
    # Đo trên 516 note: đầy đủ (system+instruction+note+target) trung vị 1803,
    # max 3696 token. Cần max_length = 4096 để phủ 100%.
    #
    # Truyền qua SFTConfig, và lọc theo field mà version SFTConfig hiện có để chạy
    # được cả TRL cũ (max_seq_length) lẫn mới (max_length).
    from trl import SFTConfig
    import inspect
    _sft_ok = set(inspect.signature(SFTConfig).parameters)
    for _name in ("max_length", "max_seq_length"):
        if _name in _sft_ok:
            ta_kwargs[_name] = args.max_seq_len
    for _name, _val in (("dataset_text_field", "text"),
                        ("packing", False), ("dataset_num_proc", 2)):
        if _name in _sft_ok:
            ta_kwargs[_name] = _val

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(**ta_kwargs),
    )
    # train ONLY on the assistant JSON (mask the prompt) — Qwen ChatML markers
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # FAIL NHANH, RÕ RÀNG nếu masking ăn gần hết nhãn (marker vẫn bị cắt vì lý do
    # nào đó). Không có guard này, triệu chứng là crash tối nghĩa
    # "'int' object has no attribute 'mean'" ở tận trong vòng train của Unsloth.
    try:
        import numpy as _np
        _lab = trainer.train_dataset["labels"]
        _kept = sum(1 for row in _lab if any(t != -100 for t in row))
        if _kept < 0.5 * len(_lab):
            raise SystemExit(
                f"CHỈ {_kept}/{len(_lab)} mẫu còn nhãn sau masking — marker "
                f"'<|im_start|>assistant' bị cắt bởi max_length={args.max_seq_len}.\n"
                f"Tăng --max-seq-len (đo được cần 4096 để phủ 100%)."
            )
        print(f"[train_llm] masking OK: {_kept}/{len(_lab)} mẫu còn nhãn.")
    except (KeyError, TypeError):
        pass  # cấu trúc dataset khác version -> bỏ qua kiểm tra, để train chạy

    trainer.train()

    # ---- save a self-contained checkpoint the LLMBackend can load ------------
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if args.save == "merged_16bit":
        model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    with open(Path(out_dir) / "backend_meta.json", "w", encoding="utf-8") as f:
        json.dump({"backend": "llm", "fine_tuned": True,
                   "base_model": base, "model_key": key,
                   "save_method": args.save, "task": args.task},
                  f, ensure_ascii=False, indent=2)

    print(f"\nSaved fine-tuned LLM to: {out_dir}")
    print(f"Run inference with:  python scripts/make_submission.py --model {key}")


if __name__ == "__main__":
    main()
