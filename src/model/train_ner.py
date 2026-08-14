"""Fine-tune a token-classification NER model for span + type.

Requires the optional deps (torch + transformers + datasets). Run only after
you have training labels — either hand gold (data/gold) or, more practically,
SILVER labels bootstrapped from the rule pipeline (scripts/make_silver.py).

Default backbone is XLM-RoBERTa: it is subword-based, needs NO Vietnamese
word-segmentation, and its fast tokenizer gives exact character offsets — so
the predicted spans line up with the raw text with zero offset drift. PhoBERT
/ ViHealthBERT are word-level (underthesea segmentation) and have no fast
tokenizer, so they align labels at WORD granularity instead via
`wordlevel_encode.py` (see that module's docstring) — both train and infer
route through it automatically, no extra flags needed.

The model predicts span+type ONLY. Assertions (negated/family/historical) stay
rule-based (src/assertions), and candidates stay in the linker.

Typical run:
    python -m src.model.train_ner \
        --model FacebookAI/xlm-roberta-base \
        --train data/silver --input input \
        --out models/ner --epochs 10

The checkpoint is written to --out (default models/ner). Point inference at it
with `Pipeline(model_dir="models/ner")` or the --model flag on the scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _require_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import datasets  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Model training needs torch/transformers/datasets.\n"
            "Install the optional block in requirements.txt first.\n"
            f"(import error: {exc})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlmr_base",
                    help="registry key (configs/models.yaml) or a HF backbone id. "
                         "e.g. xlmr_base, mdeberta_base, phobert_base, vihealthbert")
    ap.add_argument("--train", default="data/silver",
                    help="dir of <id>.json label files (silver or gold)")
    ap.add_argument("--input", default="input")
    ap.add_argument("--out", default=None,
                    help="checkpoint dir (default models/<model-key>)")
    ap.add_argument("--word-segment", dest="word_segment", action="store_true",
                    help="force word-level segmentation (auto for phobert/vihealthbert)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.0,
                    help="tách validation theo DOCUMENT (không phải theo window đã "
                         "tokenize, để tránh rò rỉ giữa 2 window cùng 1 note), vd 0.2 "
                         "= 20%%. 0 (mặc định) = không tách, train toàn bộ như trước.")
    ap.add_argument("--crf", action="store_true",
                    help="thêm tầng CRF (ma trận chuyển trạng thái giữa nhãn BIO) "
                         "lên trên encoder thay cho linear+softmax độc lập từng "
                         "token — bắt buộc B-/I- hợp lệ theo ngữ pháp BIO, triệt "
                         "tiêu lỗi 1 từ bị tách thành 2 loại nhãn khác nhau. Kiến "
                         "trúc khác hẳn baseline -> phải train lại từ đầu, checkpoint "
                         "cũ không dùng lại được.")
    args = ap.parse_args()

    _require_deps()

    import random

    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
        default_data_collator, TrainerCallback, set_seed,
    )
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    from transformers.utils import logging as hf_logging
    from datasets import Dataset

    hf_logging.set_verbosity_error()
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    from .entity_metrics import entity_prf, entity_prf_overlap

    class CleanEpochCallback(TrainerCallback):
        """1 dòng/epoch: train_loss | eval_loss | f1/P/R —
        thay cho spam tqdm + log dict thô + "Writing model shards" mỗi epoch."""

        def __init__(self, metric_for_best: str | None, greater_is_better: bool, has_eval: bool):
            self._last_train_loss = None
            self._metric_for_best = metric_for_best
            self._greater_is_better = greater_is_better
            self._has_eval = has_eval
            self._best_so_far = None  # theo dõi riêng, không dựa vào state.best_metric
                                       # (state.best_metric không cập nhật khi hoà điểm)
            self.silent = False  # tạm im lặng cho lượt evaluate() cuối (reload best)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs or "eval_loss" in logs:
                return
            self._last_train_loss = logs["loss"]
            if not self._has_eval and not self.silent:
                # Không có validation -> không có on_evaluate để in dòng tổng
                # hợp, nên in ngay đây (logging_strategy="epoch" -> đúng 1
                # dòng/epoch, giá trị là trung bình cả epoch, không phải 1 step).
                print(f"epoch {logs.get('epoch', state.epoch):>5.2f} | "
                      f"train_loss={self._last_train_loss:.4f}")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics or self.silent:
                return
            cols = [f"epoch {metrics.get('epoch', state.epoch):>5.2f}"]
            cols.append(
                f"train_loss={self._last_train_loss:.4f}"
                if self._last_train_loss is not None else "train_loss=   n/a"
            )
            cols.append(f"eval_loss={metrics.get('eval_loss', float('nan')):.4f}")
            if "eval_f1" in metrics:
                # f1 = CHỒNG LẤN (luật chấm thật, dùng chọn checkpoint)
                # f1x = span chính xác (chỉ để tham khảo, nó chỉ chi phối WER)
                cols.append(f"f1={metrics['eval_f1']:.4f}")
                cols.append(f"P={metrics.get('eval_precision', float('nan')):.4f}")
                cols.append(f"R={metrics.get('eval_recall', float('nan')):.4f}")
                if "eval_f1_exact" in metrics:
                    cols.append(f"f1x={metrics['eval_f1_exact']:.4f}")
                    j = metrics["eval_f1"]
                    cols.append(f"J_ent={j / (2 - j):.4f}" if j else "J_ent=0.0000")
            best_key = f"eval_{self._metric_for_best}" if self._metric_for_best else None
            if best_key and best_key in metrics:
                value = metrics[best_key]
                is_new_best = (
                    self._best_so_far is None
                    or (value > self._best_so_far if self._greater_is_better else value < self._best_so_far)
                )
                if is_new_best:
                    self._best_so_far = value
                    cols.append("* BEST")
            print(" | ".join(cols))

    from .ner_dataset import load_gold_pairs, label_list
    from ..registry import load_registry, resolve_model_id
    from .. import wordseg
    from . import wordlevel_encode as wl
    from .word_pool import collapse_to_word_starts

    set_seed(args.seed)

    # resolve registry key -> HF id + word_segment default + default out dir
    reg = load_registry()
    model_key = args.model
    if model_key in reg and reg[model_key].get("backend") != "encoder":
        raise SystemExit(
            f"'{model_key}' là backend '{reg[model_key].get('backend')}', không train được.\n"
            f"  - 'rules' = nhánh 0, chỉ chạy: python scripts/make_submission.py --model rules\n"
            f"  - 'qwen*' = zero-shot, chạy thẳng: python scripts/make_submission.py --model {model_key}\n"
            f"Chỉ backend 'encoder' (xlmr_base, mdeberta_base, phobert_base, ...) mới cần train."
        )
    model_id = resolve_model_id(args.model)
    word_segment = args.word_segment
    if model_key in reg:
        word_segment = word_segment or bool(reg[model_key].get("word_segment", False))
    out_dir = args.out or (f"models/{model_key}" if model_key in reg else "models/ner")
    print(f"backbone={model_id}  word_segment={word_segment}  out={out_dir}")

    labels = label_list()
    l2i = {l: i for i, l in enumerate(labels)}

    root = Path(__file__).resolve().parents[2]
    train_dir = (root / args.train) if not Path(args.train).is_absolute() else Path(args.train)
    input_dir = (root / args.input) if not Path(args.input).is_absolute() else Path(args.input)

    pairs = load_gold_pairs(input_dir, train_dir)
    if not pairs:
        raise SystemExit(
            f"No training pairs found (input={input_dir}, labels={train_dir}).\n"
            f"Run: python scripts/make_silver.py   to bootstrap silver labels."
        )
    print(f"Loaded {len(pairs)} labeled documents from {train_dir}")

    # split BEFORE tokenizing into windows, and by document, so two windows of
    # the same note never end up split across train/val (that would leak).
    val_pairs = []
    if args.val_ratio > 0:
        rng = random.Random(args.seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * args.val_ratio))
        val_idx = set(idx[:n_val])
        val_pairs = [pairs[i] for i in idx if i in val_idx]
        pairs = [pairs[i] for i in idx if i not in val_idx]
        print(f"Tách theo document: {len(pairs)} train / {len(val_pairs)} validation "
              f"(--val-ratio {args.val_ratio})")

    # for word-level backbones, segment text and transfer labels onto seg coords
    def _apply_wordseg(ps):
        out = []
        for text, items, char_tags in ps:
            seg, seg2raw = wordseg.segment_with_offsets(text)
            out.append((seg, items, wordseg.raw_char_tags_to_seg(char_tags, seg2raw)))
        return out

    if word_segment:
        pairs = _apply_wordseg(pairs)
        val_pairs = _apply_wordseg(val_pairs)
        wordseg.print_stats()

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not tok.is_fast:
        print(f"[train_ner] {args.model}: không có fast tokenizer (PhoBERT/ViHealthBERT "
              f"đều vậy) -> dùng đường align thủ công ở mức TỪ (wordlevel_encode.py) "
              f"thay vì offset_mapping ở mức KÝ TỰ.")

    def encode(text, char_tags):
        if tok.is_fast:
            enc = tok(
                text, return_offsets_mapping=True, truncation=True,
                max_length=args.max_len, stride=args.stride,
                return_overflowing_tokens=True, padding="max_length",
            )
            if not args.crf:
                out_labels = []
                for offsets in enc["offset_mapping"]:
                    lab = []
                    for (a, b) in offsets:
                        if a == b:                      # special / padding token
                            lab.append(-100)
                        else:
                            tag = char_tags[a] if a < len(char_tags) else "O"
                            lab.append(l2i.get(tag, l2i["O"]))
                    out_labels.append(lab)
                enc.pop("offset_mapping")
                enc.pop("overflow_to_sample_mapping", None)
                enc["labels"] = out_labels
                return enc
            # --crf: pool to WORD level (one label per word, at its first
            # subword) instead of one label per raw subword token — see
            # word_pool.py for why the CRF needs this shape.
            n_windows = len(enc["input_ids"])
            word_ids_all, labels_all = [], []
            for wi in range(n_windows):
                offsets = enc["offset_mapping"][wi]
                wids_out, first_pos = collapse_to_word_starts(enc.word_ids(wi))
                wlabels = []
                for t in first_pos:
                    a = offsets[t][0]
                    tag = char_tags[a] if a < len(char_tags) else "O"
                    wlabels.append(l2i.get(tag, l2i["O"]))
                wlabels += [-100] * (args.max_len - len(wlabels))
                word_ids_all.append(wids_out)
                labels_all.append(wlabels)
            enc.pop("offset_mapping")
            enc.pop("overflow_to_sample_mapping", None)
            enc["word_ids"] = word_ids_all
            enc["labels"] = labels_all
            return enc
        # slow tokenizer: align at word granularity instead of char offsets
        words, word_tags, _spans = wl.words_and_tags_from_char_tags(text, char_tags)
        windows = wl.encode_words(tok, words, args.max_len, args.stride)
        if not args.crf:
            return {
                "input_ids": [w["input_ids"] for w in windows],
                "attention_mask": [w["attention_mask"] for w in windows],
                "labels": [
                    [l2i.get(word_tags[wid], l2i["O"]) if wid is not None else -100
                     for wid in w["word_ids"]]
                    for w in windows
                ],
            }
        # --crf: pool to WORD level, same reasoning as the fast-tokenizer branch
        word_ids_all, labels_all = [], []
        for w in windows:
            wids_out, first_pos = collapse_to_word_starts(w["word_ids"])
            wlabels = [l2i.get(word_tags[w["word_ids"][t]], l2i["O"]) for t in first_pos]
            wlabels += [-100] * (args.max_len - len(wlabels))
            word_ids_all.append(wids_out)
            labels_all.append(wlabels)
        return {
            "input_ids": [w["input_ids"] for w in windows],
            "attention_mask": [w["attention_mask"] for w in windows],
            "word_ids": word_ids_all,
            "labels": labels_all,
        }

    def build_dataset(ps):
        cols = ["input_ids", "attention_mask", "labels"] + (["word_ids"] if args.crf else [])
        rows = {c: [] for c in cols}
        for text, _items, char_tags in ps:
            enc = encode(text, char_tags)
            for i in range(len(enc["input_ids"])):
                for c in cols:
                    rows[c].append(enc[c][i])
        return Dataset.from_dict(rows)

    ds = build_dataset(pairs)
    print(f"Built {len(ds)} training windows.")
    ds_val = None
    if val_pairs:
        ds_val = build_dataset(val_pairs)
        print(f"Built {len(ds_val)} validation windows.")

    id2label = {i: l for l, i in l2i.items()}
    if args.crf:
        from .crf_model import EncoderCRFForTokenClassification
        model = EncoderCRFForTokenClassification(model_id, len(labels), id2label, l2i)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_id, num_labels=len(labels), id2label=id2label, label2id=l2i,
        )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, label_ids = eval_pred
        # CRFTrainer.prediction_step already returns decoded (2D) tag ids;
        # the baseline path returns raw (3D) per-token logits needing argmax.
        preds = logits if np.asarray(logits).ndim == 2 else np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for pred_row, label_row in zip(preds, label_ids):
            cur_l, cur_p = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                cur_l.append(labels[l])
                cur_p.append(labels[p])
            true_labels.append(cur_l)
            true_preds.append(cur_p)
        # `f1` = khớp CHỒNG LẤN + TYPE -> ĐÚNG luật hệ thống chấm, và là thứ dùng
        # để chọn checkpoint (metric_for_best_model="f1").
        # `f1_exact` = khớp span chính xác, CHỈ để theo dõi chất lượng ranh giới
        # span (nó chỉ ảnh hưởng WER = 30% điểm). Chọn model bằng nó là tự phạt
        # mình ở 70% số điểm vốn không quan tâm ranh giới span.
        precision, recall, f1 = entity_prf_overlap(true_labels, true_preds)
        _, _, f1_exact = entity_prf(true_labels, true_preds)
        return {"precision": precision, "recall": recall,
                "f1": f1, "f1_exact": f1_exact}

    # HF renamed evaluation_strategy -> eval_strategy across transformers
    # versions; try the new name first, fall back to the old one.
    common_targs = dict(
        output_dir=out_dir, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, logging_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, seed=args.seed,
        report_to="none", disable_tqdm=True,
    )
    metric_for_best = None
    greater_is_better = False
    if ds_val is not None:
        metric_for_best = "f1"
        greater_is_better = True
        common_targs.update(
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best,
            greater_is_better=greater_is_better,
        )
    eval_strategy_value = "epoch" if ds_val is not None else "no"
    try:
        targs = TrainingArguments(eval_strategy=eval_strategy_value, **common_targs)
    except TypeError:
        targs = TrainingArguments(evaluation_strategy=eval_strategy_value, **common_targs)

    if args.crf:
        from .crf_model import CRFTrainer
        trainer_cls = CRFTrainer
        # every field (input_ids/attention_mask/word_ids/labels) is already
        # padded to a fixed args.max_len at dataset-build time -> plain
        # stacking is enough, no need for DataCollatorForTokenClassification's
        # dynamic label-padding.
        collator = default_data_collator
    else:
        trainer_cls = Trainer
        collator = DataCollatorForTokenClassification(tok)

    trainer = trainer_cls(
        model=model, args=targs, train_dataset=ds, eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=compute_metrics if ds_val is not None else None,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    clean_cb = CleanEpochCallback(metric_for_best, greater_is_better, has_eval=ds_val is not None)
    trainer.add_callback(clean_cb)
    if ds_val is not None:
        print("epoch  | train_loss  | eval_loss  | f1 / P / R  | * BEST = checkpoint tốt nhất tới lúc đó")
    trainer.train()
    if ds_val is not None:
        clean_cb.silent = True  # evaluate() dưới đây chỉ để in báo cáo cuối, không phải 1 epoch mới
        final = trainer.evaluate()
        print("\nValidation cuối cùng (checkpoint tốt nhất theo F1):")
        for k, v in final.items():
            print(f"  {k}: {v}")
    if args.crf:
        # bypass Trainer.save_model — it only knows how to save a
        # PreTrainedModel; our custom wrapper has its own save_pretrained.
        model.save_pretrained(out_dir)
    else:
        trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # record how to run inference (word_segment/crf must match at inference time)
    import json as _json
    with open(Path(out_dir) / "backend_meta.json", "w", encoding="utf-8") as f:
        _json.dump({"base_model": model_id, "word_segment": word_segment,
                    "model_key": model_key, "crf": args.crf},
                   f, ensure_ascii=False, indent=2)

    print(f"\nSaved NER checkpoint to: {out_dir}")
    infer_ref = model_key if model_key in reg else out_dir
    print(f"Run inference with:  python scripts/make_submission.py --model {infer_ref}")


if __name__ == "__main__":
    main()
