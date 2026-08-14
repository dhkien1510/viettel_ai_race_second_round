"""Entity-level precision/recall/F1 over BIO/IOB2 tag sequences.

Drop-in replacement for seqeval.metrics.{precision,recall,f1}_score. seqeval's
only PyPI release (1.2.2) ships no wheel, and its legacy setup.py fails to
build on Python 3.12 in some environments (e.g. Kaggle) even with
`--no-build-isolation` — so we just don't depend on it.
"""

from __future__ import annotations


def _bio_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """(start, end, type) spans, end exclusive. Tolerates a stray I- with no
    preceding B- by treating it as if it started a new entity there."""
    spans = []
    start, cur_type = None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.append((start, i, cur_type))
            start, cur_type = i, tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            continue
        else:
            if cur_type is not None:
                spans.append((start, i, cur_type))
                start, cur_type = None, None
            if tag.startswith("I-"):
                start, cur_type = i, tag[2:]
    return set(spans)


def entity_prf(true_seqs: list[list[str]], pred_seqs: list[list[str]]) -> tuple[float, float, float]:
    """Micro-averaged (precision, recall, f1), khớp span CHÍNH XÁC.

    ⚠️ ĐỪNG dùng cái này để CHỌN CHECKPOINT. Hệ thống chấm Vòng 1 khớp khái niệm
    theo CHỒNG LẤN + TYPE, không phải span chính xác — đã kiểm chứng bằng lượt nộp
    thật (lượt v7 gọt 17 span mà J_assertion đứng im tuyệt đối). Ranh giới span chỉ
    ảnh hưởng WER, tức 30% số điểm; 70% còn lại không quan tâm.
    Chọn model bằng exact-match là tự phạt mình ở 70% số điểm và có thể vứt đúng
    checkpoint chấm cao nhất. Dùng `entity_prf_overlap` bên dưới.
    """
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(true_seqs, pred_seqs):
        true_spans = _bio_to_spans(true_tags)
        pred_spans = _bio_to_spans(pred_tags)
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def entity_prf_overlap(
    true_seqs: list[list[str]], pred_seqs: list[list[str]]
) -> tuple[float, float, float]:
    """(precision, recall, f1) khớp theo CHỒNG LẤN + TYPE — ĐÚNG luật hệ thống chấm.

    Một span dự đoán khớp một span GT khi CÙNG TYPE và có giao nhau ≥1 token.
    Ghép MỘT-ĐỐI-MỘT (tham lam theo độ chồng lấn giảm dần): dự đoán thứ hai đè lên
    cùng một khái niệm GT là THỪA, không được tính khớp lần nữa — đúng như hệ thống
    chấm (nó đẩy mẫu số |GT ∪ Pred| lên).

    Cùng luật khớp với scripts/eval_score.py, nên f1 ở đây quy thẳng ra điểm:
        J_entity = F1 / (2 − F1)   và   J_assertion = A_all × J_entity
    """
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(true_seqs, pred_seqs):
        gold = sorted(_bio_to_spans(true_tags))
        pred = sorted(_bio_to_spans(pred_tags))

        cands = []
        for i, (gs, ge, gt) in enumerate(gold):
            for j, (ps, pe, pt) in enumerate(pred):
                if gt != pt:
                    continue
                ov = min(ge, pe) - max(gs, ps)
                if ov > 0:
                    cands.append((ov, i, j))
        cands.sort(key=lambda x: -x[0])

        used_g: set[int] = set()
        used_p: set[int] = set()
        for _, i, j in cands:
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)

        tp += len(used_g)
        fp += len(pred) - len(used_p)
        fn += len(gold) - len(used_g)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
