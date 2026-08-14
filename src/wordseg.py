"""Offset-preserving Vietnamese word segmentation for word-level backbones
(PhoBERT / ViHealthBERT).

`segment_with_offsets(raw)` returns:
    seg   : the underscore-joined segmented text PhoBERT expects
            ("khó thở" -> "khó_thở")
    seg2raw: list len(seg); seg2raw[i] = index of the raw char that seg char i
             came from, or None for inserted separators (spaces / underscores).

With this single map we can (a) transfer raw-coordinate BIO labels onto the
segmented text for TRAINING, and (b) map predicted segmented-coordinate spans
back onto the raw text for INFERENCE — using exactly the same alignment, so the
two never drift apart.

Two segmenter backends:
  - underthesea (default; pure Python, zero setup).
  - VnCoreNLP/RDRSegmenter — what PhoBERT/ViHealthBERT were ACTUALLY pretrained
    with (the model card recommends using the same segmenter downstream for
    consistency). Verified empirically this session: on real clinical
    sentences, underthesea and VnCoreNLP disagree ~75% of the time on medical
    compounds — e.g. underthesea splits "tiền sử" wrongly across "đái tháo
    đường" as "tiền sử_đái tháo đường", while VnCoreNLP gives the correct
    "tiền_sử đái_tháo_đường"; underthesea also splits the very common symptom
    "đánh trống ngực" into 3 separate word-tokens where VnCoreNLP keeps it as
    one "đánh_trống_ngực". Needs Java + the vendored jar/models under
    vncorenlp/ (committed in this repo, see vncorenlp/README.md).

Backend selection: WORDSEG_BACKEND env var = "underthesea" | "vncorenlp" |
unset ("auto" — prefer VnCoreNLP if its jar+models+Java+the `vncorenlp` pip
package are all present, else silently fall back to underthesea).

Every segmentation is cached to disk (data/wordseg_cache/<sha1(raw)>.json),
keyed by the exact input text — so once a corpus is segmented (by anyone, the
cache is just portable JSON) no one needs Java/VnCoreNLP installed to reuse
it. This repo ships the cache for data/synth/notes/ and data/input/
pre-populated, so training/inference on the existing corpora never touches
VnCoreNLP at all — only segmenting genuinely NEW text needs it live.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from underthesea import word_tokenize
    _HAVE_UTS = True
except Exception:  # pragma: no cover
    _HAVE_UTS = False

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "wordseg_cache"
_VNCORENLP_JAR = Path(os.environ.get("VNCORENLP_JAR", _REPO_ROOT / "vncorenlp" / "VnCoreNLP-1.2.jar"))
_VNCORENLP_RDR = _VNCORENLP_JAR.parent / "models" / "wordsegmenter" / "wordsegmenter.rdr"

_vncorenlp_engine = None  # lazy singleton: one JVM per process, not per call

stats = {"cache_hit": 0, "vncorenlp": 0, "underthesea": 0}


def print_stats() -> None:
    """Summary of how segmentation was resolved this process — call after a
    batch of segment_with_offsets() calls (e.g. train_ner.py after building the
    dataset) to see at a glance whether VnCoreNLP was actually invoked live or
    everything came from the pre-populated cache."""
    print(f"[wordseg] cache_hit={stats['cache_hit']}  "
          f"vncorenlp_live={stats['vncorenlp']}  underthesea_live={stats['underthesea']}",
          file=sys.stderr)


def available() -> bool:
    return _HAVE_UTS or _vncorenlp_ready()


def _backend_choice() -> str:
    forced = os.environ.get("WORDSEG_BACKEND", "").strip().lower()
    if forced in ("underthesea", "vncorenlp"):
        return forced
    return "auto"


def _vncorenlp_ready() -> bool:
    if not _VNCORENLP_JAR.is_file() or not _VNCORENLP_RDR.is_file():
        return False
    if shutil.which("java") is None:
        return False
    try:
        import vncorenlp  # noqa: F401
    except Exception:
        return False
    return True


def _get_vncorenlp_engine():
    global _vncorenlp_engine
    if _vncorenlp_engine is None:
        import vncorenlp
        print(f"[wordseg] khởi động VnCoreNLP server ({_VNCORENLP_JAR})...", file=sys.stderr)
        _vncorenlp_engine = vncorenlp.VnCoreNLP(address=str(_VNCORENLP_JAR), annotators="wseg", quiet=True)
        atexit.register(lambda: _vncorenlp_engine.close())
    return _vncorenlp_engine


def _segment_underthesea(raw: str) -> str:
    return word_tokenize(raw, format="text")  # underscore within words, space between


def _segment_vncorenlp(raw: str) -> str:
    engine = _get_vncorenlp_engine()
    result = engine.annotate(raw)
    words = [tok["form"] for sent in result["sentences"] for tok in sent]
    return " ".join(words)


def _cache_path(raw: str) -> Path:
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def _load_cache(raw: str):
    p = _cache_path(raw)
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["seg"], d["seg2raw"]


def _save_cache(raw: str, seg: str, seg2raw: List[Optional[int]]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(raw).write_text(
        json.dumps({"seg": seg, "seg2raw": seg2raw}, ensure_ascii=False), encoding="utf-8"
    )


def _offsets_for(raw: str, seg: str) -> List[Optional[int]]:
    """Character-by-character alignment: works identically for any segmenter's
    output, since both underthesea and VnCoreNLP use the same convention
    (underscore within a word, space between words).

    A `_`/` ` in `seg` is USUALLY a separator the segmenter inserted (no raw
    counterpart) — but if the raw text itself already contains a literal `_`
    (a rare data artifact, e.g. a synthetic note that literally wrote the
    entity-type constant "CHẨN_ĐOÁN" as a header), the segmenter may pass it
    through unchanged as real content. Treating that as "inserted separator,
    skip" desyncs the cursor for the rest of the document. Peek at raw[j]: if
    it's the SAME separator char, treat it as real 1:1 content, not a gap."""
    seg2raw: List[Optional[int]] = [None] * len(seg)
    j = 0  # raw cursor
    n = len(raw)
    for i, cs in enumerate(seg):
        if cs in ("_", " "):
            if j < n and raw[j] == cs:
                # literal separator char already present in raw -> real content
                seg2raw[i] = j
                j += 1
            elif j < n and raw[j].isspace():
                # segmenter-inserted separator standing in for a raw space
                j += 1
            continue
        # skip raw whitespace that the segmenter dropped/normalized
        while j < n and raw[j].isspace():
            j += 1
        if j < n:
            seg2raw[i] = j
            j += 1
    return seg2raw


def segment_with_offsets(raw: str) -> Tuple[str, List[Optional[int]]]:
    cached = _load_cache(raw)
    if cached is not None:
        stats["cache_hit"] += 1
        return cached

    backend = _backend_choice()
    seg = None
    if backend in ("vncorenlp", "auto") and _vncorenlp_ready():
        try:
            seg = _segment_vncorenlp(raw)
            stats["vncorenlp"] += 1
        except Exception as exc:
            if backend == "vncorenlp":
                raise SystemExit(f"WORDSEG_BACKEND=vncorenlp nhưng lỗi thật: {exc}")
            print(f"[wordseg] VnCoreNLP lỗi ({exc}) -> dùng underthesea.", file=sys.stderr)
    elif backend == "vncorenlp":
        raise SystemExit(
            f"WORDSEG_BACKEND=vncorenlp nhưng chưa sẵn sàng — cần Java, "
            f"package `vncorenlp` (pip install vncorenlp), và jar/model tại "
            f"{_VNCORENLP_JAR} (+ {_VNCORENLP_RDR}). Xem vncorenlp/README.md."
        )

    if seg is None:
        if not _HAVE_UTS:
            raise SystemExit(
                "Word-level models (PhoBERT/ViHealthBERT) need `underthesea`.\n"
                "pip install underthesea"
            )
        seg = _segment_underthesea(raw)
        stats["underthesea"] += 1

    seg2raw = _offsets_for(raw, seg)
    _save_cache(raw, seg, seg2raw)
    return seg, seg2raw


def seg_span_to_raw(seg2raw: List[Optional[int]], ss: int, se: int) -> Optional[Tuple[int, int]]:
    """Map a [ss, se) span in segmented coords to [raw_start, raw_end)."""
    raw_start = None
    raw_end = None
    for k in range(ss, min(se, len(seg2raw))):
        r = seg2raw[k]
        if r is None:
            continue
        if raw_start is None:
            raw_start = r
        raw_end = r + 1
    if raw_start is None or raw_end is None:
        return None
    return raw_start, raw_end


def raw_char_tags_to_seg(raw_tags: List[str], seg2raw: List[Optional[int]]) -> List[str]:
    """Transfer per-raw-char BIO tags onto per-seg-char tags.

    Separators inserted by the segmenter (underscore/space, seg2raw is None) get
    'O' UNLESS they sit strictly inside an entity — i.e. the previous real tag is
    B-/I-X and the next real tag is I-X of the same type — in which case they
    continue the entity (I-X). Without this bridge an entity spanning a word
    boundary (e.g. "đánh trống ngực") would be split as B- O I-, producing
    dangling I- labels."""
    out: List[Optional[str]] = [
        (raw_tags[r] if (r is not None and r < len(raw_tags)) else None) for r in seg2raw
    ]
    n = len(out)
    for i in range(n):
        if out[i] is not None:
            continue
        prev = next((out[k] for k in range(i - 1, -1, -1) if out[k] is not None), "O")
        nxt = next((out[k] for k in range(i + 1, n) if out[k] is not None), "O")
        if prev != "O" and nxt.startswith("I-") and prev[2:] == nxt[2:]:
            out[i] = nxt
        else:
            out[i] = "O"
    return [t if t is not None else "O" for t in out]
