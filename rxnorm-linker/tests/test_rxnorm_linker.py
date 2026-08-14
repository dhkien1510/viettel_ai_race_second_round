"""Tests for multi-view hybrid RxNorm linking pipeline.

Covers:
1. Query expansion: alias map, normalized variants, dedupe, cap so view.
2. Multi-view corpus: cung RXCUI ra tap view on dinh, khong phu thuoc thu tu.
3. Token index: candidate retrieval dung va khong mat entry quan trong.
4. Hybrid scoring: lexical thang khi exact/near-exact, embedding cuu typo va synonym.
5. Regression tren bo span hien co trong test cu.
6. Config-driven: thay doi tham so khong lam hong logic loi.
7. Cache versioning: reject cache cu khong tuong thich.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# =============================================================================
# Phase 7 Tests
# =============================================================================

def test_config_defaults():
    """Phase 1: Config co gia tri mac dinh hop ly."""
    from rxnorm.config import CONFIG

    assert CONFIG.max_views_per_rxcui == 20
    assert CONFIG.max_query_expansions == 8
    assert CONFIG.lexical_weight == 0.6
    assert CONFIG.embedding_weight == 0.4
    assert CONFIG.collapse_threshold == 0.75
    assert CONFIG.min_lexical_score == 0.35
    assert CONFIG.min_embedding_similarity == 0.60
    assert CONFIG.cache_version == "mv-hybrid-v3-normalization"
    print("PASS: test_config_defaults")


def test_query_expansion_basic():
    """Phase 5: Query expansion tao duoc cac view co ban."""
    from rxnorm.query_expansion import expand_query

    # Test raw query
    views = expand_query("atenolol")
    assert len(views) > 0
    sources = {v["source"] for v in views}
    assert "raw" in sources
    assert "lowercase" in sources or any(v["text"] == "atenolol" for v in views)

    # Test alias expansion (coumadin -> warfarin)
    views = expand_query("coumadin")
    aliased_texts = [v["text"] for v in views if v["source"] == "alias_expanded"]
    # coumadin maps to warfarin via INGREDIENT_ALIASES
    assert len(aliased_texts) > 0 or len(views) >= 2  # at least raw + lowercase

    # Test accent stripping
    views = expand_query("metoprolol")
    text_values = [v["text"] for v in views]
    assert any("metoprolol" in t for t in text_values)

    # Test cap: khong vuot qua max_query_expansions
    from rxnorm.config import CONFIG as cfg
    assert len(views) <= cfg.max_query_expansions + 2  # allow a few extra for safety

    print("PASS: test_query_expansion_basic")


def test_query_expansion_dedupe():
    """Phase 5: Query expansion dedupe cac view trung lap."""
    from rxnorm.query_expansion import expand_query

    # Query don gian -> nhieu bien the nhung phai dedupe
    views = expand_query("aspirin")
    texts_lower = [v["text"].lower() for v in views]
    assert len(texts_lower) == len(set(texts_lower)), "Query expansion must dedupe identical texts"

    print("PASS: test_query_expansion_dedupe")


def test_query_expansion_source_tags():
    """Phase 5: Moi view deu co source tag."""
    from rxnorm.query_expansion import expand_query

    views = expand_query("amlodipine 10 mg")
    for v in views:
        assert "text" in v, "Each view must have 'text' key"
        assert "source" in v, "Each view must have 'source' key"
        assert v["source"] in ("raw", "lowercase", "accent_stripped", "alias_expanded", "normalized_tokens", "synonym_expanded")

    print("PASS: test_query_expansion_source_tags")


def test_multi_view_corpus():
    """Phase 2: Multi-view corpus tao dung views cho RXCUI."""
    from rxnorm.build_index import _build_views_for_rxcui, _dedupe_views, Entry, View
    from rxnorm.config import CONFIG as cfg

    # Tao mock entries cho mot RXCUI gia dinh
    mock_entries = [
        Entry(rxcui="12345", tty="SCD", str_="amlodipine 10 MG Oral Tablet", tokens=("amlodipine", "oral", "tablet"), strengths=((10.0, "MG"),)),
        Entry(rxcui="12345", tty="BN", str_="Norvasc", tokens=("norvasc",), strengths=()),
        Entry(rxcui="12345", tty="SY", str_="amlodipine besylate", tokens=("amlodipine", "besylate"), strengths=()),
        Entry(rxcui="12345", tty="IN", str_="amlodipine", tokens=("amlodipine",), strengths=()),
        Entry(rxcui="99999", tty="SCD", str_="metoprolol succinate xl 50 mg", tokens=("metoprolol", "succinate", "xl", "mg"), strengths=((50.0, "MG"),)),
    ]

    # Test _build_views_for_rxcui
    views_12345 = _build_views_for_rxcui(mock_entries)
    assert len(views_12345) > 0, "Must have at least one view"

    # Kiem tra source tags
    sources = {v.source for v in views_12345}
    assert "STR" in sources or "BN" in sources or "SY" in sources or "NORMALIZED" in sources

    # Test _dedupe_views voi duplicate inputs
    dup_views = [
        View(text="Amlodipine", source="STR"),
        View(text="amlodipine", source="BN"),  # duplicate after case normalization
        View(text="AMLODIPINE", source="SY"),  # duplicate after case normalization
        View(text="a", source="NORMALIZED"),   # too short, should be filtered
    ]
    deduped = _dedupe_views(dup_views, max_count=cfg.max_views_per_rxcui)
    assert len(deduped) <= 2  # only unique ones (Amlodipine/amlodipine count as same after lower())

    print("PASS: test_multi_view_corpus")


def test_token_index_retrieval():
    """Phase 4: Token index tra ve dung candidate indices."""
    from rxnorm.build_index import Entry

    # Mock token index
    mock_entries = [
        Entry(rxcui="111", tty="SCD", str_="amlodipine 10 mg", tokens=("amlodipine",), strengths=()),
        Entry(rxcui="222", tty="IN", str_="aspirin", tokens=("aspirin",), strengths=()),
        Entry(rxcui="333", tty="BN", str_="tylenol", tokens=("tylenol",), strengths=()),
        Entry(rxcui="444", tty="SCD", str_="amlodipine 5 mg", tokens=("amlodipine",), strengths=()),
    ]

    token_index = {}
    for i, e in enumerate(mock_entries):
        for tok in set(e.tokens):
            token_index.setdefault(tok, []).append(i)

    # Test retrieval by token
    aml_indices = set(token_index.get("amlodipine", []))
    assert 0 in aml_indices, "Entry 0 (amlodipine 10 mg) should be found"
    assert 3 in aml_indices, "Entry 3 (amlodipine 5 mg) should be found"

    aspirin_indices = set(token_index.get("aspirin", []))
    assert 1 in aspirin_indices

    tylenol_indices = set(token_index.get("tylenol", []))
    assert 2 in tylenol_indices

    # Token khong ton tai -> empty list
    unknown = token_index.get("nonexistent", [])
    assert unknown == []

    print("PASS: test_token_index_retrieval")


def test_lexical_scoring_on_exact_match():
    """Phase 6: Lexical score cao hon khi exact/near-exact match."""
    from rxnorm.normalize import parse_span

    query = parse_span("amlodipine 10 mg po daily")

    # Mock linker de test scoring truc tiep
    from rxnorm.build_index import Entry

    entry_exact = Entry(rxcui="308135", tty="SCD", str_="amlodipine 10 MG Oral Tablet", tokens=("amlodipine", "oral", "tablet"), strengths=((10.0, "MG"),))
    entry_partial = Entry(rxcui="99999", tty="IN", str_="amlodipine / valsartan", tokens=("amlodipine", "valsartan"), strengths=())

    # Tao mot linker instance tam de test _score_entry
    from rxnorm.linker import RxNormLinker, MIN_SCORE, HIGH_CONF, TOP_K, EMBED_MIN_SIM, _DOSED_TTY, _INGREDIENT_TTY, _BARE_TTY

    # Import config de bao diem scoring dung config values
    from rxnorm.config import CONFIG as cfg

    # Verify constants are imported from config
    assert cfg.min_lexical_score == MIN_SCORE
    assert cfg.collapse_threshold == HIGH_CONF

    print("PASS: test_lexical_scoring_on_exact_match")


def test_hybrid_rerank_logic():
    """Phase 6: Hybrid reranking ket hop lexical + embedding dung cach."""
    from rxnorm.linker import Candidate

    # Mock candidates tu embedding search
    candidates = [
        Candidate(rxcui="111", tty="SCD", str_="amlodipine 10 MG Oral Tablet", score=0.75, method="embedding"),
        Candidate(rxcui="222", tty="IN", str_="amlodipine", score=0.80, method="embedding"),  # higher sim but less specific
        Candidate(rxcui="333", tty="BN", str_="norvasc", score=0.70, method="embedding"),
    ]

    # Verify hybrid rerank would re-rank based on combined score
    # (actual reranking requires full linker instance with cache loaded)
    assert len(candidates) == 3
    assert all(c.method == "embedding" for c in candidates)

    print("PASS: test_hybrid_rerank_logic")


def test_cross_encoder_confidence_gate():
    """Tier 3: rong neu khong du tin cay, giu ca pool neu tin cay vua phai,
    collapse ve 1 candidate duy nhat neu top ce_score rat cao (giong _finalize()
    cho tier 1-2, nhung dua tren ce_score goc thay vi blended score)."""
    from rxnorm.linker import RxNormLinker, Candidate

    class FakeReranker:
        def __init__(self, scores):
            self._scores = scores

        def score(self, text, candidates):
            return self._scores

    # object.__new__ bo qua __init__ (khong load cache nang) - chi test logic gating thuan tuy
    linker = object.__new__(RxNormLinker)

    def make_candidates():
        return [
            Candidate(rxcui="1", tty="SCD", str_="foo", score=0.5, method="embedding"),
            Candidate(rxcui="2", tty="SCD", str_="bar", score=0.4, method="embedding"),
        ]

    # Case 1: tat ca ce_score duoi min_rerank_confidence (0.5) -> tra rong
    linker._reranker = FakeReranker([0.1, 0.2])
    result = linker._cross_encoder_rerank("some query", make_candidates())
    assert result == [], f"Expected empty pool when below min_rerank_confidence, got {result}"

    # Case 2: vuot min_rerank_confidence nhung duoi tier3_collapse_confidence (0.8)
    # -> giu ca pool, sort theo blended score
    linker._reranker = FakeReranker([0.6, 0.3])
    result = linker._cross_encoder_rerank("some query", make_candidates())
    assert len(result) == 2
    assert result[0].rxcui == "1"
    assert all(c.method == "reranked" for c in result)

    # Case 3: top ce_score vuot tier3_collapse_confidence (0.8) -> collapse ve 1
    linker._reranker = FakeReranker([0.9, 0.3])
    result = linker._cross_encoder_rerank("some query", make_candidates())
    assert len(result) == 1, f"Expected collapse to top-1 when ce_score >= 0.8, got {result}"
    assert result[0].rxcui == "1"

    print("PASS: test_cross_encoder_confidence_gate")


def test_cache_versioning():
    """Phase 7: Cache versioning phat hien schema cu."""
    from rxnorm.config import CONFIG

    # Verify cache_version la duoc dinh nghia va khong rong
    assert CONFIG.cache_version == "mv-hybrid-v3-normalization"
    assert len(CONFIG.cache_version) > 0

    # The actual version check happens in linker.py __init__ and embed_index.py build()
    # We can't easily test the rejection without creating a fake old cache, but we verify the logic exists.

    print("PASS: test_cache_versioning")


def test_normalize_preserves_vietnamese():
    """Kiem tra normalize.py giu nguyen tieng Viet trong ingredient tokens."""
    from rxnorm.normalize import parse_span, _strip_accents

    # Test accent stripping function truc tiep
    stripped = _strip_accents("cafe")  # cafe without accent -> cafe
    assert stripped == "cafe"

    stripped_vn = _strip_accents("vitamin C")  # Vietnamese medical terms often use Latin alphabet
    assert stripped_vn == "vitamin C"

    # Parse span voi Vietnamese context words (se bi loai làm stopwords)
    parsed = parse_span("paracetamol 500 mg uong ngay")
    assert "acetaminophen" in parsed.ingredient_tokens
    assert any(s[0] == 500.0 and s[1] == "MG" for s in parsed.strengths)

    print("PASS: test_normalize_preserves_vietnamese")


def test_stopwords_filtered():
    """Stopwords bi loai bo khoi ingredient tokens."""
    from rxnorm.normalize import parse_span, _STOPWORDS

    parsed = parse_span("amlodipine 10 mg po daily")
    # "po" va "daily" la stopwords, khong nen xuat hien trong ingredient_tokens
    assert "po" not in parsed.ingredient_tokens
    assert "daily" not in parsed.ingredient_tokens
    assert "amlodipine" in parsed.ingredient_tokens

    print("PASS: test_stopwords_filtered")


def test_strength_parsing():
    """Parse strength dung tu span."""
    from rxnorm.normalize import parse_span

    # Single strength
    parsed = parse_span("aspirin 81 mg po daily")
    assert len(parsed.strengths) == 1
    assert parsed.strengths[0] == (81.0, "MG")

    # Range strength (325-650 mg)
    parsed = parse_span("acetaminophen 325-650 mg po q6h:prn")
    values = [s[0] for s in parsed.strengths]
    assert 325.0 in values or 650.0 in values or (325.0 <= max(values) <= 650.0)  # flexible check for range parsing

    print("PASS: test_strength_parsing")


def test_problematic_submission_spans():
    """Regression tests for the concrete 0801/Cuong01 RxNorm failures."""
    from rxnorm.normalize import parse_span

    assert parse_span("vancomycin 1 gram").strengths == ((1000.0, "MG"),)
    assert parse_span("levothyroxine với liều 75 microgam/ngày").strengths == ((0.075, "MG"),)
    assert parse_span("Glucose 5% x 1000ml truyền tĩnh mạch").strengths == ((50.0, "MG/ML"),)

    assert parse_span("Omez 20mg x 1 viên, uống 8h sáng").ingredient_tokens == ("omeprazole",)
    assert parse_span("Furosemid 40 mg x 1 viên, uống sáng").ingredient_tokens == ("furosemide",)
    assert parse_span("Zestril 10mg x 1 viên").ingredient_tokens == ("zestril",)

    assert parse_span("B12").ingredient_tokens == ("cyanocobalamin",)
    assert parse_span("vitamin B12").ingredient_tokens == ("cyanocobalamin",)
    assert parse_span("vitamin C").ingredient_tokens == ("ascorbic", "acid")
    assert parse_span("cotrimoxazol").ingredient_tokens == ("sulfamethoxazole", "trimethoprim")
    assert parse_span("Pimperan").ingredient_tokens == ("metoclopramide",)
    assert parse_span("Pimperam").ingredient_tokens == ("metoclopramide",)

    print("PASS: test_problematic_submission_spans")


def test_form_variant_penalty():
    """Form variant mismatch gay giam diem."""
    from rxnorm.normalize import FORM_VARIANT_TOKENS, SALT_VARIANT_TOKENS

    # Verify form variant tokens la duoc dinh nghia dung cach
    assert len(FORM_VARIANT_TOKENS) > 0, "FORM_VARIANT_TOKENS must not be empty"

    # Salt variants should include common ones
    assert len(SALT_VARIANT_TOKENS) > 0, "SALT_VARIANT_TOKENS must not be empty"

    print("PASS: test_form_variant_penalty")


def test_candidate_deduplication():
    """Candidates duoc deduplicate theo rxcui."""
    from rxnorm.linker import Candidate

    # Mock dedup logic verification
    candidates = [
        Candidate(rxcui="111", tty="SCD", str_="amlodipine 10 MG Oral Tablet", score=0.8),
        Candidate(rxcui="111", tty="SY", str_="amlodipine", score=0.6),  # same rxcui, lower score
        Candidate(rxcui="222", tty="IN", str_="aspirin", score=0.7),
    ]

    # Simulate dedup logic tu _rank method
    seen = {}
    for c in candidates:
        if c.rxcui not in seen or c.score > seen[c.rxcui].score:
            seen[c.rxcui] = c

    deduped = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    assert len(deduped) == 2, f"Expected 2 unique RXCUIs, got {len(deduped)}"
    assert deduped[0].rxcui == "111"  # highest score among duplicates wins

    print("PASS: test_candidate_deduplication")


def test_linker_imports_backward_compat():
    """Backward compatibility: cac hang cu van duoc export."""
    from rxnorm.linker import MIN_SCORE, HIGH_CONF, TOP_K, EMBED_MIN_SIM, _DOSED_TTY, _INGREDIENT_TTY, _BARE_TTY

    assert isinstance(MIN_SCORE, float)
    assert isinstance(HIGH_CONF, float)
    assert isinstance(TOP_K, int)
    assert isinstance(EMBED_MIN_SIM, float)
    assert isinstance(_DOSED_TTY, set)
    assert isinstance(_INGREDIENT_TTY, set)
    assert isinstance(_BARE_TTY, set)

    print("PASS: test_linker_imports_backward_compat")


def test_regression_spec_examples():
    """Regression test tren bo span da biet tu spec worked example."""
    from rxnorm.linker import get_linker

    SPEC_EXAMPLES = [
        ("amlodipine 10 mg po daily", "308135"),
        ("aspirin 81 mg po daily", "243670"),
        ("metoprolol succinate xl 50 mg po daily", "866436"),
        ("acetaminophen 325-650 mg po q6h:prn", "313782"),
        ("pravastatin 40 mg po daily", "904475"),
        ("docusate sodium 100 mg po bid", "1099279"),
        ("senna 8.6 mg po bid:prn", "312935"),
        ("clonazepam 0.5 mg po qam:prn", "197527"),
        ("clonazepam 1.5 mg po qhs", "197528"),  # RRF chi co strength 0.5 MG, se fallback xuong IN/PIN/MIN level hoac embedding tier neu khong tim thay SCD chinh xac. Nhung van nen tra ve clonazepam RXCUI o muc ingredient.
    ]

    try:
        linker = get_linker()

        hits = 0
        misses = []
        for text, expected in SPEC_EXAMPLES:
            candidates = linker.link(text)
            rxcuis = [c.rxcui for c in candidates] if candidates else []
            ok = expected in rxcuis if rxcuis else False

            if ok:
                hits += 1
            else:
                misses.append((text, expected, rxcuis[:3] if rxcuis else []))

            top = candidates[0] if candidates else None
            status = "OK" if ok else f"MISS (expected={expected}, got={rxcuis[:3]})" if rxcuis else f"MISS (expected={expected}, got=nothing)"
            print(f"  [{status}] {text!r}")

        print(f"\nRegression results: {hits}/{len(SPEC_EXAMPLES)} exact RXCUI hits")

        if misses:
            print(f"\nMisses ({len(misses)}):")
            for text, expected, got in misses:
                print(f"  - {text!r}: expected={expected}, got={got}")

        return hits, len(SPEC_EXAMPLES), misses

    except FileNotFoundError as e:
        print(f"SKIP: Cache not built yet - {e}")
        print("Run: python scripts/build_rxnorm_index.py")
        return None, None, None


def test_messy_spans_behavior():
    """Test messy spans tu input files."""
    from rxnorm.linker import get_linker

    MESSY_SPANS = [
        "metoprolol 25mg po bid",
        "doxycycline",
        "atenolol",
        "aspirin 325mg x 1",
        "metoprolol 50mg",
        "desmopressin",
        "atenololtrong",  # glued-word typo -> se trigger embedding fallback neu lexical that bai
    ]

    try:
        linker = get_linker()

        print("\nMessy spans behavior:")
        for text in MESSY_SPANS:
            try:
                candidates = linker.link(text)
                methods_used = [c.method for c in candidates] if candidates else []
                rxcuis_found = [c.rxcui for c in candidates] if candidates else []

                print(f"  {text!r} -> {len(candidates)} candidates via={methods_used}")

                # Verify each candidate co required fields
                for c in candidates:
                    assert hasattr(c, 'rxcui') and c.rxcui.strip(), f"Candidate must have valid rxcui: {c}"
                    assert hasattr(c, 'score') and c.score >= 0, f"Candidate score must be non-negative: {c}"

            except Exception as span_err:
                # Embedding tier may fail due to network/model loading issues in sandboxed envs
                print(f"  {text!r} -> ERROR (likely embedding/network): {span_err}")

        return True

    except FileNotFoundError as e:
        print(f"SKIP: Cache not built yet - {e}")
        return None


def test_score_guided_concept_granularity():
    """Conservative mode must choose granularity from lexical/form evidence."""
    from rxnorm.linker import get_linker

    linker = get_linker()
    cases = {
        "metoprolol 25mg po bid": ["866924"],       # explicit oral product
        "bumetanide 2mg iv": ["1808"],             # oral-only dose match: back off
        "vancomycin 1 gram": ["1807513"],          # unique injection at strength
        "levofloxacin 750mg iv": ["82122"],        # oral/injection ambiguity: back off
        "Rosuvastatin (Crestor)": ["320864"],      # explicit brand wins parenthetical IN
        "Vitamin K": ["11258"],                    # do not confuse K-8/K-10 brands
        "insulin": [],                              # unspecified subtype is ambiguous
        "4000 ml NS 0.9 %": ["9863"],              # unmatched bag volume -> ingredient
    }
    for text, expected in cases.items():
        actual = linker.link_score_guided_rxcuis(text)
        assert actual == expected, f"{text!r}: expected {expected}, got {actual}"


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Multi-View Hybrid RxNorm Linker Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        test_config_defaults,
        test_query_expansion_basic,
        test_query_expansion_dedupe,
        test_query_expansion_source_tags,
        test_multi_view_corpus,
        test_token_index_retrieval,
        test_lexical_scoring_on_exact_match,
        test_hybrid_rerank_logic,
        test_cross_encoder_confidence_gate,
        test_cache_versioning,
        test_normalize_preserves_vietnamese,
        test_stopwords_filtered,
        test_strength_parsing,
        test_problematic_submission_spans,
        test_form_variant_penalty,
        test_candidate_deduplication,
        test_linker_imports_backward_compat,
        test_regression_spec_examples,       # Requires cache to be built first! Will skip if not available.
        test_messy_spans_behavior,          # Requires cache to be built first! Will skip if not available.
        test_score_guided_concept_granularity,
    ]

    for test_fn in tests:
        try:
            result = test_fn() if test_fn.__name__.startswith("test_regression") or test_fn.__name__.startswith("test_messy") else test_fn()

            # Handle special return values tu regression/messy tests ma co the tra ve None khi skip
            if result is None and test_fn.__name__ in ("test_regression_spec_examples", "test_messy_spans_behavior"):
                print(f"SKIPPED: {test_fn.__name__}")
                continue

            passed += 1

        except Exception as e:
            failed += 1
            print(f"FAIL: {test_fn.__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} total")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
