"""Unit tests for the deterministic QuoteVerifier and multi-tier repair ladder."""

import time
import uuid

from app.schemas.query import RetrievedChunk
from app.services.retrieval.verifier import (
    QuoteVerifier,
    VerificationResult,
    normalize_for_matching,
)


def make_chunk(
    source_id: int,
    text: str,
    title: str = "Test Document",
    contextualized: str | None = None,
    element_type: str = "prose",
) -> RetrievedChunk:
    """Helper to construct a RetrievedChunk for verifier tests."""
    return RetrievedChunk(
        source_id=source_id,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        page_start=1,
        page_end=1,
        section_path="Section > Item",
        element_type=element_type,
        text=text,
        contextualized_text=contextualized if contextualized is not None else f"[{title}] {text}",
        similarity_score=0.90,
    )


def test_normalize_for_matching_strips_punct_quotes_dashes_and_folds_case() -> None:
    dirty = "  “NVIDIA’s Revenue” — was $18.1B, up 206%!  "
    norm = normalize_for_matching(dirty)
    assert norm == "nvidia s revenue was $18 1b up 206%"


def test_normalize_for_matching_empty_and_whitespace() -> None:
    assert normalize_for_matching("") == ""
    assert normalize_for_matching("   \n\t  ") == ""
    assert normalize_for_matching("!?,.-—") == ""


def test_tier1_exact_substring_match_in_primary_chunk() -> None:
    chunk = make_chunk(
        source_id=1,
        text="Data Center revenue rose 217% to $47.5B in fiscal 2024.",
    )
    verifier = QuoteVerifier()
    res = verifier.verify_citation(
        claim_or_quote="Data Center revenue rose 217% to $47.5B in fiscal 2024.",
        primary_chunk=chunk,
        candidate_chunks=[chunk],
    )

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "exact"
    assert res.verified_quote == "Data Center revenue rose 217% to $47.5B in fiscal 2024."
    assert res.rehomed_source_id is None


def test_tier2_case_folded_and_normalized_match_returns_original_slice() -> None:
    chunk = make_chunk(
        source_id=1,
        text="Operating margin expanded to 54.1% for the third quarter.",
    )
    verifier = QuoteVerifier()
    # Case variation + omitted period
    res = verifier.verify_citation(
        claim_or_quote="operating margin expanded to 54.1% for the third quarter",
        primary_chunk=chunk,
        candidate_chunks=[chunk],
    )

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "normalized"
    assert res.verified_quote == "Operating margin expanded to 54.1% for the third quarter"
    assert res.rehomed_source_id is None


def test_tier2_unicode_quotes_and_punctuation_variation() -> None:
    chunk = make_chunk(
        source_id=1,
        text='The "Hopper Architecture" delivered unprecedented compute throughput.',
    )
    verifier = QuoteVerifier()
    # Model used smart quotes
    res = verifier.verify_citation(
        claim_or_quote="The “Hopper Architecture” delivered unprecedented compute throughput.",
        primary_chunk=chunk,
        candidate_chunks=[chunk],
    )

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "normalized"
    assert "Hopper Architecture" in (res.verified_quote or "")
    assert res.rehomed_source_id is None


def test_tier3_bounded_fuzzy_match_minor_ocr_or_word_variation() -> None:
    chunk = make_chunk(
        source_id=1,
        text=(
            "The company achieved record gross margins of forty-five percent "
            "across all key operating divisions in 2023."
        ),
    )
    verifier = QuoteVerifier()
    # Minor word variation / typo
    claim = (
        "The company achieved record gross margin of forty-five percent "
        "across all key operating divisions in 2023."
    )
    res = verifier.verify_citation(
        claim_or_quote=claim,
        primary_chunk=chunk,
        candidate_chunks=[chunk],
    )

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "fuzzy"
    assert res.verified_quote is not None
    assert "gross margins" in res.verified_quote  # Extracted from original chunk text
    assert res.rehomed_source_id is None


def test_tier4_cross_chunk_rehoming_when_primary_fails() -> None:
    chunk1 = make_chunk(
        source_id=1,
        text="Automotive revenue was $281 million, down 4% sequentially.",
    )
    chunk2 = make_chunk(
        source_id=2,
        text="Professional Visualization revenue reached $416 million, up 11% sequentially.",
    )
    verifier = QuoteVerifier()
    # Claim matches chunk 2, but primary chunk passed was chunk 1
    claim = "Professional Visualization revenue reached $416 million, up 11% sequentially."
    res = verifier.verify_citation(
        claim_or_quote=claim,
        primary_chunk=chunk1,
        candidate_chunks=[chunk1, chunk2],
    )

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.rehomed_source_id == 2
    assert res.match_tier == "exact"
    assert res.verified_quote == claim


def test_tier4_cross_chunk_rehoming_with_normalized_match() -> None:
    chunk1 = make_chunk(
        source_id=1,
        text="Gaming revenue was flat sequentially.",
    )
    chunk2 = make_chunk(
        source_id=2,
        text="Data Center compute revenue grew 232% to $14.5 billion.",
    )
    verifier = QuoteVerifier()
    # Lowercase claim matching chunk 2
    claim = "data center compute revenue grew 232% to $14.5 billion"
    res = verifier.verify_citation(
        claim_or_quote=claim,
        primary_chunk=chunk1,
        candidate_chunks=[chunk1, chunk2],
    )

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.rehomed_source_id == 2
    assert res.match_tier == "normalized"
    assert res.verified_quote == "Data Center compute revenue grew 232% to $14.5 billion"


def test_tier5_hallucinated_fabricated_claim_fails() -> None:
    chunk = make_chunk(
        source_id=1,
        text="Total cash and cash equivalents stood at $5.2 billion as of year end.",
    )
    verifier = QuoteVerifier()
    hallucination = "Tesla manufactured 450,000 electric sedans during the first half of the year."
    res = verifier.verify_citation(
        claim_or_quote=hallucination,
        primary_chunk=chunk,
        candidate_chunks=[chunk],
    )

    assert res.is_verified is False
    assert res.verified_quote is None
    assert res.rehomed_source_id is None
    assert res.match_tier is None


def test_empty_or_whitespace_claim_fails() -> None:
    chunk = make_chunk(source_id=1, text="Some document text.")
    verifier = QuoteVerifier()

    res_empty = verifier.verify_citation("", chunk, [chunk])
    assert res_empty.is_verified is False
    assert res_empty.verified_quote is None

    res_whitespace = verifier.verify_citation("   \n\t ", chunk, [chunk])
    assert res_whitespace.is_verified is False
    assert res_whitespace.verified_quote is None


def test_short_string_does_not_fuzzy_match() -> None:
    chunk = make_chunk(
        source_id=1,
        text="Revenue was up 20% year over year.",
    )
    verifier = QuoteVerifier()
    # String is under 15 characters, so Tier 3 fuzzy matching must NOT trigger
    short_claim = "Revenue up 10%"
    res = verifier.verify_citation(
        claim_or_quote=short_claim,
        primary_chunk=chunk,
        candidate_chunks=[chunk],
    )
    assert res.is_verified is False


def test_verify_claim_or_quote_alias() -> None:
    chunk = make_chunk(source_id=1, text="Simple test claim text.")
    verifier = QuoteVerifier()
    assert verifier.verify_claim_or_quote == verifier.verify_citation
    res = verifier.verify_claim_or_quote("Simple test claim text.", chunk, [chunk])
    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "exact"


def test_tier2_non_ascii_turkish_i_and_ligature_normalization() -> None:
    text = "Ofis İSTANBUL merkezlidir ve operasyon devam ediyor."
    chunk = make_chunk(source_id=1, text=text)
    verifier = QuoteVerifier()
    # Turkish capital I with dot 'İ' folded to 'i' in quote
    quote = "ofis istanbul merkezlidir"
    res = verifier.verify_citation(quote, chunk, [chunk])

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "normalized"
    assert res.verified_quote == "Ofis İSTANBUL merkezlidir"

    # Ligature handling
    ligature_text = "Specialized compute ﬁles were loaded."
    ligature_chunk = make_chunk(source_id=2, text=ligature_text)
    ligature_quote = "specialized compute files were loaded"
    res_lig = verifier.verify_citation(ligature_quote, ligature_chunk, [ligature_chunk])
    assert res_lig.is_verified is True
    assert res_lig.match_tier == "normalized"
    assert res_lig.verified_quote is not None
    assert "compute ﬁles" in res_lig.verified_quote


def test_hyphenated_line_break_normalization() -> None:
    text = "This reduces oper-\nating costs by 20%."
    chunk = make_chunk(source_id=1, text=text)
    verifier = QuoteVerifier()
    quote = "operating cost"
    res = verifier.verify_citation(quote, chunk, [chunk])

    assert isinstance(res, VerificationResult)
    assert res.is_verified is True
    assert res.match_tier == "normalized"
    assert res.verified_quote is not None
    assert "oper-\nating cost" in res.verified_quote


def test_fuzzy_match_latency_on_large_candidate_pool() -> None:
    # 5 candidate chunks of 500 words each
    long_text = (
        "The quarterly financial results were reported by management today in "
        "accordance with GAAP accounting standards and statutory frameworks. "
    ) * 30  # ~510 words
    chunks = [make_chunk(source_id=i + 1, text=f"[Chunk {i + 1}] {long_text}") for i in range(5)]
    verifier = QuoteVerifier()
    unverified_claim = (
        "Tesla manufactured 450,000 electric sedans during the first half of the year."
    )

    t0 = time.perf_counter()
    res = verifier.verify_citation(unverified_claim, chunks[0], chunks)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert res.is_verified is False
    assert res.match_tier is None
    assert dt_ms < 5.0, f"Verification exceeded 5ms limit: took {dt_ms:.2f}ms"


def test_preserve_quantitative_symbols_in_normalized_and_sliced_haystack() -> None:
    haystack = (
        "The global enterprise division earned €50M, £20M in UK, "
        "¥500M in Japan, and $100M total (30%)."
    )
    chunk = make_chunk(source_id=1, text=haystack)
    verifier = QuoteVerifier()

    # Slice containing $ and %
    res = verifier.verify_citation("earned €50M and $100M total (30%)", chunk, [chunk])
    assert res.is_verified is True
    assert "$" in res.verified_quote if res.verified_quote else False
    assert "%" in res.verified_quote if res.verified_quote else False

    # Financial slice with €
    res_euro = verifier.verify_citation("division earned €50M", chunk, [chunk])
    assert res_euro.is_verified is True
    assert res_euro.verified_quote == "division earned €50M"

    # Financial slice with £
    res_pound = verifier.verify_citation("£20M in UK", chunk, [chunk])
    assert res_pound.is_verified is True
    assert res_pound.verified_quote == "£20M in UK"

    # Financial slice with ¥
    res_yen = verifier.verify_citation("¥500M in Japan", chunk, [chunk])
    assert res_yen.is_verified is True
    assert res_yen.verified_quote == "¥500M in Japan"


def test_fuzzy_match_number_verification_rejects_hallucinated_numbers() -> None:
    chunk = make_chunk(
        source_id=1,
        text=(
            "Operating margin expanded substantially to thirty percent across all core "
            "business sectors in fiscal 2024."
        ),
    )
    verifier = QuoteVerifier()

    # Claim has number 2025 instead of 2024 (high token overlap but number mismatch)
    claim_wrong_year = (
        "Operating margin expanded substantially to thirty percent across all core "
        "business sectors in fiscal 2025."
    )
    res = verifier.verify_citation(claim_wrong_year, chunk, [chunk])
    assert res.is_verified is False

    # Claim with matching number 2024 (minor word typo)
    claim_correct_year = (
        "Operating margin expanded notably to thirty percent across all core "
        "business sectors in fiscal 2024."
    )
    res2 = verifier.verify_citation(claim_correct_year, chunk, [chunk])
    assert res2.is_verified is True
    assert res2.match_tier == "fuzzy"


def test_fuzzy_match_rejects_comma_formatted_numbers_even_with_high_overlap() -> None:
    chunk = make_chunk(
        source_id=1,
        text=(
            "Total Data Center segment revenue for fiscal year was $49,000 million "
            "representing substantial growth. Footnote 45 discusses international trade."
        ),
    )
    verifier = QuoteVerifier()

    # Quote has $45,000 while sentence has $49,000. Even though '45' is in the footnote,
    # the sentence itself does not have 45000, so it must be rejected!
    claim_wrong_revenue = (
        "Total Data Center segment revenue for fiscal year was $45,000 million "
        "representing substantial growth."
    )
    res = verifier.verify_citation(claim_wrong_revenue, chunk, [chunk])
    assert res.is_verified is False


def test_verifier_strips_breadcrumb_headers_before_matching() -> None:
    chunk = make_chunk(
        source_id=1,
        text="Data Center revenue was $47.5 billion, up 217% from a year ago.",
    )
    verifier = QuoteVerifier()

    # The LLM copied the injected breadcrumb header into its quote
    quote_with_breadcrumb = (
        "[Document: NVIDIA 2024 10-K | Section: 2. Financial Results | Page: 3] "
        "Data Center revenue was $47.5 billion, up 217% from a year ago."
    )
    res = verifier.verify_citation(quote_with_breadcrumb, chunk, [chunk])
    assert res.is_verified is True
    assert res.match_tier == "exact"
    assert res.verified_quote == "Data Center revenue was $47.5 billion, up 217% from a year ago."


def test_numeric_token_extraction_units_and_signs() -> None:
    from app.services.retrieval.verifier import extract_numeric_tokens

    tokens = extract_numeric_tokens("Earned $100M, $47.5B, -32% margin, +15% growth, $47,525M.")
    assert "100m" in tokens
    assert "47.5b" in tokens
    assert "-32%" in tokens
    assert "+15%" in tokens
    assert "47525m" in tokens


def test_sign_inversion_rejected() -> None:
    chunk = make_chunk(1, "Operating margin was -32% in fiscal 2024.")
    verifier = QuoteVerifier()
    res = verifier.verify_citation("Operating margin was +32% in fiscal 2024", chunk, [chunk])
    assert not res.is_verified


def test_partial_number_substring_rejected() -> None:
    chunk = make_chunk(1, "Total revenue rose to $47500 million.")
    verifier = QuoteVerifier()
    res = verifier.verify_citation("Total revenue rose to $47", chunk, [chunk])
    assert not res.is_verified


def test_verifier_accepts_breadcrumb_year() -> None:
    chunk = make_chunk(
        1,
        text="Data Center revenue was $47.5B.",
        contextualized="[Doc: 10-K 2024 | Section: Item 7] Data Center revenue was $47.5B.",
    )
    verifier = QuoteVerifier()
    res = verifier.verify_citation("In 2024, Data Center revenue was $47.5B", chunk, [chunk])
    assert res.is_verified


def test_verifier_verifies_table_row_claim() -> None:
    table_text = "| Segment | Fiscal 2024 |\n|---|---|\n| Automotive | $1,091 |"
    chunk = make_chunk(1, text=table_text, element_type="table")
    verifier = QuoteVerifier()
    res = verifier.verify_citation("Automotive revenue was $1,091 in Fiscal 2024", chunk, [chunk])
    assert res.is_verified
