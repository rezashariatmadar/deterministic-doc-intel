"""Deterministic quote verification and multi-tier evidence repair ladder."""

import difflib
import re
import string
import unicodedata
from dataclasses import dataclass

from app.schemas.query import RetrievedChunk

# Punctuation and quotation mark characters to strip/normalize
_EXTRA_PUNCTUATION = {"—", "–", "−", "…", "“", "”", "‘", "’", "«", "»"}

# Quantitative and currency symbols to preserve during normalization
_QUANTITATIVE_SYMBOLS = {"$", "%", "€", "£", "¥"}

# Hyphenated word split across line breaks (e.g. "oper-\nating" -> "operating")
_HYPHEN_LINEBREAK_REGEX = re.compile(r"-\s*[\r\n]+\s*")

# Breadcrumb header prefix pattern injected into chunk context
_BREADCRUMB_PREFIX_REGEX = re.compile(
    r"^\[Document:[^\]]+(?:\s*\|\s*(?:Section|Page):[^\]]+)*\]\s*",
    re.IGNORECASE,
)


# Precompiled pattern for numeric tokens preserving signs, suffixes, and decimals
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([+-]?\d+(?:[.,]\d+)*(?:[kKmMbBtT]|%)?)(?![A-Za-z0-9_])"
)


def extract_numeric_tokens(text: str) -> set[str]:
    """
    Extract normalized numeric tokens
    (commas removed, decimals and financial/percentage suffixes preserved).
    """
    raw_nums = _NUMERIC_TOKEN_PATTERN.findall(text)
    tokens: set[str] = set()
    for n in raw_nums:
        clean_n = n.replace(",", "").lower()
        if any(c.isdigit() for c in clean_n):
            tokens.add(clean_n)
            base = re.sub(r"[kmb%]$", "", clean_n)
            if base != clean_n and base:
                tokens.add(base)
    return tokens


def normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """
    Normalize text for matching while building a character index map back to original text.

    Strips punctuation, normalizes unicode quotes/dashes, collapses whitespace, and folds case.
    Also removes hyphenated line breaks so words broken across lines merge cleanly.
    Returns:
        tuple[str, list[int]]: The normalized string and a list where each element
        is the character index in `text` corresponding to that character in the output.
    """
    if not text:
        return "", []

    # Identify hyphenated line break ranges to skip completely (e.g. "oper-\nating" -> "operating")
    skip_indices: set[int] = set()
    for m in _HYPHEN_LINEBREAK_REGEX.finditer(text):
        for i in range(m.start(), m.end()):
            skip_indices.add(i)

    chars: list[str] = []
    mapping: list[int] = []
    in_whitespace = False

    is_pure_ascii = text.isascii()

    for idx, char in enumerate(text):
        if idx in skip_indices:
            continue

        # Preserve '-' or '+' when immediately preceding a digit (e.g. "-32%", "+15%")
        is_attached_sign = False
        if char in ("-", "−", "+"):
            j = idx + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j].isdigit():
                is_attached_sign = True

        norm_chars = char if is_pure_ascii else unicodedata.normalize("NFKC", char)

        for nc in norm_chars:
            if is_attached_sign and nc in ("-", "−", "+"):
                sign_char = "-" if nc in ("-", "−") else "+"
                chars.append(sign_char)
                mapping.append(idx)
                in_whitespace = False
                continue

            is_punct = nc not in _QUANTITATIVE_SYMBOLS and (
                nc in string.punctuation
                or (not is_pure_ascii and unicodedata.category(nc).startswith("P"))
                or nc in _EXTRA_PUNCTUATION
            )
            if is_punct or nc.isspace():
                if not in_whitespace and chars:
                    chars.append(" ")
                    mapping.append(idx)
                    in_whitespace = True
            else:
                folded = nc.lower() if is_pure_ascii else nc.casefold()
                for fc in folded:
                    if not is_pure_ascii and unicodedata.category(fc) == "Mn":
                        continue
                    chars.append(fc)
                    mapping.append(idx)
                in_whitespace = False

    # Strip trailing whitespace
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()

    return "".join(chars), mapping


def normalize_for_matching(text: str) -> str:
    """Strip punctuation, normalize unicode quotes/dashes, collapse whitespace, and fold case."""
    norm_text, _ = normalize_with_mapping(text)
    return norm_text


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of verifying a claim or quote against retrieved chunks."""

    is_verified: bool
    verified_quote: str | None = None
    rehomed_source_id: int | None = None
    match_tier: str | None = None


class QuoteVerifier:
    """
    Deterministic quote verifier and repair ladder for citations.

    Never trusts LLM quotes as ground truth. Verifies and repairs citations against
    retrieved chunks using a 5-tier deterministic repair ladder:
    - Tier 1: exact substring match in primary chunk.
    - Tier 2: normalized match in primary chunk mapped back to original text slice.
    - Tier 3: bounded fuzzy match using SequenceMatcher across sliding windows.
    - Tier 4: cross-chunk re-homing to other candidate chunks.
    - Tier 5: unverified (hallucination or ungrounded claim).
    """

    @staticmethod
    def _clean_boundary(best_slice: str, clean_quote: str) -> str:
        """Clean boundary punctuation if clean_quote did not have it."""
        if clean_quote and clean_quote[-1] not in ".!?,:;" and best_slice[-1] in ".!?,:;":
            best_slice = best_slice.rstrip(".!?,:; ")
        if clean_quote and clean_quote[0] not in "\"'([{" and best_slice[0] in "\"'([{":
            best_slice = best_slice.lstrip("\"'([{ ")
        return best_slice

    def _normalized_match(self, quote: str, text: str) -> str | None:
        """Attempt normalized matching and return original slice from text."""
        if not quote.strip() or not text.strip():
            return None

        clean_quote = quote.strip()

        # Fast path 1: simple case-fold match for pure ASCII text only (avoids unicode drift)
        if clean_quote.isascii() and text.isascii():
            quote_lower = clean_quote.lower()
            text_lower = text.lower()
            pos = 0
            while True:
                pos = text_lower.find(quote_lower, pos)
                if pos == -1:
                    break
                end = pos + len(clean_quote)
                is_num_cont = False
                if end < len(text) and (
                    text[end].isdigit()
                    or (text[end] in (".", ",") and end + 1 < len(text) and text[end + 1].isdigit())
                ):
                    is_num_cont = True
                if pos > 0 and text[pos - 1].isdigit():
                    is_num_cont = True
                if not is_num_cont:
                    return text[pos:end]
                pos += 1

        # Path 2: full normalization with character position mapping
        norm_quote = normalize_for_matching(clean_quote)
        if not norm_quote:
            return None

        norm_text, char_map = normalize_with_mapping(text)
        if not norm_text or not char_map:
            return None

        norm_pos = 0
        while True:
            norm_pos = norm_text.find(norm_quote, norm_pos)
            if norm_pos == -1:
                break
            start_orig = char_map[norm_pos]
            end_orig = char_map[norm_pos + len(norm_quote) - 1] + 1
            is_num_cont = False
            if end_orig < len(text) and (
                text[end_orig].isdigit()
                or (
                    text[end_orig] in (".", ",")
                    and end_orig + 1 < len(text)
                    and text[end_orig + 1].isdigit()
                )
            ):
                is_num_cont = True
            if start_orig > 0 and text[start_orig - 1].isdigit():
                is_num_cont = True
            if not is_num_cont:
                return text[start_orig:end_orig]
            norm_pos += 1

        return None

    def _fuzzy_match(
        self,
        quote: str,
        text: str,
        threshold: float = 0.75,
    ) -> str | None:
        """Bounded fuzzy match using difflib.SequenceMatcher across sliding windows."""
        clean_quote = quote.strip()
        if len(clean_quote) < 15 or not text.strip():
            return None

        # Verify numbers present in quote: require that they appear in the candidate text
        quote_numbers = extract_numeric_tokens(clean_quote)
        if quote_numbers:
            # Fast rejection string check before full regex tokenization on chunk text
            text_lower_nocomma = text.lower().replace(",", "")
            if not all(num in text_lower_nocomma for num in quote_numbers):
                return None
            chunk_numbers = extract_numeric_tokens(text)
            if not quote_numbers.issubset(chunk_numbers):
                return None

        quote_lower = clean_quote.lower()
        q_words = re.findall(r"\w+", quote_lower)
        if not q_words:
            return None
        q_set = set(q_words)
        q_len = len(q_words)

        if quote_numbers and not quote_numbers.issubset(extract_numeric_tokens(text)):
            return None

        # Fast pre-filter: if chunk has insufficient token overlap with quote, skip sentence search
        chunk_words = set(re.findall(r"\w+", text.lower()))
        if len(q_set & chunk_words) / max(q_len, 1) < 0.35:
            return None

        sentence_matches = list(re.finditer(r"(?:[^\n.!?]|(?<=\d)\.(?=\d))+", text))
        if not sentence_matches:
            return None

        best_slice: str | None = None
        best_ratio = 0.0

        # Step 1: Pre-filter sentences and evaluate sliding sentence windows
        sent_spans: list[tuple[int, int, str, str, set[str], int]] = []
        for sm in sentence_matches:
            s = sm.group().strip()
            if s:
                s_lower = s.lower()
                s_set = set(re.findall(r"\w+", s_lower))
                overlap = len(q_set & s_set)
                sent_spans.append((sm.start(), sm.end(), s, s_lower, s_set, overlap))

        for i in range(len(sent_spans)):
            start, _, s_text, s_lower, s_set, overlap = sent_spans[i]
            # Pre-filter using token overlap and numeric consistency (lazy number extraction)
            if overlap / max(q_len, 1) >= 0.35 and (
                not quote_numbers or quote_numbers.issubset(extract_numeric_tokens(s_text))
            ):
                r = difflib.SequenceMatcher(None, quote_lower, s_lower).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_slice = s_text

            if i + 1 < len(sent_spans):
                _, next_end, _, _, next_set, _ = sent_spans[i + 1]
                pair_overlap = len(q_set & (s_set | next_set))
                if pair_overlap / max(q_len, 1) >= 0.35:
                    pair_text = text[start:next_end].strip()
                    pair_nums = extract_numeric_tokens(pair_text)
                    if not quote_numbers or quote_numbers.issubset(pair_nums):
                        r = difflib.SequenceMatcher(None, quote_lower, pair_text.lower()).ratio()
                        if r > best_ratio:
                            best_ratio = r
                            best_slice = pair_text

        # Early return after Step 1 sentence check
        if (
            best_ratio >= threshold
            and best_slice
            and (not quote_numbers or quote_numbers.issubset(extract_numeric_tokens(best_slice)))
        ):
            return self._clean_boundary(best_slice, clean_quote)

        # Step 2: Phrase sliding window only on candidate sentences matching numbers
        candidate_spans = [
            span
            for span in sent_spans
            if span[5] / max(q_len, 1) >= 0.4
            and (not quote_numbers or quote_numbers.issubset(extract_numeric_tokens(span[2])))
        ]
        if candidate_spans:
            candidate_spans.sort(key=lambda s: s[5], reverse=True)
            top_span = candidate_spans[0]
            s_text = top_span[2]

            # Pre-tokenize sentence into word match objects once
            w_matches = list(re.finditer(r"\S+", s_text))
            num_w = len(w_matches)
            min_w = max(1, q_len - 3)
            max_w = min(num_w, max(q_len + 6, int(q_len * 2.0)))

            for w in range(min_w, max_w + 1):
                for i in range(num_w - w + 1):
                    c_start = w_matches[i].start()
                    c_end = w_matches[i + w - 1].end()
                    cand = s_text[c_start:c_end].strip()
                    cand_nums = extract_numeric_tokens(cand)
                    if quote_numbers and not quote_numbers.issubset(cand_nums):
                        continue
                    r = difflib.SequenceMatcher(None, quote_lower, cand.lower()).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_slice = cand

        effective_threshold = threshold if not quote_numbers else min(threshold, 0.65)
        if (
            best_ratio >= effective_threshold
            and best_slice
            and (not quote_numbers or quote_numbers.issubset(extract_numeric_tokens(best_slice)))
        ):
            return self._clean_boundary(best_slice, clean_quote)

        return None

    def _verify_table_claim(self, claim: str, table_text: str) -> str | None:
        """Verify natural language claim against structured markdown table cells."""
        claim_nums = extract_numeric_tokens(claim)
        table_nums = extract_numeric_tokens(table_text)
        if claim_nums and not claim_nums.issubset(table_nums):
            return None

        lines = [
            line.strip()
            for line in table_text.splitlines()
            if line.strip()
            and not line.startswith("|-")
            and not line.startswith("| -")
            and not set(line.replace("|", "").strip()) <= {"-", " "}
        ]
        if not lines:
            return None

        header_tokens = set(re.findall(r"\w+", lines[0].lower()))
        claim_tokens = set(re.findall(r"\w+", claim.lower()))

        for row in lines[1:]:
            row_tokens = set(re.findall(r"\w+", row.lower()))
            row_nums = extract_numeric_tokens(row)
            if (claim_nums and claim_nums.intersection(row_nums)) or (claim_tokens & row_tokens):
                overlap = claim_tokens & (row_tokens | header_tokens)
                if len(overlap) >= min(2, len(claim_tokens)):
                    return row
        return None

    def verify_citation(
        self,
        claim_or_quote: str,
        primary_chunk: RetrievedChunk,
        candidate_chunks: list[RetrievedChunk] | None = None,
    ) -> VerificationResult:
        """
        Verify claim or quote against primary chunk and candidate chunks
        using the deterministic evidence verification ladder.
        """
        if not claim_or_quote or not claim_or_quote.strip():
            return VerificationResult(
                is_verified=False,
                verified_quote=None,
                rehomed_source_id=None,
                match_tier=None,
            )

        clean_quote = claim_or_quote.strip()
        clean_quote = _BREADCRUMB_PREFIX_REGEX.sub("", clean_quote).strip()

        # Tier 1: exact substring match in primary chunk with boundary check
        pos = 0
        found_exact = False
        while True:
            pos = primary_chunk.text.find(clean_quote, pos)
            if pos == -1:
                break
            end = pos + len(clean_quote)
            is_num_cont = False
            if end < len(primary_chunk.text) and (
                primary_chunk.text[end].isdigit()
                or (
                    primary_chunk.text[end] in (".", ",")
                    and end + 1 < len(primary_chunk.text)
                    and primary_chunk.text[end + 1].isdigit()
                )
            ):
                is_num_cont = True
            if pos > 0 and primary_chunk.text[pos - 1].isdigit():
                is_num_cont = True
            if not is_num_cont:
                found_exact = True
                break
            pos += 1

        if found_exact:
            return VerificationResult(
                is_verified=True,
                verified_quote=clean_quote,
                rehomed_source_id=None,
                match_tier="exact",
            )

        # Tier 2: normalized match in primary chunk
        norm_res = self._normalized_match(clean_quote, primary_chunk.text)
        if norm_res is not None:
            return VerificationResult(
                is_verified=True,
                verified_quote=norm_res,
                rehomed_source_id=None,
                match_tier="normalized",
            )

        # Tier 3: bounded fuzzy match in primary chunk
        fuzzy_res = self._fuzzy_match(clean_quote, primary_chunk.text)
        if fuzzy_res is not None:
            return VerificationResult(
                is_verified=True,
                verified_quote=fuzzy_res,
                rehomed_source_id=None,
                match_tier="fuzzy",
            )

        # Contextualized text fallback (verifies breadcrumbs: year, section, document title)
        if (
            primary_chunk.contextualized_text
            and primary_chunk.contextualized_text != primary_chunk.text
        ):
            context_norm = self._normalized_match(clean_quote, primary_chunk.contextualized_text)
            if context_norm is not None:
                return VerificationResult(
                    is_verified=True,
                    verified_quote=context_norm,
                    rehomed_source_id=None,
                    match_tier="normalized",
                )
            context_fuzzy = self._fuzzy_match(clean_quote, primary_chunk.contextualized_text)
            if context_fuzzy is not None:
                return VerificationResult(
                    is_verified=True,
                    verified_quote=context_fuzzy,
                    rehomed_source_id=None,
                    match_tier="fuzzy",
                )

        # Structured table claim verification
        if primary_chunk.element_type == "table":
            table_row = self._verify_table_claim(clean_quote, primary_chunk.text)
            if table_row is not None:
                return VerificationResult(
                    is_verified=True,
                    verified_quote=table_row,
                    rehomed_source_id=None,
                    match_tier="table",
                )

        # Tier 4: cross-chunk re-homing across other candidate chunks
        if candidate_chunks:
            other_candidates = [
                c for c in candidate_chunks if c.source_id != primary_chunk.source_id
            ]

            # Check Tier 1 across candidates
            for cand in other_candidates:
                if clean_quote in cand.text:
                    return VerificationResult(
                        is_verified=True,
                        verified_quote=clean_quote,
                        rehomed_source_id=cand.source_id,
                        match_tier="exact",
                    )

            # Check Tier 2 across candidates
            for cand in other_candidates:
                cand_norm_res = self._normalized_match(clean_quote, cand.text)
                if cand_norm_res is not None:
                    return VerificationResult(
                        is_verified=True,
                        verified_quote=cand_norm_res,
                        rehomed_source_id=cand.source_id,
                        match_tier="normalized",
                    )

            # Check Tier 3 across candidates
            for cand in other_candidates:
                cand_fuzzy_res = self._fuzzy_match(clean_quote, cand.text)
                if cand_fuzzy_res is not None:
                    return VerificationResult(
                        is_verified=True,
                        verified_quote=cand_fuzzy_res,
                        rehomed_source_id=cand.source_id,
                        match_tier="fuzzy",
                    )

        # Tier 5: unverified
        return VerificationResult(
            is_verified=False,
            verified_quote=None,
            rehomed_source_id=None,
            match_tier=None,
        )

    # Alias as specified by interface
    verify_claim_or_quote = verify_citation
