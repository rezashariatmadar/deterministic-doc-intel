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

    for idx, char in enumerate(text):
        if idx in skip_indices:
            continue

        # Normalize NFKC characters (e.g. ligatures)
        norm_chars = unicodedata.normalize("NFKC", char)
        for nc in norm_chars:
            is_punct = nc not in _QUANTITATIVE_SYMBOLS and (
                nc in string.punctuation
                or unicodedata.category(nc).startswith("P")
                or nc in _EXTRA_PUNCTUATION
            )
            if is_punct or nc.isspace():
                if not in_whitespace and chars:
                    chars.append(" ")
                    mapping.append(idx)
                    in_whitespace = True
            else:
                folded = nc.casefold()
                for fc in folded:
                    # Skip non-spacing combining marks (e.g. dot above \u0307 in Turkish 'İ')
                    if unicodedata.category(fc) == "Mn":
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
            pos = text_lower.find(quote_lower)
            if pos != -1:
                return text[pos : pos + len(clean_quote)]

        # Path 2: full normalization with character position mapping
        norm_quote = normalize_for_matching(clean_quote)
        if not norm_quote:
            return None

        norm_text, char_map = normalize_with_mapping(text)
        if not norm_text or not char_map:
            return None

        norm_pos = norm_text.find(norm_quote)
        if norm_pos != -1:
            start_orig = char_map[norm_pos]
            end_orig = char_map[norm_pos + len(norm_quote) - 1] + 1
            return text[start_orig:end_orig]

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
        quote_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", clean_quote))
        if quote_numbers and not all(n in text for n in quote_numbers):
            return None

        quote_lower = clean_quote.lower()
        q_words = re.findall(r"\w+", quote_lower)
        if not q_words:
            return None
        q_set = set(q_words)
        q_len = len(q_words)

        # Fast pre-filter: if chunk has insufficient token overlap with quote, skip sentence search
        chunk_words = set(re.findall(r"\w+", text.lower()))
        if len(q_set & chunk_words) / max(q_len, 1) < 0.35:
            return None

        sentence_matches = list(re.finditer(r"[^.!?\n]+(?:[.!?\n]+|$)", text))
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
            # Pre-filter using token overlap to prevent unnecessary SequenceMatcher calls
            if overlap / max(q_len, 1) >= 0.35 and (
                not quote_numbers or all(n in s_text for n in quote_numbers)
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
                    if not quote_numbers or all(n in pair_text for n in quote_numbers):
                        r = difflib.SequenceMatcher(None, quote_lower, pair_text.lower()).ratio()
                        if r > best_ratio:
                            best_ratio = r
                            best_slice = pair_text

        # Early return after Step 1 sentence check
        if (
            best_ratio >= threshold
            and best_slice
            and (not quote_numbers or all(n in best_slice for n in quote_numbers))
        ):
            return self._clean_boundary(best_slice, clean_quote)

        # Step 2: Phrase sliding window only on the top 1 candidate sentence by token overlap
        candidate_spans = [
            span
            for span in sent_spans
            if span[5] / max(q_len, 1) >= 0.4
            and (not quote_numbers or all(n in span[2] for n in quote_numbers))
        ]
        if candidate_spans:
            candidate_spans.sort(key=lambda s: s[5], reverse=True)
            top_span = candidate_spans[0]
            s_text = top_span[2]

            # Pre-tokenize sentence into word match objects once
            w_matches = list(re.finditer(r"\S+", s_text))
            num_w = len(w_matches)
            min_w = max(1, q_len - 3)
            max_w = min(num_w, q_len + 4)

            for w in range(min_w, max_w + 1):
                for i in range(num_w - w + 1):
                    c_start = w_matches[i].start()
                    c_end = w_matches[i + w - 1].end()
                    cand = s_text[c_start:c_end].strip()
                    if quote_numbers and not all(n in cand for n in quote_numbers):
                        continue
                    r = difflib.SequenceMatcher(None, quote_lower, cand.lower()).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_slice = cand

        if (
            best_ratio >= threshold
            and best_slice
            and (not quote_numbers or all(n in best_slice for n in quote_numbers))
        ):
            return self._clean_boundary(best_slice, clean_quote)

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

        # Tier 1: exact substring match in primary chunk
        if clean_quote in primary_chunk.text:
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
