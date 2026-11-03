"""Deterministic quote verification and multi-tier evidence repair ladder."""

import difflib
import re
import string
import unicodedata
from dataclasses import dataclass

from app.schemas.query import RetrievedChunk

# Punctuation and quotation mark characters to strip/normalize
_EXTRA_PUNCTUATION = {"—", "–", "−", "…", "“", "”", "‘", "’", "«", "»"}


def normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """
    Normalize text for matching while building a character index map back to original text.

    Strips punctuation, normalizes unicode quotes/dashes, collapses whitespace, and folds case.
    Returns:
        tuple[str, list[int]]: The normalized string and a list where each element
        is the character index in `text` corresponding to that character in the output.
    """
    if not text:
        return "", []

    chars: list[str] = []
    mapping: list[int] = []
    in_whitespace = False

    for idx, char in enumerate(text):
        # Normalize NFKC characters (e.g. ligatures)
        norm_chars = unicodedata.normalize("NFKC", char)
        for nc in norm_chars:
            is_punct = (
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

    def _normalized_match(self, quote: str, text: str) -> str | None:
        """Attempt normalized matching and return original slice from text."""
        if not quote.strip() or not text.strip():
            return None

        clean_quote = quote.strip()

        # Fast path 1: simple case-fold match
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

        quote_lower = clean_quote.lower()
        quote_words = quote_lower.split()
        q_len = len(quote_words)
        q_set = set(re.findall(r"\w+", quote_lower))
        if not q_set:
            return None

        best_slice: str | None = None
        best_ratio = 0.0

        # 1. Check individual sentences and 2-sentence pairs
        sentence_matches = list(re.finditer(r"[^.!?\n]+(?:[.!?\n]+|$)", text))
        for i, sm in enumerate(sentence_matches):
            s_slice = sm.group().strip()
            if s_slice:
                r = difflib.SequenceMatcher(None, quote_lower, s_slice.lower()).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_slice = s_slice

            if i + 1 < len(sentence_matches):
                s_pair = text[sm.start() : sentence_matches[i + 1].end()].strip()
                if s_pair:
                    r = difflib.SequenceMatcher(None, quote_lower, s_pair.lower()).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_slice = s_pair

        # 2. Sliding window of words in text
        word_matches = list(re.finditer(r"\S+", text))
        total_words = len(word_matches)

        min_w = max(1, q_len - 3)
        max_w = min(total_words, q_len + 4)

        for w in range(min_w, max_w + 1):
            for i in range(total_words - w + 1):
                start_pos = word_matches[i].start()
                end_pos = word_matches[i + w - 1].end()
                cand_slice = text[start_pos:end_pos].strip()

                cand_words = set(re.findall(r"\w+", cand_slice.lower()))
                overlap = len(q_set & cand_words)
                if overlap / max(len(q_set), 1) < 0.4:
                    continue

                r = difflib.SequenceMatcher(None, quote_lower, cand_slice.lower()).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_slice = cand_slice

        if best_ratio >= threshold and best_slice:
            # Clean up boundary punctuation if quote did not have it
            if clean_quote and clean_quote[-1] not in ".!?,:;" and best_slice[-1] in ".!?,:;":
                best_slice = best_slice.rstrip(".!?,:; ")
            if clean_quote and clean_quote[0] not in "\"'([{" and best_slice[0] in "\"'([{":
                best_slice = best_slice.lstrip("\"'([{ ")
            return best_slice

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
                c
                for c in candidate_chunks
                if c.source_id != primary_chunk.source_id or c.chunk_id != primary_chunk.chunk_id
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
