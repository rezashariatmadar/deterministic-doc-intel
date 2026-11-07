"""Single-call query rewriting, keyword extraction, and optional HyDE generation."""

import json
import logging
import re
from dataclasses import dataclass

from app.services.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class TransformedQuery:
    """Encapsulates optimized query representations for dense and sparse retrieval."""

    original_query: str
    semantic_query: str
    lexical_query: str
    sub_queries: list[str]
    hypothetical_passage: str | None = None


_JSON_BLOCK_REGEX = re.compile(r"\{.*\}", re.DOTALL)


class QueryTransformer:
    """
    Optimizes queries using an LLM in a single pass to minimize latency.
    Extracts semantic expansions for dense retrieval and keywords for BM25.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider

    async def transform(
        self,
        query: str,
        enable_hyde: bool = False,
    ) -> TransformedQuery:
        """Transform user query into semantic, lexical, and sub-queries."""
        clean_q = query.strip()
        fallback = TransformedQuery(
            original_query=clean_q,
            semantic_query=clean_q,
            lexical_query=clean_q,
            sub_queries=[clean_q],
            hypothetical_passage=None,
        )

        if not self.llm_provider or not clean_q:
            return fallback

        hyde_prompt = (
            '4. "hypothetical_passage": A short 1-2 sentence plausible factual passage '
            "answering the query."
            if enable_hyde
            else ""
        )

        prompt = (
            "You are a search query optimizer for a technical & financial document retrieval system.\n"
            "Analyze the user query and output a strict JSON object with these fields:\n"
            '1. "semantic_query": An expanded, clear natural language query optimal for vector embedding.\n'
            '2. "lexical_query": Essential keywords, exact terms in quotes, entities, and numerical '
            "identifiers optimal for BM25 keyword search.\n"
            '3. "sub_queries": A list of 1 to 3 distinct sub-questions if the query is multi-part or '
            "comparative. If the query is simple, provide just the original query.\n"
            f"{hyde_prompt}\n\n"
            f"User Query: {clean_q}\n\n"
            "Output ONLY valid JSON:"
        )

        try:
            raw_resp = await self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a query rewrite assistant. Output valid JSON only.",
                temperature=0.0,
            )
            # Extract JSON block if wrapped in markdown formatting
            match = _JSON_BLOCK_REGEX.search(raw_resp)
            if not match:
                logger.warning("QueryTransformer: LLM response did not contain JSON block.")
                return fallback

            data = json.loads(match.group())
            sem = str(data.get("semantic_query") or clean_q).strip()
            lex = str(data.get("lexical_query") or clean_q).strip()
            subs = data.get("sub_queries")
            if not isinstance(subs, list) or not subs:
                subs = [clean_q]
            else:
                subs = [str(s).strip() for s in subs if str(s).strip()] or [clean_q]

            hyde_passage = (
                str(data.get("hypothetical_passage")).strip()
                if enable_hyde and data.get("hypothetical_passage")
                else None
            )

            return TransformedQuery(
                original_query=clean_q,
                semantic_query=sem,
                lexical_query=lex,
                sub_queries=subs,
                hypothetical_passage=hyde_passage,
            )
        except Exception as exc:
            logger.warning("QueryTransformer failed (%s). Falling back to raw query.", exc)
            return fallback
