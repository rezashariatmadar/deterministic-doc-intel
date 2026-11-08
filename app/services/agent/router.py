"""2-Tier Hybrid Semantic Intent Router for dispatching queries to agent execution paths."""

import json
import logging
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from app.schemas.query import IntentRoute
from app.services.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    """Result of semantic intent classification."""

    route: IntentRoute
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    tier: Literal["tier1_rules", "tier2_llm"] = "tier1_rules"


class SemanticIntentRouter:
    """2-tier semantic router: deterministic regex rules (<1ms) with LLM fallback."""

    # Tier 1 regex patterns
    CHITCHAT_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"^\s*(?:hi|hello|hey|howdy|greetings)(?:\s+(?:there|all|everyone|bot|assistant))?[!.]*\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*good\s+(?:morning|afternoon|evening|day)[!.]*\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:who\s+are\s+you|what\s+can\s+you\s+do|how\s+are\s+you|what\s+is\s+your\s+name|help)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:thanks|thank\s+you)(?:\s+(?:so\s+much|very\s+much|a\s+lot))?[!.]*\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:bye|goodbye|see\s+ya|farewell)[!.]*\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:tell\s+me\s+a\s+joke|write\s+a\s+poem|sing\s+a\s+song)\b",
            re.IGNORECASE,
        ),
    ]

    SQL_EXPLICIT_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(?:sql|query\s+the\s+database|financial_metrics|database\s+table|tabular\s+data)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:select\s+.*from|group\s+by|order\s+by|where\s+fiscal_year)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:sum\s+of|average\s+revenue|max\s+net\s+income|highest\s+revenue|lowest\s+eps)\b",
            re.IGNORECASE,
        ),
    ]

    HYBRID_SQL_RAG_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(?:and\s+explain\s+why|and\s+summarize\s+the\s+reasons|with\s+management\s+discussion|and\s+cite\s+risks)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:metrics?\s+and\s+(?:narrative|10-k|filing|overview|strategy))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:compare\s+(?:revenue|income|eps|margins?).*and\s+(?:explain|discuss|reasons?))\b",
            re.IGNORECASE,
        ),
    ]

    MULTI_HOP_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(?:step\s+by\s+step|multi-hop|first\s+.*then\s+.*and\s+finally)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:how\s+did\s+.*impact\s+.*and\s+what\s+was\s+the\s+management\s+response)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:correlate\s+.*with\s+.*across\s+both\s+companies)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider

    def classify_tier1(self, query: str) -> RoutingDecision | None:
        """Evaluate deterministic rules in <1ms."""
        q = query.strip()

        # 1. Chitchat & greeting patterns
        for pattern in self.CHITCHAT_PATTERNS:
            if pattern.search(q):
                return RoutingDecision(
                    route=IntentRoute.CHITCHAT_OR_OUT_OF_DOMAIN,
                    confidence=0.99,
                    reasoning="Matched deterministic conversational chitchat pattern.",
                    tier="tier1_rules",
                )

        # 2. Hybrid SQL + RAG markers
        for pattern in self.HYBRID_SQL_RAG_PATTERNS:
            if pattern.search(q):
                return RoutingDecision(
                    route=IntentRoute.HYBRID_SQL_AND_RAG,
                    confidence=0.92,
                    reasoning="Matched hybrid numeric metric and narrative explanation pattern.",
                    tier="tier1_rules",
                )

        # 3. Explicit SQL & tabular analytics markers
        for pattern in self.SQL_EXPLICIT_PATTERNS:
            if pattern.search(q):
                return RoutingDecision(
                    route=IntentRoute.STRUCTURED_SQL_ANALYTICS,
                    confidence=0.95,
                    reasoning="Matched explicit SQL/aggregation database keyword pattern.",
                    tier="tier1_rules",
                )

        # 4. Multi-hop complex reasoning markers
        for pattern in self.MULTI_HOP_PATTERNS:
            if pattern.search(q):
                return RoutingDecision(
                    route=IntentRoute.MULTI_HOP_AGENTIC_RAG,
                    confidence=0.90,
                    reasoning="Matched multi-hop / multi-step investigation pattern.",
                    tier="tier1_rules",
                )

        return None

    async def classify(self, query: str) -> RoutingDecision:
        """Route query using Tier 1 deterministic rules or Tier 2 LLM classification."""
        tier1_result = self.classify_tier1(query)
        if tier1_result is not None:
            return tier1_result

        # Tier 2: LLM classification if available
        if self.llm_provider is not None:
            tier2_decision = await self._classify_tier2(query)
            if tier2_decision is not None:
                return tier2_decision

        # Default fallback: FAST_TRACK_HYBRID_RAG
        return RoutingDecision(
            route=IntentRoute.FAST_TRACK_HYBRID_RAG,
            confidence=0.75,
            reasoning="Defaulted to standard hybrid RAG search.",
            tier="tier1_rules",
        )

    async def _classify_tier2(self, query: str) -> RoutingDecision | None:
        """Call LLM with structured output schema for ambiguous queries."""
        assert self.llm_provider is not None

        prompt = (
            "Classify the following user query into one of these 5 intent routes:\n"
            "1. CHITCHAT_OR_OUT_OF_DOMAIN - Greetings, small talk, general chit-chat.\n"
            "2. FAST_TRACK_HYBRID_RAG - Standard single-topic factual search in filings.\n"
            "3. MULTI_HOP_AGENTIC_RAG - Complex, multi-part inquiries.\n"
            "4. STRUCTURED_SQL_ANALYTICS - Queries requiring numerical aggregation or SQL.\n"
            "5. HYBRID_SQL_AND_RAG - Numbers AND textual context/explanation from 10-K filings.\n\n"
            f'Query: "{query}"\n\n'
            "Return JSON ONLY with this schema:\n"
            "{\n"
            '  "route": "<ONE_OF_THE_5_ROUTES>",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "<brief explanation>"\n'
            "}"
        )

        try:
            raw_response = await self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are an intent routing classifier. Always respond in valid JSON.",
                temperature=0.0,
            )
            # Clean JSON formatting
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            data = json.loads(cleaned)
            route_str = data.get("route", "").strip().upper()
            if route_str in IntentRoute.__members__:
                return RoutingDecision(
                    route=IntentRoute(route_str),
                    confidence=float(data.get("confidence", 0.85)),
                    reasoning=str(data.get("reasoning", "LLM classified route.")),
                    tier="tier2_llm",
                )
        except Exception as exc:
            logger.warning("Tier 2 LLM routing classification failed: %s", exc)

        return None
