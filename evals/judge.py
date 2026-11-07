"""Hybrid LLM-as-a-Judge and deterministic generation/hallucination evaluation service."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.query import RAGResponse
from app.services.providers.base import LLMProvider
from evals.schemas import GoldenQuery

logger = logging.getLogger("evals.judge")

REFUSAL_KEYWORDS: tuple[str, ...] = (
    "do not contain sufficient information",
    "cannot answer",
    "insufficient information",
    "not mentioned",
    "provided documents do not",
    "not enough information",
    "cannot be answered",
    "no information",
    "does not provide",
    "not found in the provided",
    "not contained in the",
    "out of scope",
    "does not mention",
    "no information is provided",
    "unable to answer",
)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial, highly rigorous evaluation judge assessing RAG responses.\n"
    "Evaluate the generated answer against the retrieved context, user question, and reference\n"
    "ground truth across 3 dimensions on a scale from 0.0 to 1.0:\n\n"
    "1. Faithfulness (0.0 to 1.0): Are all claims strictly supported by the retrieved context\n"
    "   without hallucinations? (1.0 = fully grounded, 0.0 = completely hallucinated).\n"
    "2. Answer Relevance (0.0 to 1.0): Does the response directly address the user question?\n"
    "   (1.0 = directly relevant, 0.0 = irrelevant or falsely refused an answerable question).\n"
    "3. Ground Truth Correctness (0.0 to 1.0): Does the response factually and semantically agree\n"
    "   with the reference ground truth answer? (1.0 = accurate, 0.0 = incorrect).\n\n"
    "For unanswerable / out-of-scope questions:\n"
    "- If the generated answer correctly states that documents lack sufficient information\n"
    "  (abstains), score Faithfulness 1.0, Relevance 1.0, and Correctness 1.0.\n"
    "- If the generated answer invents claims or answers an unanswerable question,\n"
    "  score Faithfulness and Correctness 0.0.\n\n"
    "Output strictly a single JSON object with the following structure:\n"
    "{\n"
    '  "faithfulness": <float between 0.0 and 1.0>,\n'
    '  "answer_relevance": <float between 0.0 and 1.0>,\n'
    '  "ground_truth_correctness": <float between 0.0 and 1.0>,\n'
    '  "reasoning": "<concise explanation of score assignments>"\n'
    "}"
)


class JudgeResult(BaseModel):
    """Evaluation result for a single generated RAG response."""

    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 - hallucination score, measuring factual grounding in retrieved context",
    )
    answer_relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="Degree to which response directly answers the user prompt",
    )
    ground_truth_correctness: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic and factual agreement with ground truth answer",
    )
    citation_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Verified citations count divided by total citations count",
    )
    abstention_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "1.0 if correctly abstained on unanswerable or answered on answerable, else 0.0"
        ),
    )
    reasoning: str = Field(
        default="",
        description="Concise rationale or LLM judge commentary",
    )


class RAGJudge:
    """Hybrid LLM-as-a-Judge and deterministic metrics evaluator."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        default_mode: str = "deterministic",
    ) -> None:
        self.llm_provider = llm_provider
        self.default_mode = default_mode

    def compute_abstention_accuracy(self, query: GoldenQuery, response: RAGResponse) -> float:
        """
        Evaluate whether the model correctly abstained or answered.
        Returns 1.0 if correctly abstained on unanswerable, or correctly answered on answerable.
        """
        answer_lower = response.answer.lower()
        is_refusal = (not response.is_answerable) or any(
            k in answer_lower for k in REFUSAL_KEYWORDS
        )

        if not query.is_answerable:
            return 1.0 if is_refusal else 0.0
        else:
            return 0.0 if is_refusal else 1.0

    def compute_citation_precision(self, query: GoldenQuery, response: RAGResponse) -> float:
        """
        Compute citation precision: verified citations count / total citations count.
        1.0 if no citations and answerable refusal, or 0 unverified citations.
        """
        unverified_matches = re.findall(r"\[unverified:\s*(\d+)\]", response.answer)
        unverified_count = len(unverified_matches)

        all_brackets = re.findall(r"\[([^\]]+)\]", response.answer)
        verified_in_text = [b for b in all_brackets if b.strip().isdigit()]
        verified_count = len(verified_in_text)

        if verified_count == 0 and len(response.citations) > 0:
            verified_count = len(response.citations)

        total_citations = verified_count + unverified_count
        if total_citations == 0:
            # No citations attempted
            return 1.0

        if unverified_count == 0:
            return 1.0

        return round(verified_count / total_citations, 4)

    def compute_deterministic_ground_truth_correctness(
        self, query: GoldenQuery, response: RAGResponse
    ) -> float:
        """Compute lexical and numeric ground-truth agreement without external LLM calls."""
        answer_lower = response.answer.lower()
        is_refusal = (not response.is_answerable) or any(
            k in answer_lower for k in REFUSAL_KEYWORDS
        )

        if not query.is_answerable:
            return 1.0 if is_refusal else 0.0

        if is_refusal:
            return 0.0

        gt_lower = re.sub(r"['’]s\b", "", query.ground_truth_answer.lower())
        ans_lower = re.sub(r"['’]s\b", "", answer_lower)

        # 1. Required substrings match
        substring_score = 1.0
        if query.required_substrings:
            hits = sum(1 for s in query.required_substrings if s.lower() in ans_lower)
            substring_score = hits / len(query.required_substrings)

        # 2. Numeric containment check
        gt_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*%?\b", gt_lower))
        numeric_score = 1.0
        if gt_numbers:
            num_hits = 0
            for num in gt_numbers:
                num_clean = num.replace(",", "")
                if num in ans_lower or num_clean in ans_lower.replace(",", ""):
                    num_hits += 1
            numeric_score = num_hits / len(gt_numbers)

        # 3. Content token overlap
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "was",
            "were",
            "are",
            "be",
            "been",
            "as",
            "that",
            "this",
            "it",
            "its",
            "has",
            "had",
            "have",
            "not",
            "under",
            "total",
        }
        gt_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in gt_lower.split()]
        gt_tokens = {w for w in gt_words if len(w) > 2 and w not in stopwords}
        ans_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in ans_lower.split()]
        ans_tokens = {w for w in ans_words if len(w) > 2 and w not in stopwords}

        token_recall = len(gt_tokens & ans_tokens) / len(gt_tokens) if gt_tokens else 1.0

        correctness = 0.40 * numeric_score + 0.35 * substring_score + 0.25 * token_recall
        return round(min(1.0, max(0.0, correctness)), 4)

    def compute_deterministic_faithfulness(
        self, query: GoldenQuery, response: RAGResponse
    ) -> float:
        """
        Deterministic grounding and hallucination check based on sources
        and citation verifications.
        """
        answer_lower = response.answer.lower()
        is_refusal = (not response.is_answerable) or any(
            k in answer_lower for k in REFUSAL_KEYWORDS
        )

        if not query.is_answerable:
            return 1.0 if is_refusal else 0.0

        if is_refusal:
            return 1.0

        unverified_matches = re.findall(r"\[unverified:\s*(\d+)\]", response.answer)
        unverified_penalty = 0.35 * len(unverified_matches)

        if not response.sources:
            return 0.0 if not response.citations else 0.5

        context_text = re.sub(r"['’]s\b", "", " ".join(s.text.lower() for s in response.sources))
        ans_words = [
            re.sub(r"^[^\w]+|[^\w]+$", "", w)
            for w in re.sub(r"['’]s\b", "", response.answer.lower()).split()
        ]
        stopwords = {
            "the",
            "and",
            "or",
            "this",
            "that",
            "with",
            "from",
            "were",
            "been",
            "have",
            "which",
            "about",
            "their",
            "under",
            "other",
        }
        content_tokens = [w for w in ans_words if len(w) > 2 and w not in stopwords]
        if not content_tokens:
            token_support = 1.0
        else:
            supported = sum(1 for w in content_tokens if w in context_text)
            token_support = supported / len(content_tokens)

        citation_score = (
            1.0
            if (response.citations and not unverified_matches)
            else (0.5 if not unverified_matches else max(0.0, 1.0 - unverified_penalty))
        )

        faithfulness = 0.6 * token_support + 0.4 * citation_score - unverified_penalty
        return round(min(1.0, max(0.0, faithfulness)), 4)

    def compute_deterministic_answer_relevance(
        self, query: GoldenQuery, response: RAGResponse
    ) -> float:
        """Deterministic relevance estimate checking query term coverage and required substrings."""
        answer_lower = response.answer.lower()
        is_refusal = (not response.is_answerable) or any(
            k in answer_lower for k in REFUSAL_KEYWORDS
        )

        if not query.is_answerable:
            return 1.0 if is_refusal else 0.2

        if is_refusal:
            return 0.0

        clean_query = re.sub(r"['’]s\b", "", query.query.lower())
        q_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in clean_query.split()]
        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "was",
            "were",
            "the",
            "a",
            "an",
            "as",
            "of",
            "in",
            "for",
            "to",
            "at",
            "explain",
            "describe",
            "calculate",
            "compare",
            "contrast",
            "determine",
            "summarize",
            "state",
            "list",
            "does",
            "did",
            "their",
            "about",
            "between",
            "many",
            "much",
            "total",
            "under",
            "more",
            "than",
            "from",
            "with",
            "into",
            "according",
            "its",
            "defined",
            "form",
            "paper",
        }
        q_tokens = [w for w in q_words if len(w) > 2 and w not in stopwords]

        ans_clean = re.sub(r"['’]s\b", "", answer_lower)
        if not q_tokens:
            token_recall = 1.0
        else:
            hits = 0
            for t in q_tokens:
                stem = t[:4] if len(t) >= 5 else t
                if t in ans_clean or stem in ans_clean:
                    hits += 1
            token_recall = hits / len(q_tokens)

        # Substring coverage
        sub_score = 1.0
        if query.required_substrings:
            sub_hits = sum(1 for s in query.required_substrings if s.lower() in ans_clean)
            sub_score = sub_hits / len(query.required_substrings)

        relevance = 0.50 * token_recall + 0.30 * sub_score + 0.20
        relevance = min(1.0, max(0.1, relevance))

        if len(response.answer.strip()) < 15:
            relevance *= 0.5

        return round(relevance, 4)

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        """Safely parse structured JSON from model output, stripping code fences."""
        cleaned = text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        return data

    async def evaluate_response(
        self,
        query: GoldenQuery,
        response: RAGResponse,
        mode: str = "llm",
    ) -> JudgeResult:
        """
        Evaluate a single RAGResponse against a GoldenQuery.

        Deterministic metrics (abstention accuracy and citation precision) are computed
        directly from the response metadata and text.
        Faithfulness, relevance, and correctness are evaluated via the LLM judge in 'llm' mode,
        with graceful fallback to deterministic metrics on parsing failure or in
        'deterministic' mode.
        """
        abstention_acc = self.compute_abstention_accuracy(query, response)
        citation_prec = self.compute_citation_precision(query, response)

        use_llm = mode.lower() == "llm" and self.llm_provider is not None

        if use_llm and self.llm_provider is not None:
            context_str = "\n".join(
                f"[Source {s.source_id}] (Doc: {s.document_title}, Page: {s.page_start}): {s.text}"
                for s in response.sources
            )
            prompt = (
                f"[User Question]\n{query.query}\n\n"
                f"[Question Is Answerable]\n{query.is_answerable}\n\n"
                f"[Retrieved Context]\n{context_str or 'No retrieved context.'}\n\n"
                f"[Generated Answer]\n{response.answer}\n\n"
                f"[Reference Ground Truth Answer]\n{query.ground_truth_answer}\n\n"
                f"Evaluate the generated answer and return the JSON evaluation:"
            )

            try:
                raw_judge_output = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    temperature=0.0,
                )
                parsed = self._extract_json_payload(raw_judge_output)

                faithfulness = round(min(1.0, max(0.0, float(parsed["faithfulness"]))), 4)
                relevance = round(min(1.0, max(0.0, float(parsed["answer_relevance"]))), 4)
                correctness = round(
                    min(1.0, max(0.0, float(parsed["ground_truth_correctness"]))), 4
                )
                reasoning = str(parsed.get("reasoning", "")).strip()

                return JudgeResult(
                    faithfulness=faithfulness,
                    answer_relevance=relevance,
                    ground_truth_correctness=correctness,
                    citation_precision=citation_prec,
                    abstention_accuracy=abstention_acc,
                    reasoning=reasoning,
                )
            except Exception as exc:
                logger.warning(
                    "LLM judge failed or returned invalid JSON (%s); "
                    "falling back to deterministic evaluation.",
                    exc,
                )
                fallback_reasoning = f"Fallback to deterministic judge: {exc}"

                faithfulness = self.compute_deterministic_faithfulness(query, response)
                relevance = self.compute_deterministic_answer_relevance(query, response)
                correctness = self.compute_deterministic_ground_truth_correctness(query, response)

                return JudgeResult(
                    faithfulness=faithfulness,
                    answer_relevance=relevance,
                    ground_truth_correctness=correctness,
                    citation_precision=citation_prec,
                    abstention_accuracy=abstention_acc,
                    reasoning=fallback_reasoning,
                )

        # Pure deterministic mode
        faithfulness = self.compute_deterministic_faithfulness(query, response)
        relevance = self.compute_deterministic_answer_relevance(query, response)
        correctness = self.compute_deterministic_ground_truth_correctness(query, response)

        return JudgeResult(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            ground_truth_correctness=correctness,
            citation_precision=citation_prec,
            abstention_accuracy=abstention_acc,
            reasoning="Evaluated via deterministic offline heuristics.",
        )
