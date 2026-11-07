"""Token counting and cost estimation service across LLM and embedding models."""

import tiktoken

# Pricing rates per 1,000,000 tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "prompt": 0.15,
        "completion": 0.60,
        "embedding": 0.02,
    },
    "gemini-2.5-flash": {
        "prompt": 0.075,
        "completion": 0.30,
        "embedding": 0.025,
    },
    "gemini-1.5-flash": {
        "prompt": 0.075,
        "completion": 0.30,
        "embedding": 0.025,
    },
    "gpt-4o": {
        "prompt": 2.50,
        "completion": 10.00,
        "embedding": 0.02,
    },
    "text-embedding-3-small": {
        "prompt": 0.02,
        "completion": 0.02,
        "embedding": 0.02,
    },
    "text-embedding-3-large": {
        "prompt": 0.13,
        "completion": 0.13,
        "embedding": 0.13,
    },
    "gemini-embedding-001": {
        "prompt": 0.025,
        "completion": 0.025,
        "embedding": 0.025,
    },
    "default": {
        "prompt": 0.15,
        "completion": 0.60,
        "embedding": 0.02,
    },
}


class CostEstimator:
    """Estimates token counts and financial costs for API requests."""

    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        """
        Estimate token count using tiktoken with fallback to cl100k_base
        or character approximation.
        """
        if not text:
            return 0

        try:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback heuristic: roughly 4 characters per token
            return max(1, len(text) // 4)

    @staticmethod
    def estimate_cost(
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        embedding_tokens: int = 0,
    ) -> float:
        """Calculate estimated total cost in USD for the query based on model pricing."""
        pricing = MODEL_PRICING.get(model.lower(), MODEL_PRICING["default"])
        prompt_rate = pricing.get("prompt", MODEL_PRICING["default"]["prompt"])
        completion_rate = pricing.get("completion", MODEL_PRICING["default"]["completion"])
        embedding_rate = pricing.get("embedding", MODEL_PRICING["default"]["embedding"])

        total_cost = (
            (prompt_tokens * prompt_rate)
            + (completion_tokens * completion_rate)
            + (embedding_tokens * embedding_rate)
        ) / 1_000_000.0

        return round(total_cost, 8)
