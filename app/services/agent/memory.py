"""Multi-turn conversational memory with Redis sliding-window session store."""

import json
import logging
import re
import time
from typing import Any, ClassVar

from app.services.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    """Sliding-window conversational memory backed by Redis with in-memory fallback."""

    # Ambiguous pronoun and context reference patterns that require contextualization
    PRONOUN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:it|its|their|theirs|they|them|this|that|these|those|"
        r"the former|the latter|his|her|hers|the company|same year)\b",
        re.IGNORECASE,
    )
    ELLIPSIS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:what about|how about|and in|why|and what|compare that to|and for)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        redis_client: Any = None,
        llm_provider: LLMProvider | None = None,
        max_turns: int = 6,
        ttl: int = 86400,
    ) -> None:
        self.redis = redis_client
        self.llm_provider = llm_provider
        self.max_turns = max_turns
        self.ttl = ttl
        # Local in-memory session store fallback when Redis is absent
        self._local_sessions: dict[str, list[dict[str, Any]]] = {}

    def _get_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    async def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
    ) -> None:
        """Append a user-assistant conversation turn and trim to sliding window."""
        if not session_id:
            return

        turn_data = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": time.time(),
        }

        # In-memory fallback tracking
        if session_id not in self._local_sessions:
            self._local_sessions[session_id] = []
        self._local_sessions[session_id].append(turn_data)
        if len(self._local_sessions[session_id]) > self.max_turns:
            self._local_sessions[session_id] = self._local_sessions[session_id][-self.max_turns :]

        # Redis persistence
        if self.redis is not None:
            try:
                key = self._get_key(session_id)
                serialized = json.dumps(turn_data)
                pipe = self.redis.pipeline()
                pipe.rpush(key, serialized)
                pipe.ltrim(key, -self.max_turns, -1)
                pipe.expire(key, self.ttl)
                await pipe.execute()
            except Exception as exc:
                logger.warning(
                    "Failed to persist turn to Redis for session %s: %s",
                    session_id,
                    exc,
                )

    async def get_history(
        self,
        session_id: str,
        max_turns: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the last K conversation turns for a given session."""
        if not session_id:
            return []

        limit = max_turns or self.max_turns

        if self.redis is not None:
            try:
                key = self._get_key(session_id)
                raw_items = await self.redis.lrange(key, -limit, -1)
                if raw_items:
                    parsed: list[dict[str, Any]] = []
                    for item in raw_items:
                        item_str = item.decode("utf-8") if isinstance(item, bytes) else item
                        parsed.append(json.loads(item_str))
                    return parsed
            except Exception as exc:
                logger.warning("Failed to retrieve conversation history from Redis: %s", exc)

        # Fallback to local in-memory store
        turns = self._local_sessions.get(session_id, [])
        return turns[-limit:] if turns else []

    async def clear(self, session_id: str) -> None:
        """Clear conversation history for a given session ID."""
        if not session_id:
            return

        self._local_sessions.pop(session_id, None)
        if self.redis is not None:
            try:
                await self.redis.delete(self._get_key(session_id))
            except Exception as exc:
                logger.warning(
                    "Failed to clear conversation history in Redis for session %s: %s",
                    session_id,
                    exc,
                )

    def _needs_contextualization(self, query: str) -> bool:
        """Check if query contains ambiguous pronouns or elliptical follow-ups."""
        return bool(self.PRONOUN_PATTERN.search(query) or self.ELLIPSIS_PATTERN.search(query))

    async def contextualize_query(self, session_id: str | None, query: str) -> str:
        """Resolve pronouns and coreferences against conversation history before cache lookup."""
        if not session_id:
            return query

        history = await self.get_history(session_id)
        if not history:
            return query

        if not self._needs_contextualization(query):
            return query

        # Format history context
        history_lines = []
        for turn in history[-3:]:
            history_lines.append(f"User: {turn['user']}")
            preview = turn["assistant"][:160].replace("\n", " ")
            history_lines.append(f"Assistant: {preview}...")
        history_str = "\n".join(history_lines)

        # 1. Attempt LLM coreference resolution if provider available
        if self.llm_provider is not None:
            prompt = (
                "Given the preceding conversation turns and a follow-up user query, "
                "rewrite the follow-up query into a standalone, fully-qualified question "
                "that resolves all pronouns (e.g. 'it', 'they', 'its') and elliptical references.\n"
                "Do NOT answer the question. Output ONLY the rewritten standalone question.\n\n"
                f"Conversation History:\n{history_str}\n\n"
                f"Follow-up Query: {query}\n\n"
                "Rewritten Standalone Query:"
            )
            try:
                rewritten = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are an expert NLP coreference resolution assistant.",
                    temperature=0.0,
                )
                cleaned = rewritten.strip().strip('"').strip("'")
                if cleaned and len(cleaned) >= 5:
                    logger.info("Contextualized query from '%s' to '%s'", query, cleaned)
                    return cleaned
            except Exception as exc:
                logger.warning(
                    "LLM query contextualization failed, falling back to heuristic: %s",
                    exc,
                )

        # 2. Deterministic heuristic fallback
        last_turn = history[-1]
        last_user_query = last_turn["user"]

        # Search history turns in reverse for known corporate entities
        target_entity: str | None = None
        for turn in reversed(history):
            combined_turn_text = f"{turn.get('user', '')} {turn.get('assistant', '')}"
            found_entities = re.findall(
                r"\b(Apple|NVIDIA|AAPL|NVDA|Microsoft|Google)\b",
                combined_turn_text,
                re.IGNORECASE,
            )
            if found_entities:
                target_entity = found_entities[0]
                break

        if target_entity:
            # Replace possessive "its", "their", "theirs" with "{Entity}'s"
            resolved = re.sub(
                r"\b(?:its|their|theirs)\b", f"{target_entity}'s", query, flags=re.IGNORECASE
            )
            # Replace "it", "they", "them", "the company", "this/that company" with "{Entity}"
            resolved = re.sub(
                r"\b(?:it|they|them|the company|this company|that company)\b",
                target_entity,
                resolved,
                flags=re.IGNORECASE,
            )
            if resolved != query:
                return resolved

        # If elliptical query like "What about 2024?", append context from previous user query
        if self.ELLIPSIS_PATTERN.search(query):
            return f"{query} regarding {last_user_query}"

        return query
