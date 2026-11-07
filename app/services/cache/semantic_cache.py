"""Redis Semantic Cache Layer with In-Process Cosine Similarity and Cascade Invalidation."""

import hashlib
import json
import logging
import math
import time
import uuid
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.schemas.query import QueryFilters, RAGResponse

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    Semantic cache layer storing grounded RAG answers indexed by query embeddings.

    Uses Redis hashes scoped by canonical search mode and filters, evaluating in-process
    dot-product cosine similarity over up to 200 items per scope.
    Tracks document IDs in reverse index sets for cascade invalidation upon document deletion.
    """

    def __init__(self, redis: "Redis | Any", threshold: float = 0.94, ttl: int = 86400) -> None:
        self.redis = redis
        self.threshold = threshold
        self.ttl = ttl

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        """Compute L2 unit norm of a vector for in-process cosine similarity."""
        norm = math.sqrt(math.fsum(x * x for x in vec))
        if norm <= 1e-12:
            return vec
        return [x / norm for x in vec]

    def canonical_scope_hash(
        self,
        search_mode: str,
        filters: dict[str, Any] | QueryFilters | None = None,
        top_k: int = 5,
        enable_reranker: bool | None = None,
    ) -> str:
        """
        Produce a deterministic SHA-256 hex digest of sorted canonical filters, search mode,
        top_k, and enable_reranker parameters.
        """
        if filters is None:
            raw_filters: dict[str, Any] = {}
        elif hasattr(filters, "model_dump"):
            raw_filters = filters.model_dump(exclude_none=True)
        elif isinstance(filters, dict):
            raw_filters = {k: v for k, v in filters.items() if v is not None}
        else:
            raw_filters = {}

        payload = {
            "search_mode": search_mode,
            "filters": raw_filters,
            "top_k": top_k,
            "enable_reranker": enable_reranker,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def lookup(
        self, query_vector: list[float], scope_hash: str
    ) -> tuple[RAGResponse | None, float]:
        """
        Look up the nearest cached answer within a scope hash using cosine similarity.

        Returns (RAGResponse, similarity_score) if similarity >= threshold,
        or (None, max_similarity_score) on cache miss.
        Fail-open design: logs a warning and returns (None, 0.0) on Redis network errors.
        """
        try:
            scope_key = f"semcache:scope:{scope_hash}"
            entries = await self.redis.hgetall(scope_key)
            if not entries:
                return None, 0.0

            normalized_query = self._l2_normalize(query_vector)
            best_score = -float("inf")
            best_response_data: dict[str, Any] | None = None

            # Evaluate up to 200 entries per scope
            items = list(entries.values())[:200]
            for raw_val in items:
                if isinstance(raw_val, bytes):
                    raw_val = raw_val.decode("utf-8")
                entry = json.loads(raw_val)
                cached_vec: list[float] | None = entry.get("vector")
                if not cached_vec:
                    continue

                # In-process dot-product of L2-normalized vectors
                normalized_cached = self._l2_normalize(cached_vec)
                score = math.fsum(
                    a * b for a, b in zip(normalized_query, normalized_cached, strict=False)
                )
                if score > best_score:
                    best_score = score
                    best_response_data = entry.get("response")

            if best_score >= self.threshold and best_response_data is not None:
                rag_response = RAGResponse.model_validate(best_response_data)
                return rag_response, best_score

            return None, (best_score if best_score != -float("inf") else 0.0)
        except Exception as exc:
            logger.warning("Redis error during semantic cache lookup: %s", exc)
            return None, 0.0

    async def store(
        self, query_vector: list[float], scope_hash: str, response: RAGResponse
    ) -> bool:
        """
        Store a verified RAG answer and query embedding into the semantic cache.

        Validation guards prevent storing unanswerable answers, low confidence responses (< 0.60),
        or responses with unverified citation tags.
        Maintains reverse indices under semcache:doc:{doc_id} for cascading invalidation.
        Returns True if stored, False otherwise.
        """
        # Grounding validation guards
        if not response.is_answerable:
            return False
        if response.confidence_score < 0.60:
            return False
        if "[unverified:" in response.answer:
            return False
        for citation in response.citations:
            if citation.supporting_quote and "[unverified:" in citation.supporting_quote:
                return False

        # Extract document IDs from citations and sources
        doc_ids: set[str] = set()
        for citation in response.citations:
            if citation.document_id is not None:
                doc_ids.add(str(citation.document_id))
        for source in response.sources:
            if source.document_id is not None:
                doc_ids.add(str(source.document_id))

        entry_id = str(uuid.uuid4())
        normalized_vector = self._l2_normalize(query_vector)
        payload = {
            "vector": normalized_vector,
            "response": response.model_dump(mode="json"),
            "timestamp": time.time(),
            "document_ids": sorted(doc_ids),
        }
        serialized = json.dumps(payload)

        try:
            scope_key = f"semcache:scope:{scope_hash}"

            # Bound Redis Hash size to 200 items by evicting oldest entries
            try:
                current_len = await self.redis.hlen(scope_key)
                if current_len >= 200:
                    entries = await self.redis.hgetall(scope_key)
                    parsed_entries: list[tuple[str, float]] = []
                    for k, v in entries.items():
                        k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                        v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                        try:
                            entry_data = json.loads(v_str)
                            ts = float(entry_data.get("timestamp", 0.0))
                        except Exception:
                            ts = 0.0
                        parsed_entries.append((k_str, ts))
                    parsed_entries.sort(key=lambda x: x[1])
                    excess = (current_len - 200) + 1
                    to_delete = [item[0] for item in parsed_entries[:excess]]
                    if to_delete:
                        await self.redis.hdel(scope_key, *to_delete)
            except Exception as evict_err:
                logger.debug("Failed eviction check during semantic cache store: %s", evict_err)

            await self.redis.hset(scope_key, entry_id, serialized)
            await self.redis.expire(scope_key, self.ttl)

            for doc_id in doc_ids:
                doc_key = f"semcache:doc:{doc_id}"
                await self.redis.sadd(doc_key, f"{scope_hash}:{entry_id}")
                await self.redis.expire(doc_key, self.ttl)

            return True
        except Exception as exc:
            logger.warning("Redis error during semantic cache store: %s", exc)
            return False

    async def invalidate_document(self, document_id: UUID | str) -> int:
        """
        Invalidate all cached scope entries referencing a given document ID.

        Reads members from semcache:doc:{doc_id}, removes each entry from its
        respective semcache:scope:{scope_hash}, deletes the document set,
        and returns the count of invalidated entries.
        """
        doc_id_str = str(document_id)
        doc_key = f"semcache:doc:{doc_id_str}"
        try:
            members = await self.redis.smembers(doc_key)
            if not members:
                return 0

            invalidated_count = 0
            for member in members:
                if isinstance(member, bytes):
                    member = member.decode("utf-8")
                if ":" in member:
                    scope_hash, entry_id = member.split(":", 1)
                    scope_key = f"semcache:scope:{scope_hash}"
                    deleted = await self.redis.hdel(scope_key, entry_id)
                    if deleted:
                        invalidated_count += 1

            await self.redis.delete(doc_key)
            return invalidated_count
        except Exception as exc:
            logger.warning("Redis error during document cache invalidation: %s", exc)
            return 0
