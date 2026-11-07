"""Unit tests for Redis semantic cache layer with cascade invalidation and grounding guards."""

import math
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.schemas.query import Citation, RAGResponse, RetrievedChunk
from app.services.cache.semantic_cache import SemanticCacheService


class MockRedis:
    """In-memory async Redis double simulating hashes and sets."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}
        self.sets: dict[str, set[str]] = {}
        self.expirations: dict[str, int] = {}

    async def hset(self, name: str, key: str, value: Any) -> int:
        if name not in self.hashes:
            self.hashes[name] = {}
        self.hashes[name][key] = value
        return 1

    async def hgetall(self, name: str) -> dict[str, Any]:
        return dict(self.hashes.get(name, {}))

    async def hdel(self, name: str, *keys: str) -> int:
        count = 0
        if name in self.hashes:
            for k in keys:
                if k in self.hashes[name]:
                    del self.hashes[name][k]
                    count += 1
        return count

    async def sadd(self, name: str, *values: str) -> int:
        if name not in self.sets:
            self.sets[name] = set()
        count = 0
        for val in values:
            if val not in self.sets[name]:
                self.sets[name].add(val)
                count += 1
        return count

    async def smembers(self, name: str) -> set[str]:
        return set(self.sets.get(name, set()))

    async def delete(self, *names: str) -> int:
        count = 0
        for n in names:
            if n in self.hashes:
                del self.hashes[n]
                count += 1
            if n in self.sets:
                del self.sets[n]
                count += 1
        return count

    async def expire(self, name: str, time: int) -> bool:
        self.expirations[name] = time
        return True


def create_sample_rag_response(
    answer: str = "NVIDIA revenue grew to $47.5B [1].",
    is_answerable: bool = True,
    confidence_score: float = 0.95,
    doc_id: uuid.UUID | None = None,
) -> RAGResponse:
    target_doc_id = doc_id or uuid.uuid4()
    return RAGResponse(
        answer=answer,
        is_answerable=is_answerable,
        confidence_score=confidence_score,
        citations=[
            Citation(
                source_id=1,
                document_id=target_doc_id,
                document_title="NVIDIA 10-K",
                page_start=45,
                page_end=45,
                section_path="Item 7",
                supporting_quote="revenue grew to $47.5B",
            )
        ],
        sources=[
            RetrievedChunk(
                source_id=1,
                chunk_id=uuid.uuid4(),
                document_id=target_doc_id,
                document_title="NVIDIA 10-K",
                page_start=45,
                page_end=45,
                section_path="Item 7",
                element_type="narrative_text",
                text="Revenue grew to $47.5B.",
                contextualized_text="Revenue grew to $47.5B.",
                similarity_score=0.92,
            )
        ],
        latency_ms=150.0,
    )


@pytest.mark.asyncio
async def test_cosine_similarity_hit_and_miss() -> None:
    redis = MockRedis()
    cache = SemanticCacheService(redis=redis, threshold=0.94, ttl=3600)
    scope_hash = cache.canonical_scope_hash(search_mode="hybrid")

    v1 = [1.0, 0.0, 0.0]
    response = create_sample_rag_response()

    # Empty cache lookup returns miss
    res, score = await cache.lookup(v1, scope_hash)
    assert res is None
    assert score == 0.0

    # Store entry
    stored = await cache.store(v1, scope_hash, response)
    assert stored is True

    # Exact match: similarity = 1.0 >= 0.94 -> Hit
    hit_res, hit_score = await cache.lookup(v1, scope_hash)
    assert hit_res is not None
    assert pytest.approx(hit_score, abs=1e-5) == 1.0
    assert hit_res.answer == response.answer
    assert hit_res.confidence_score == response.confidence_score

    # Near match: cos(theta) = 0.95 >= 0.94 -> Hit
    # v2 = [0.95, sqrt(1 - 0.95^2), 0.0]
    y_near = math.sqrt(1.0 - 0.95**2)
    v_near = [0.95, y_near, 0.0]
    near_res, near_score = await cache.lookup(v_near, scope_hash)
    assert near_res is not None
    assert pytest.approx(near_score, abs=1e-5) == 0.95

    # Distant match: cos(theta) = 0.90 < 0.94 -> Miss
    y_far = math.sqrt(1.0 - 0.90**2)
    v_far = [0.90, y_far, 0.0]
    far_res, far_score = await cache.lookup(v_far, scope_hash)
    assert far_res is None
    assert pytest.approx(far_score, abs=1e-5) == 0.90


@pytest.mark.asyncio
async def test_validation_guards_prevent_storage() -> None:
    redis = MockRedis()
    cache = SemanticCacheService(redis=redis, threshold=0.94)
    scope_hash = cache.canonical_scope_hash("hybrid")
    vec = [1.0, 0.0, 0.0]

    # Guard 1: Not answerable
    unanswerable = create_sample_rag_response(is_answerable=False, confidence_score=0.95)
    assert await cache.store(vec, scope_hash, unanswerable) is False

    # Guard 2: Low confidence (< 0.60)
    low_conf = create_sample_rag_response(is_answerable=True, confidence_score=0.55)
    assert await cache.store(vec, scope_hash, low_conf) is False

    # Guard 3: Contains unverified citation tag
    unverified = create_sample_rag_response(
        answer="Some unsupported statement [unverified: 1].",
        is_answerable=True,
        confidence_score=0.85,
    )
    assert await cache.store(vec, scope_hash, unverified) is False

    # Valid response stores successfully
    valid = create_sample_rag_response(is_answerable=True, confidence_score=0.85)
    assert await cache.store(vec, scope_hash, valid) is True


@pytest.mark.asyncio
async def test_document_reverse_index_tracking_and_cascade_invalidation() -> None:
    redis = MockRedis()
    cache = SemanticCacheService(redis=redis, threshold=0.94)
    scope_hash = cache.canonical_scope_hash("hybrid")
    vec = [1.0, 0.0, 0.0]

    doc_id = uuid.uuid4()
    response = create_sample_rag_response(doc_id=doc_id)

    # Store entry
    stored = await cache.store(vec, scope_hash, response)
    assert stored is True

    # Verify reverse index set was created
    doc_key = f"semcache:doc:{doc_id}"
    members = await redis.smembers(doc_key)
    assert len(members) == 1
    member = next(iter(members))
    assert member.startswith(f"{scope_hash}:")

    # Verify cache hit prior to invalidation
    hit_res, _ = await cache.lookup(vec, scope_hash)
    assert hit_res is not None

    # Invalidate document
    invalidated_count = await cache.invalidate_document(doc_id)
    assert invalidated_count == 1

    # Verify doc reverse index set is purged
    remaining_members = await redis.smembers(doc_key)
    assert len(remaining_members) == 0

    # Verify cache misses after document invalidation
    miss_res, _ = await cache.lookup(vec, scope_hash)
    assert miss_res is None


@pytest.mark.asyncio
async def test_redis_error_resilience() -> None:
    failing_redis = AsyncMock()
    failing_redis.hgetall.side_effect = ConnectionError("Redis server unreachable")
    failing_redis.hset.side_effect = ConnectionError("Redis server unreachable")
    failing_redis.smembers.side_effect = ConnectionError("Redis server unreachable")

    cache = SemanticCacheService(redis=failing_redis, threshold=0.94)
    scope_hash = cache.canonical_scope_hash("hybrid")
    vec = [1.0, 0.0, 0.0]
    response = create_sample_rag_response()

    # Lookup gracefully returns miss on connection error
    lookup_res, score = await cache.lookup(vec, scope_hash)
    assert lookup_res is None
    assert score == 0.0

    # Store gracefully returns False on connection error
    store_success = await cache.store(vec, scope_hash, response)
    assert store_success is False

    # Invalidation gracefully returns 0 on connection error
    invalidated = await cache.invalidate_document(uuid.uuid4())
    assert invalidated == 0


def test_canonical_scope_hash_determinism() -> None:
    cache = SemanticCacheService(redis=AsyncMock())

    # Filter dict order invariance
    hash1 = cache.canonical_scope_hash("hybrid", {"category": "tech", "page_min": 1})
    hash2 = cache.canonical_scope_hash("hybrid", {"page_min": 1, "category": "tech"})
    assert hash1 == hash2

    # Different search modes produce different hashes
    hash_dense = cache.canonical_scope_hash("dense", {"category": "tech", "page_min": 1})
    assert hash1 != hash_dense

    # None / empty filters are deterministic
    hash_none = cache.canonical_scope_hash("hybrid", None)
    hash_empty = cache.canonical_scope_hash("hybrid", {})
    assert hash_none == hash_empty
