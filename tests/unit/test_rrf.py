"""Unit tests for Reciprocal Rank Fusion (RRF) algorithm and edge cases."""

import uuid
from unittest.mock import MagicMock

import pytest

from app.services.retrieval.rrf import reciprocal_rank_fusion


def make_mock_chunk(chunk_id: uuid.UUID | None = None) -> MagicMock:
    m = MagicMock()
    m.id = chunk_id or uuid.uuid4()
    return m


def test_rrf_combines_dense_and_sparse_with_correct_math() -> None:
    c1, c2, c3, c4 = [make_mock_chunk() for _ in range(4)]

    # c1 is rank 1 dense, rank 2 sparse
    # c2 is rank 2 dense, not in sparse
    # c3 is not in dense, rank 1 sparse
    # c4 is rank 3 dense, rank 3 sparse
    dense_results = [(c1, 0.95), (c2, 0.85), (c4, 0.70)]
    sparse_results = [(c3, 15.2), (c1, 10.5), (c4, 5.0)]

    rrf_k = 60
    weight_dense = 0.65
    weight_sparse = 0.35

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        weight_dense=weight_dense,
        weight_sparse=weight_sparse,
        rrf_k=rrf_k,
    )

    assert len(fused) == 4
    # Top chunk should be c1 because it appears near top of both lists
    top_chunk, top_score, d_rank, s_rank = fused[0]
    assert top_chunk.id == c1.id
    assert d_rank == 1
    assert s_rank == 2

    # Expected score for c1: 0.65 / (60 + 1) + 0.35 / (60 + 2)
    expected_c1 = (0.65 / 61.0) + (0.35 / 62.0)
    assert pytest.approx(top_score, rel=1e-5) == expected_c1

    # Verify c2 (only in dense) has sparse_rank None and sparse_contrib 0.0
    c2_entry = next(entry for entry in fused if entry[0].id == c2.id)
    assert c2_entry[2] == 2
    assert c2_entry[3] is None
    expected_c2 = 0.65 / (60 + 2)
    assert pytest.approx(c2_entry[1], rel=1e-5) == expected_c2

    # Verify c3 (only in sparse) has dense_rank None and dense_contrib 0.0
    c3_entry = next(entry for entry in fused if entry[0].id == c3.id)
    assert c3_entry[2] is None
    assert c3_entry[3] == 1
    expected_c3 = 0.35 / (60 + 1)
    assert pytest.approx(c3_entry[1], rel=1e-5) == expected_c3

    # Output must be strictly descending in rrf_score
    scores = [entry[1] for entry in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_pure_dense_mode() -> None:
    c1, c2 = make_mock_chunk(), make_mock_chunk()
    dense_results = [(c1, 0.9), (c2, 0.8)]
    sparse_results = []

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        weight_dense=1.0,
        weight_sparse=0.0,
        rrf_k=60,
    )

    assert len(fused) == 2
    assert fused[0][0].id == c1.id
    assert fused[1][0].id == c2.id


def test_rrf_pure_sparse_mode() -> None:
    c1, c2 = make_mock_chunk(), make_mock_chunk()
    dense_results = []
    sparse_results = [(c2, 12.0), (c1, 8.0)]

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        weight_dense=0.0,
        weight_sparse=1.0,
        rrf_k=60,
    )

    assert len(fused) == 2
    assert fused[0][0].id == c2.id
    assert fused[1][0].id == c1.id


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], [], 0.65, 0.35, 60) == []
