"""Reciprocal Rank Fusion (RRF) for combining vector and lexical search rankings."""

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[tuple[Any, float]],
    sparse_results: list[tuple[Any, float]],
    weight_dense: float = 0.65,
    weight_sparse: float = 0.35,
    rrf_k: int = 60,
) -> list[tuple[Any, float, int | None, int | None]]:
    """
    Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Formula:
      RRF_score(d) = (weight_dense / (rrf_k + rank_dense)) + (weight_sparse / (rrf_k + rank_sparse))

    If an item is missing from either list, its contribution for that modality is strictly 0.0.
    Ranks are 1-based integers (1, 2, ...).

    Returns:
      List of tuples: (item, rrf_score, dense_rank_1_based, sparse_rank_1_based)
      sorted in descending order of rrf_score.
    """
    dense_rank_map: dict[Any, tuple[int, Any]] = {}
    for idx, (chunk, _score) in enumerate(dense_results, start=1):
        dense_rank_map[chunk.id] = (idx, chunk)

    sparse_rank_map: dict[Any, tuple[int, Any]] = {}
    for idx, (chunk, _score) in enumerate(sparse_results, start=1):
        sparse_rank_map[chunk.id] = (idx, chunk)

    all_chunk_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())
    if not all_chunk_ids:
        return []

    fused_results: list[tuple[Any, float, int | None, int | None]] = []

    for cid in all_chunk_ids:
        dense_entry = dense_rank_map.get(cid)
        sparse_entry = sparse_rank_map.get(cid)

        dense_rank = dense_entry[0] if dense_entry else None
        sparse_rank = sparse_entry[0] if sparse_entry else None
        chunk = dense_entry[1] if dense_entry else sparse_entry[1]  # type: ignore[index]

        dense_contrib = (weight_dense / (rrf_k + dense_rank)) if dense_rank is not None else 0.0
        sparse_contrib = (weight_sparse / (rrf_k + sparse_rank)) if sparse_rank is not None else 0.0
        rrf_score = dense_contrib + sparse_contrib

        fused_results.append((chunk, rrf_score, dense_rank, sparse_rank))

    fused_results.sort(key=lambda x: x[1], reverse=True)
    return fused_results
