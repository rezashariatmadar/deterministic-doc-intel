"""Hybrid retrieval service with RRF fusion, Cross-Encoder reranking, and parent resolution."""

from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.repositories.chunk_repo import ChunkRepository
from app.schemas.query import QueryFilters, RetrievedChunk
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.retrieval.query_transformer import QueryTransformer
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.retrieval.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from uuid import UUID


class VectorSearchService:
    """Orchestrates hybrid dense + sparse retrieval, fusion, reranking, and parent resolution."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.chunk_repo = ChunkRepository(session)
        self.settings = get_settings()
        self.reranker = (
            reranker
            if reranker is not None
            else (CrossEncoderReranker() if self.settings.ENABLE_RERANKER else None)
        )
        self.transformer = QueryTransformer(llm_provider) if llm_provider else None

    async def search(
        self,
        query: str,
        filters: QueryFilters | None = None,
        top_k: int = 5,
        max_tokens: int | None = None,
        search_mode: str = "hybrid",
        weight_dense: float | None = None,
        weight_sparse: float | None = None,
        enable_reranker: bool | None = None,
        enable_query_rewrite: bool = False,
        enable_hyde: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Execute dense, sparse, or hybrid retrieval with optional query expansion,
        RRF ranking fusion, neural reranking, and hierarchical parent chunk resolution.
        """
        if not query.strip():
            return []

        # 1. Query optimization (Rewrite / Expansion / HyDE)
        semantic_query = query
        lexical_query = query
        sub_queries: list[str] = []
        if enable_query_rewrite and self.transformer:
            transformed = await self.transformer.transform(query, enable_hyde=enable_hyde)
            semantic_query = transformed.semantic_query
            lexical_query = transformed.lexical_query
            sub_queries = [
                sq for sq in transformed.sub_queries if sq.strip() and sq.strip() != query.strip()
            ]
            if enable_hyde and transformed.hypothetical_passage:
                semantic_query = f"{semantic_query} {transformed.hypothetical_passage}"

        # 2. Extract metadata pre-filters
        min_similarity = (
            filters.min_similarity
            if filters and filters.min_similarity is not None
            else self.settings.SIMILARITY_THRESHOLD
        )
        doc_id = filters.document_id if filters else None
        category = filters.category if filters else None
        element_types = filters.element_types if filters else None
        page_min = filters.page_min if filters else None
        page_max = filters.page_max if filters else None

        # 3. Candidate retrieval (Sequential execution to preserve single AsyncSession stability)
        candidate_k = self.settings.HYBRID_SEARCH_CANDIDATES if search_mode == "hybrid" else top_k
        dense_results: list[tuple[Any, float]] = []
        sparse_results: list[tuple[Any, float]] = []

        if search_mode in ("hybrid", "dense"):
            query_vector = await self.embedding_provider.embed_query(semantic_query)
            dense_results = await self.chunk_repo.search_similar(
                query_vector=query_vector,
                top_k=candidate_k,
                threshold=min_similarity,
                document_id=doc_id,
                category=category,
                element_types=element_types,
                page_min=page_min,
                page_max=page_max,
            )
            # Retrieve for distinct sub-queries if query expansion was enabled
            if sub_queries:
                sub_vectors = await self.embedding_provider.embed_texts(sub_queries[:2])
                seen_dense = {c.id: (c, s) for c, s in dense_results}
                for s_vec in sub_vectors:
                    s_res = await self.chunk_repo.search_similar(
                        query_vector=s_vec,
                        top_k=candidate_k,
                        threshold=min_similarity,
                        document_id=doc_id,
                        category=category,
                        element_types=element_types,
                        page_min=page_min,
                        page_max=page_max,
                    )
                    for sc, ss in s_res:
                        if sc.id not in seen_dense or ss > seen_dense[sc.id][1]:
                            seen_dense[sc.id] = (sc, ss)
                dense_results = sorted(seen_dense.values(), key=lambda x: x[1], reverse=True)[
                    :candidate_k
                ]

        if search_mode in ("hybrid", "sparse"):
            sparse_results = await self.chunk_repo.search_lexical(
                query=lexical_query,
                top_k=candidate_k,
                document_id=doc_id,
                category=category,
                element_types=element_types,
                page_min=page_min,
                page_max=page_max,
            )
            if sub_queries:
                seen_sparse = {c.id: (c, s) for c, s in sparse_results}
                for sq in sub_queries[:2]:
                    sq_res = await self.chunk_repo.search_lexical(
                        query=sq,
                        top_k=candidate_k,
                        document_id=doc_id,
                        category=category,
                        element_types=element_types,
                        page_min=page_min,
                        page_max=page_max,
                    )
                    for sc, ss in sq_res:
                        if sc.id not in seen_sparse or ss > seen_sparse[sc.id][1]:
                            seen_sparse[sc.id] = (sc, ss)
                sparse_results = sorted(seen_sparse.values(), key=lambda x: x[1], reverse=True)[
                    :candidate_k
                ]

        # 4. Score fusion & RetrievedChunk construction
        w_dense = weight_dense if weight_dense is not None else self.settings.WEIGHT_DENSE
        w_sparse = weight_sparse if weight_sparse is not None else self.settings.WEIGHT_SPARSE
        rrf_k = self.settings.RRF_K

        dense_score_map = {c.id: score for c, score in dense_results}
        sparse_score_map = {c.id: score for c, score in sparse_results}

        fused: list[tuple[Any, float, int | None, int | None]]
        if search_mode == "dense":
            fused = [(c, score, idx, None) for idx, (c, score) in enumerate(dense_results, start=1)]
        elif search_mode == "sparse":
            fused = [
                (c, score, None, idx) for idx, (c, score) in enumerate(sparse_results, start=1)
            ]
        else:
            fused = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                weight_dense=w_dense,
                weight_sparse=w_sparse,
                rrf_k=rrf_k,
            )

        candidates: list[RetrievedChunk] = []
        for c, score, d_rank, s_rank in fused:
            doc_title = c.document.title if c.document and c.document.title else "Unknown Document"
            raw_meta = getattr(c, "chunk_metadata", None)
            if not isinstance(raw_meta, dict):
                raw_meta = c.metadata if isinstance(c.metadata, dict) else {}
            chunk_meta = raw_meta if isinstance(raw_meta, dict) else {}
            chunk_bbox = (
                chunk_meta.get("bbox") if isinstance(chunk_meta.get("bbox"), dict) else None
            )

            # Keep similarity_score normalized in [0.0, 1.0] for downstream confidence calculation
            if c.id in dense_score_map:
                sim_score = round(dense_score_map[c.id], 4)
            else:
                raw_lex = sparse_score_map.get(c.id, 0.0)
                sim_score = round(min(0.9, max(0.4, 0.4 + 0.5 * min(1.0, raw_lex))), 4)

            candidates.append(
                RetrievedChunk(
                    source_id=len(candidates) + 1,
                    chunk_id=c.id,
                    parent_chunk_id=c.parent_chunk_id,
                    document_id=c.document_id,
                    document_title=doc_title,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    section_path=c.section_path,
                    element_type=c.element_type,
                    text=c.raw_text,
                    contextualized_text=c.contextualized_text,
                    similarity_score=sim_score,
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                    rrf_score=round(score, 6) if search_mode == "hybrid" else None,
                    token_count=getattr(c, "token_count", None),
                    bbox=chunk_bbox,
                    metadata=chunk_meta,
                )
            )

        # 5. Cross-Encoder neural reranking (if enabled and candidates exist)
        should_rerank = (
            enable_reranker if enable_reranker is not None else self.settings.ENABLE_RERANKER
        )
        if should_rerank and self.reranker and candidates:
            candidates = await self.reranker.rerank(
                query=query,
                chunks=candidates,
                top_n=top_k * 2,
                threshold=self.settings.RERANKER_THRESHOLD,
            )

        # 6. Hierarchical parent resolution & deduplication
        parent_ids: list[UUID] = [
            c.parent_chunk_id for c in candidates if c.parent_chunk_id is not None
        ]
        parent_map = await self.chunk_repo.get_parents_by_ids(parent_ids) if parent_ids else {}

        resolved_chunks: list[RetrievedChunk] = []
        seen_parent_ids: set[UUID] = set()

        for c in candidates:
            if c.parent_chunk_id and c.parent_chunk_id in parent_map:
                if c.parent_chunk_id in seen_parent_ids:
                    # Deduplicate child chunks sharing the same parent
                    continue
                seen_parent_ids.add(c.parent_chunk_id)
                parent = parent_map[c.parent_chunk_id]
                # Swap chunk text to parent's broader context for quote verification
                c.text = parent.raw_text
                c.contextualized_text = parent.contextualized_text
                c.page_start = parent.page_start
                c.page_end = parent.page_end
                c.section_path = parent.section_path
                c.token_count = parent.token_count
                resolved_chunks.append(c)
            else:
                resolved_chunks.append(c)

        # 7. Token budgeting & 1-based sequential source_id re-indexing
        results: list[RetrievedChunk] = []
        token_ceiling = max_tokens if max_tokens is not None else self.settings.MAX_CONTEXT_TOKENS
        accumulated_tokens = 0

        for chunk in resolved_chunks:
            chunk_tokens = (
                chunk.token_count if chunk.token_count is not None else len(chunk.text.split())
            )
            if accumulated_tokens + chunk_tokens > token_ceiling:
                if results:
                    break
                break

            accumulated_tokens += chunk_tokens
            chunk.source_id = len(results) + 1
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results
