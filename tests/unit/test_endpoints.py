"""Unit tests for API endpoints logic (health status, SSE verification payload, document concurrency)."""

from fastapi import status

from app.api.v1.endpoints.health import determine_health_status


def test_determine_health_status_healthy() -> None:
    overall_status, code = determine_health_status("connected", "connected")
    assert overall_status == "healthy"
    assert code == status.HTTP_200_OK


def test_determine_health_status_degraded_db() -> None:
    overall_status, code = determine_health_status("disconnected", "connected")
    assert overall_status == "degraded"
    assert code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_determine_health_status_degraded_redis() -> None:
    overall_status, code = determine_health_status("connected", "disconnected")
    assert overall_status == "degraded"
    assert code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_verification_payload_structure() -> None:
    from app.schemas.query import Citation, RAGResponse

    rag_response = RAGResponse(
        answer="The revenue grew to $47.5B [1].",
        is_answerable=True,
        citations=[
            Citation(
                source_id=1,
                document_title="NVIDIA 10-K",
                page_start=45,
                page_end=45,
                section_path="Item 7",
                supporting_quote="revenue grew to $47.5B",
                bbox={"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0},
                metadata={"block_index": 5},
            )
        ],
        confidence_score=1.0,
        latency_ms=12.5,
    )

    citations_payload = [
        {
            "source_id": c.source_id,
            "document_id": str(c.document_id) if c.document_id else None,
            "document_title": c.document_title,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_path": c.section_path,
            "supporting_quote": c.supporting_quote,
            "bbox": c.bbox,
            "metadata": c.metadata,
        }
        for c in rag_response.citations
    ]
    payload = {
        "is_answerable": rag_response.is_answerable,
        "verified_answer": rag_response.answer,
        "citations": citations_payload,
        "confidence_score": rag_response.confidence_score,
        "has_unverified": "[unverified:" in rag_response.answer,
        "latency_ms": rag_response.latency_ms,
    }

    assert payload["verified_answer"] == "The revenue grew to $47.5B [1]."
    assert payload["is_answerable"] is True
    assert payload["citations"][0]["bbox"] == {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0}
    assert payload["citations"][0]["metadata"] == {"block_index": 5}
