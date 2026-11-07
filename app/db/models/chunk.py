"""SQLAlchemy model for document chunks with pgvector embeddings."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.document import Document
    from app.db.models.parent_chunk import ParentChunk

settings = get_settings()


class DocumentChunk(Base):
    """Represents a discrete semantic chunk of a document with vector embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parent_chunks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Text with prepended document and section hierarchy breadcrumbs",
    )
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', contextualized_text)", persisted=True),
        nullable=False,
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Hierarchical section path (e.g. Chapter 1 > Financial Results)",
    )
    element_type: Mapped[str] = mapped_column(
        String(30),
        default="prose",
        nullable=False,
        comment="prose | table | header | list | code",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Name and version of model used to generate embedding",
    )
    embedding = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION),
        nullable=False,
    )
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship back to parent document
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    parent_chunk: Mapped["ParentChunk | None"] = relationship(
        "ParentChunk",
        back_populates="child_chunks",
    )

    __table_args__ = (
        # HNSW Index for sub-millisecond approximate nearest neighbor cosine search
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index for JSONB metadata pre-filtering
        Index(
            "ix_document_chunks_metadata_gin",
            "metadata",
            postgresql_using="gin",
        ),
        # GIN index for full-text lexical search
        Index(
            "ix_document_chunks_search_vector_gin",
            "search_vector",
            postgresql_using="gin",
        ),
    )
