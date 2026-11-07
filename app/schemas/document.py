"""Document and chunk Pydantic schemas (DTOs)."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field


class DocumentBase(BaseModel):
    filename: str = Field(..., max_length=255)
    title: str | None = Field(default=None, max_length=255)
    source_type: str = Field(default="pdf", max_length=50)
    category: str | None = Field(default=None, max_length=50)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    file_hash: str = Field(..., min_length=64, max_length=64)
    total_pages: int = Field(default=0, ge=0)
    task_id: str | None = None
    processing_stage: str | None = None


class DocumentRead(DocumentBase):
    id: UUID
    file_hash: str
    total_pages: int
    total_chunks: int
    status: str
    task_id: str | None = None
    processing_stage: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    processing_stage: str | None = None
    document_id: str | None = None
    error: str | None = None
    chunk_count: int = 0
    page_count: int = 0


class ChunkBase(BaseModel):
    chunk_index: int = Field(..., ge=0)
    parent_chunk_id: UUID | None = Field(default=None)
    raw_text: str = Field(..., min_length=1)
    contextualized_text: str = Field(..., min_length=1)
    token_count: int = Field(..., gt=0)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    section_path: str = Field(..., min_length=1)
    element_type: str = Field(default="prose", max_length=30)
    embedding_model: str = Field(..., max_length=100)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("chunk_metadata", "metadata"),
        serialization_alias="metadata",
    )


class ChunkCreate(ChunkBase):
    document_id: UUID
    embedding: list[float]


class ChunkRead(ChunkBase):
    id: UUID
    document_id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class ParentChunkBase(BaseModel):
    parent_index: int = Field(..., ge=0)
    raw_text: str = Field(..., min_length=1)
    contextualized_text: str = Field(..., min_length=1)
    token_count: int = Field(..., gt=0)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    section_path: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("chunk_metadata", "metadata"),
        serialization_alias="metadata",
    )


class ParentChunkCreate(ParentChunkBase):
    document_id: UUID


class ParentChunkRead(ParentChunkBase):
    id: UUID
    document_id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}
