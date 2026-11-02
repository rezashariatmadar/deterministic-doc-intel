"""Repository pattern for Document database operations."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.schemas.document import DocumentCreate


class DocumentRepository:
    """Encapsulates all database operations for Document entities."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, doc_in: DocumentCreate) -> Document:
        """Create and persist a new Document record."""
        doc = Document(
            filename=doc_in.filename,
            file_hash=doc_in.file_hash,
            title=doc_in.title,
            source_type=doc_in.source_type,
            category=doc_in.category,
            total_pages=doc_in.total_pages,
            extra_metadata=doc_in.extra_metadata,
            status="PENDING",
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_by_id(self, doc_id: UUID) -> Document | None:
        """Fetch document by primary key UUID."""
        stmt = select(Document).where(Document.id == doc_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_hash(self, file_hash: str) -> Document | None:
        """Fetch document by unique SHA-256 file hash for idempotency."""
        stmt = select(Document).where(Document.file_hash == file_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 20) -> list[Document]:
        """Fetch paginated list of documents."""
        stmt = select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self) -> int:
        """Return total count of documents."""
        stmt = select(func.count()).select_from(Document)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def update_status(
        self,
        doc_id: UUID,
        status: str,
        total_chunks: int | None = None,
        error_message: str | None = None,
    ) -> Document | None:
        """Update processing status and chunk statistics."""
        doc = await self.get_by_id(doc_id)
        if not doc:
            return None

        doc.status = status
        if total_chunks is not None:
            doc.total_chunks = total_chunks
        if error_message is not None:
            doc.error_message = error_message

        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def delete(self, doc_id: UUID) -> bool:
        """Delete document (cascades automatically to all associated chunks)."""
        doc = await self.get_by_id(doc_id)
        if not doc:
            return False

        await self.session.delete(doc)
        await self.session.commit()
        return True
