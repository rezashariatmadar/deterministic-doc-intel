"""Automated Corpus Ingestion and Seeding CLI.

Discovers documents declared in the corpus manifest, validates cryptographic hashes,
and ingests documents into the RAG platform (synchronously or asynchronously via Celery).
"""

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import sys

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import setup_logging
from app.db.repositories.document_repo import DocumentRepository
from app.db.session import get_task_scoped_session
from app.schemas.document import DocumentCreate
from app.services.ingestion.pipeline import process_document_ingestion
from app.services.providers.base import EmbeddingProvider
from app.services.providers.factory import get_embedding_provider
from app.workers.tasks import ingest_document_task

logger = logging.getLogger("scripts.ingest_corpus")


def load_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Load and parse the corpus manifest JSON file."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if "documents" not in data or not isinstance(data["documents"], list):
        raise ValueError(f"Invalid manifest format in {path}: 'documents' list missing")

    return data


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def discover_documents(
    manifest_data: dict[str, Any],
    corpus_dir: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover documents from manifest, verify file existence and SHA-256 hashes.

    Returns:
        tuple of (valid_documents, failed_documents)
    """
    corpus_path = Path(corpus_dir)
    valid_docs: list[dict[str, Any]] = []
    failed_docs: list[dict[str, Any]] = []

    for item in manifest_data["documents"]:
        filename = item.get("filename", "")
        expected_sha = item.get("sha256", "")
        file_path = corpus_path / filename

        if not file_path.is_file():
            logger.warning("Corpus file does not exist: %s", file_path)
            failed_docs.append({**item, "error": f"File not found: {file_path}"})
            continue

        actual_sha = compute_sha256(file_path)
        if expected_sha and actual_sha.lower() != expected_sha.lower():
            logger.warning(
                "SHA-256 mismatch for %s: expected %s, got %s",
                filename,
                expected_sha,
                actual_sha,
            )
            failed_docs.append(
                {
                    **item,
                    "error": f"SHA-256 mismatch (expected {expected_sha}, got {actual_sha})",
                }
            )
            continue

        valid_docs.append(
            {
                **item,
                "file_path": file_path,
                "computed_sha256": actual_sha,
            }
        )

    return valid_docs, failed_docs


async def ingest_corpus(
    manifest_path: Path | str = "data/raw_corpus/manifest.json",
    dry_run: bool = False,
    async_mode: bool = False,
    batch_size: int = 10,
    corpus_dir: Path | str | None = None,
    session: AsyncSession | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Execute corpus discovery and ingestion.

    Args:
        manifest_path: Path to manifest JSON.
        dry_run: If True, only discovers and verifies files without DB writes.
        async_mode: If True, dispatches ingestion to Celery background workers.
        batch_size: Batch size for processing.
        corpus_dir: Optional override for directory containing corpus PDFs.
        session: Optional external AsyncSession (for tests/custom transactions).
        embedding_provider: Optional EmbeddingProvider instance.

    Returns:
        Summary dict containing ingestion statistics.
    """
    manifest_p = Path(manifest_path)
    manifest_data = load_manifest(manifest_p)

    target_corpus_dir = Path(corpus_dir) if corpus_dir else manifest_p.parent
    valid_docs, failed_docs = discover_documents(manifest_data, target_corpus_dir)

    total_discovered = len(manifest_data["documents"])
    logger.info(
        "Discovered %d documents in manifest: %d valid, %d failed",
        total_discovered,
        len(valid_docs),
        len(failed_docs),
    )

    if dry_run:
        logger.info("DRY-RUN mode enabled: skipping database writes and worker dispatch")
        return {
            "total_discovered": total_discovered,
            "verified": len(valid_docs),
            "ingested": 0,
            "skipped": 0,
            "failed": len(failed_docs),
            "dry_run": True,
            "async_mode": async_mode,
            "valid_files": [d["filename"] for d in valid_docs],
            "failed_files": [d.get("filename", "") for d in failed_docs],
        }

    emb_provider = embedding_provider or get_embedding_provider()

    ingested_count = 0
    skipped_count = 0
    error_count = len(failed_docs)

    async def _process_batch(
        docs: list[dict[str, Any]], sess: AsyncSession
    ) -> tuple[int, int, int]:
        ingested = 0
        skipped = 0
        errors = 0
        doc_repo = DocumentRepository(sess)

        for doc_info in docs:
            filename = doc_info["filename"]
            file_path: Path = doc_info["file_path"]
            file_hash = doc_info["computed_sha256"]
            title = doc_info.get("title", filename)
            category = doc_info.get("category", "financial_report")
            total_pages = doc_info.get("total_pages", 0)

            try:
                # Idempotency check: check if file_hash already exists
                existing = await doc_repo.get_by_hash(file_hash)
                if existing and existing.status == "COMPLETED":
                    logger.info("Skipping already indexed document: %s (ID: %s)", filename, existing.id)
                    skipped += 1
                    continue
                elif existing:
                    # Clean up incomplete previous run
                    await doc_repo.delete(existing.id)

                if async_mode:
                    doc = await doc_repo.create(
                        DocumentCreate(
                            filename=filename,
                            title=title,
                            source_type="pdf",
                            category=category,
                            file_hash=file_hash,
                            total_pages=total_pages,
                            processing_stage="QUEUED",
                        )
                    )
                    staging_dir = Path("data/uploads")
                    staging_dir.mkdir(parents=True, exist_ok=True)
                    staging_path = staging_dir / f"{doc.id}.pdf"
                    staging_path.write_bytes(file_path.read_bytes())

                    task = ingest_document_task.delay(str(doc.id), str(staging_path))
                    await doc_repo.update_status(
                        doc_id=doc.id,
                        status="PENDING",
                        processing_stage="QUEUED",
                        task_id=task.id,
                    )
                    logger.info("Dispatched async ingestion task %s for %s", task.id, filename)
                    ingested += 1
                else:
                    doc = await doc_repo.create(
                        DocumentCreate(
                            filename=filename,
                            title=title,
                            source_type="pdf",
                            category=category,
                            file_hash=file_hash,
                            total_pages=total_pages,
                            processing_stage="PARSING",
                        )
                    )
                    completed = await process_document_ingestion(
                        document_id=doc.id,
                        file_path=file_path,
                        session=sess,
                        task_id=None,
                        embedding_provider=emb_provider,
                    )
                    logger.info(
                        "Ingested document %s (%s) with %d pages",
                        completed.title,
                        completed.id,
                        completed.total_pages,
                    )
                    ingested += 1
            except Exception as exc:
                logger.error("Failed to ingest document %s: %s", filename, exc, exc_info=True)
                errors += 1

        return ingested, skipped, errors

    if session is not None:
        ingested_count, skipped_count, batch_errors = await _process_batch(valid_docs, session)
        error_count += batch_errors
    else:
        async with get_task_scoped_session() as sess:
            # Process in batches
            for i in range(0, len(valid_docs), batch_size):
                batch = valid_docs[i : i + batch_size]
                ing, skp, err = await _process_batch(batch, sess)
                ingested_count += ing
                skipped_count += skp
                error_count += err

    return {
        "total_discovered": total_discovered,
        "verified": len(valid_docs),
        "ingested": ingested_count,
        "skipped": skipped_count,
        "failed": error_count,
        "dry_run": False,
        "async_mode": async_mode,
        "valid_files": [d["filename"] for d in valid_docs],
        "failed_files": [d.get("filename", "") for d in failed_docs],
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest raw corpus documents declared in manifest into the RAG platform."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/raw_corpus/manifest.json",
        help="Path to corpus manifest JSON file (default: data/raw_corpus/manifest.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Verify manifest and source file existence/SHA-256 without database writes or worker dispatch.",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        default=False,
        help="Dispatch ingestion to Celery background task queue rather than processing synchronously.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process per batch (default: 10).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Optional override directory where PDF files are stored (defaults to manifest parent directory).",
    )
    return parser.parse_args(args)


def main() -> int:
    """Entry point for corpus ingestion CLI."""
    setup_logging()
    args = parse_args()

    logger.info("Starting corpus ingestion with manifest: %s", args.manifest)
    try:
        summary = asyncio.run(
            ingest_corpus(
                manifest_path=args.manifest,
                dry_run=args.dry_run,
                async_mode=args.async_mode,
                batch_size=args.batch_size,
                corpus_dir=args.corpus_dir,
            )
        )
    except FileNotFoundError as exc:
        logger.error("Manifest error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        return 1

    print("\n" + "=" * 50)
    print("Corpus Ingestion Summary")
    print("=" * 50)
    print(f"Total Discovered: {summary['total_discovered']}")
    print(f"Verified Valid:   {summary['verified']}")
    print(f"Ingested:         {summary['ingested']}")
    print(f"Skipped (Cached): {summary['skipped']}")
    print(f"Failed:           {summary['failed']}")
    print(f"Dry Run:          {summary['dry_run']}")
    print(f"Async Mode:       {summary['async_mode']}")
    print("=" * 50 + "\n")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
