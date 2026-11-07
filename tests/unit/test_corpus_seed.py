"""Unit tests for corpus seed script and document discovery."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.db.models.document import Document
from scripts.ingest_corpus import (
    compute_sha256,
    discover_documents,
    ingest_corpus,
    load_manifest,
    main,
    parse_args,
)


def test_load_manifest_valid() -> None:
    """Test loading valid manifest file."""
    data = load_manifest("data/raw_corpus/manifest.json")
    assert "documents" in data
    assert isinstance(data["documents"], list)
    assert len(data["documents"]) >= 3


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    """Test loading non-existent manifest file raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent_manifest.json"
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        load_manifest(missing_path)


def test_load_manifest_invalid_format(tmp_path: Path) -> None:
    """Test loading manifest with missing 'documents' raises ValueError."""
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text('{"domain": "test"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid manifest format"):
        load_manifest(bad_manifest)


def test_compute_sha256(tmp_path: Path) -> None:
    """Test SHA-256 calculation on a known file."""
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"hello world")
    # sha256 of b"hello world" is b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert compute_sha256(test_file) == expected_hash


def test_discover_documents_success() -> None:
    """Test document discovery against existing raw corpus files."""
    manifest_data = load_manifest("data/raw_corpus/manifest.json")
    valid_docs, failed_docs = discover_documents(manifest_data, "data/raw_corpus")

    assert len(failed_docs) == 0, f"Failed docs: {failed_docs}"
    assert len(valid_docs) == len(manifest_data["documents"])
    for d in valid_docs:
        assert Path(d["file_path"]).exists()
        assert len(d["computed_sha256"]) == 64


def test_discover_documents_missing_file(tmp_path: Path) -> None:
    """Test document discovery when declared file does not exist."""
    manifest = {
        "documents": [
            {
                "filename": "ghost_document.pdf",
                "sha256": "87bbc56c26a2489ead845613dce13c0d02368a929415436fc55a8f1b924ef2c7",
            }
        ]
    }
    valid_docs, failed_docs = discover_documents(manifest, tmp_path)
    assert len(valid_docs) == 0
    assert len(failed_docs) == 1
    assert "File not found" in failed_docs[0]["error"]


def test_discover_documents_sha_mismatch(tmp_path: Path) -> None:
    """Test document discovery when SHA-256 hash does not match."""
    dummy_pdf = tmp_path / "doc.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")
    manifest = {
        "documents": [
            {
                "filename": "doc.pdf",
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            }
        ]
    }
    valid_docs, failed_docs = discover_documents(manifest, tmp_path)
    assert len(valid_docs) == 0
    assert len(failed_docs) == 1
    assert "SHA-256 mismatch" in failed_docs[0]["error"]


@pytest.mark.asyncio
async def test_ingest_corpus_dry_run() -> None:
    """Test ingest_corpus in dry-run mode skips DB operations."""
    summary = await ingest_corpus(
        manifest_path="data/raw_corpus/manifest.json",
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["total_discovered"] >= 3
    assert summary["verified"] == summary["total_discovered"]
    assert summary["ingested"] == 0
    assert summary["skipped"] == 0
    assert summary["failed"] == 0


@pytest.mark.asyncio
async def test_ingest_corpus_sync_dispatch(tmp_path: Path) -> None:
    """Test synchronous ingestion dispatch with mock repository and pipeline."""
    sample_pdf = tmp_path / "test_doc.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4 fake pdf binary")
    sample_sha = compute_sha256(sample_pdf)

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "documents": [
            {
                "filename": "test_doc.pdf",
                "sha256": sample_sha,
                "title": "Test Title",
                "total_pages": 1,
            }
        ]
    }
    import json

    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    fake_session = AsyncMock()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = uuid4()
    mock_doc.title = "Test Title"
    mock_doc.total_pages = 1
    mock_doc.status = "COMPLETED"

    with (
        patch("scripts.ingest_corpus.DocumentRepository") as mock_repo_cls,
        patch(
            "scripts.ingest_corpus.process_document_ingestion", new_callable=AsyncMock
        ) as mock_process,
    ):
        repo_instance = MagicMock()
        repo_instance.get_by_hash = AsyncMock(return_value=None)
        repo_instance.create = AsyncMock(return_value=mock_doc)
        mock_repo_cls.return_value = repo_instance
        mock_process.return_value = mock_doc

        summary = await ingest_corpus(
            manifest_path=manifest_file,
            dry_run=False,
            async_mode=False,
            corpus_dir=tmp_path,
            session=fake_session,
        )

        assert summary["ingested"] == 1
        assert summary["skipped"] == 0
        assert summary["failed"] == 0
        repo_instance.create.assert_awaited_once()
        mock_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_corpus_idempotency_skip(tmp_path: Path) -> None:
    """Test that already-completed documents are skipped without reprocessing."""
    sample_pdf = tmp_path / "test_doc.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4 fake pdf binary")
    sample_sha = compute_sha256(sample_pdf)

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "documents": [
            {
                "filename": "test_doc.pdf",
                "sha256": sample_sha,
                "title": "Test Title",
                "total_pages": 1,
            }
        ]
    }
    import json

    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    fake_session = AsyncMock()
    existing_doc = MagicMock(spec=Document)
    existing_doc.id = uuid4()
    existing_doc.status = "COMPLETED"

    with (
        patch("scripts.ingest_corpus.DocumentRepository") as mock_repo_cls,
        patch(
            "scripts.ingest_corpus.process_document_ingestion", new_callable=AsyncMock
        ) as mock_process,
    ):
        repo_instance = MagicMock()
        repo_instance.get_by_hash = AsyncMock(return_value=existing_doc)
        mock_repo_cls.return_value = repo_instance

        summary = await ingest_corpus(
            manifest_path=manifest_file,
            dry_run=False,
            async_mode=False,
            corpus_dir=tmp_path,
            session=fake_session,
        )

        assert summary["ingested"] == 0
        assert summary["skipped"] == 1
        assert summary["failed"] == 0
        mock_process.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_corpus_async_dispatch(tmp_path: Path) -> None:
    """Test async ingestion dispatches Celery task and stages file."""
    sample_pdf = tmp_path / "test_doc.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4 fake pdf binary")
    sample_sha = compute_sha256(sample_pdf)

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "documents": [
            {
                "filename": "test_doc.pdf",
                "sha256": sample_sha,
                "title": "Test Title",
                "total_pages": 1,
            }
        ]
    }
    import json

    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    fake_session = AsyncMock()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = uuid4()
    mock_doc.status = "PENDING"

    mock_task = MagicMock()
    mock_task.id = "celery-task-12345"

    with (
        patch("scripts.ingest_corpus.DocumentRepository") as mock_repo_cls,
        patch(
            "scripts.ingest_corpus.ingest_document_task.delay", return_value=mock_task
        ) as mock_delay,
    ):
        repo_instance = MagicMock()
        repo_instance.get_by_hash = AsyncMock(return_value=None)
        repo_instance.create = AsyncMock(return_value=mock_doc)
        repo_instance.update_status = AsyncMock(return_value=mock_doc)
        mock_repo_cls.return_value = repo_instance

        summary = await ingest_corpus(
            manifest_path=manifest_file,
            dry_run=False,
            async_mode=True,
            corpus_dir=tmp_path,
            session=fake_session,
        )

        assert summary["ingested"] == 1
        mock_delay.assert_called_once()
        repo_instance.update_status.assert_awaited_once_with(
            doc_id=mock_doc.id,
            status="PENDING",
            processing_stage="QUEUED",
            task_id="celery-task-12345",
        )


def test_cli_parse_args() -> None:
    """Test CLI argument parsing."""
    args = parse_args(
        ["--manifest", "custom.json", "--dry-run", "--async-mode", "--batch-size", "5"]
    )
    assert args.manifest == "custom.json"
    assert args.dry_run is True
    assert args.async_mode is True
    assert args.batch_size == 5


def test_cli_main_dry_run() -> None:
    """Test CLI main() with dry-run returns exit code 0."""
    with patch("sys.argv", ["ingest_corpus.py", "--dry-run"]):
        code = main()
        assert code == 0


def test_cli_main_missing_manifest() -> None:
    """Test CLI main() with non-existent manifest returns exit code 1."""
    with patch("sys.argv", ["ingest_corpus.py", "--manifest", "does_not_exist.json"]):
        code = main()
        assert code == 1
