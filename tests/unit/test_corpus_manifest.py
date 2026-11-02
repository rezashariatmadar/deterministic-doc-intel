"""Unit tests for corpus curation manifest."""

import json
from pathlib import Path


def test_corpus_manifest_validity() -> None:
    manifest_path = Path("data/raw_corpus/manifest.json")
    assert manifest_path.exists(), "Manifest file must exist"

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "documents" in data
    assert len(data["documents"]) >= 1

    for doc in data["documents"]:
        doc_file = Path("data/raw_corpus") / doc["filename"]
        assert doc_file.exists(), f"Corpus file {doc['filename']} must exist"
        assert len(doc["sha256"]) == 64, "SHA-256 hash must be 64 hexadecimal characters"
        assert "title" in doc
        assert "total_pages" in doc
        assert "challenging_elements" in doc
        assert len(doc["challenging_elements"]) >= 1
