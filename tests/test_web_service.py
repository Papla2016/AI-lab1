"""Small UI service tests that do not require Chroma/Ollama."""
from pathlib import Path
import pytest
from src.config import Settings
from src.web_service import DocumentRouter, RAGLabService
from src.router import Route


def test_explicit_document_router():
    assert DocumentRouter().classify("Какая погода?") == Route.DOCUMENT_QUESTION


def test_upload_rejects_traversal_and_duplicate(tmp_path):
    settings = Settings(data_dir=tmp_path / "documents")
    service = RAGLabService(settings)
    for unsafe in ("../secret.txt", "nested/file.md", "evil.exe", "..", "folder\\x.txt"):
        with pytest.raises(ValueError):
            service.save_document(unsafe, b"text")
    assert service.save_document("manual.md", b"# Manual") == "manual.md"
    with pytest.raises(ValueError):
        service.save_document("manual.md", b"overwrite")
    assert (settings.data_dir / "manual.md").read_bytes() == b"# Manual"


def test_empty_corpus_preserves_index(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    service = RAGLabService(settings, store_factory=lambda _: None)
    with pytest.raises(ValueError, match="Прежний индекс"):
        service.index_documents()


def test_reject_invalid_retrieval_parameters():
    check = RAGLabService._validate
    with pytest.raises(ValueError):
        check(" ", "dense", "zero_shot", 5, 3)
    with pytest.raises(ValueError):
        check("Q?", "sparse", "zero_shot", 5, 3)
    with pytest.raises(ValueError):
        check("Q?", "dense", "unknown", 5, 3)
    with pytest.raises(ValueError):
        check("Q?", "dense", "few_shot", 2, 4)
    check("Q?", "hybrid", "cot_structured", 12, 5)
