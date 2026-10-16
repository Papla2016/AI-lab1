"""Load UTF-8 text, Markdown, and per-page PDF documents."""
import logging
from pathlib import Path
from pypdf import PdfReader
from .schemas import Document

LOG = logging.getLogger(__name__)

def load_documents(data_dir: Path, errors: list[str] | None = None) -> list[Document]:
    documents: list[Document] = []
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        # Uploaded files live inside data/. Never follow a symlink outside it.
        if not path.resolve().is_relative_to(data_dir.resolve()):
            if errors is not None:
                errors.append(f"{path.name}: файл находится за пределами каталога data/.")
            continue
        try:
            source = path.relative_to(data_dir).as_posix()
            previous_count = len(documents)
            if path.suffix.lower() == ".pdf":
                for number, page in enumerate(PdfReader(path).pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(Document(text, {"source": source, "page": number}))
            else:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    documents.append(Document(text, {"source": source}))
            if len(documents) == previous_count and errors is not None:
                errors.append(f"{source}: не найден читаемый текст; для сканов PDF требуется OCR.")
        except Exception as exc:  # continue indexing other user files
            LOG.error("Не удалось прочитать %s: %s", path, exc)
            if errors is not None:
                errors.append(f"{path.name}: не удалось прочитать файл ({type(exc).__name__}).")
    return documents
