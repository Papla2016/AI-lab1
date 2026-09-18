"""Dependency-free recursive character splitter."""
import hashlib
from .schemas import Document

SEPARATORS = ("\n\n", "\n", ". ", " ")

def _split(text: str, size: int, overlap: int) -> list[str]:
    """Split at the best nearby semantic boundary while enforcing a hard size limit."""
    output: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + size, len(text))
        end = hard_end
        if hard_end < len(text):
            # Prefer stronger separators, but never make an extremely short chunk.
            for separator in SEPARATORS:
                boundary = text.rfind(separator, start + size // 2, hard_end)
                if boundary >= 0:
                    end = boundary + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            output.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return output

def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Document]:
    chunks: list[Document] = []
    for doc in documents:
        raw = _split(doc.text, size, overlap)
        for position, part in enumerate(raw):
            identity = f"{doc.metadata.get('source')}:{doc.metadata.get('page', 0)}:{position}:{part}"
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
            chunks.append(Document(text=part, metadata=metadata))
    return chunks
