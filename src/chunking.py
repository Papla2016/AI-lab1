"""Dependency-free recursive character splitter."""
import hashlib
from .schemas import Document

SEPARATORS = ("\n\n", "\n", ". ", " ", "")

def _split(text: str, size: int, separators: tuple[str, ...] = SEPARATORS) -> list[str]:
    if len(text) <= size:
        return [text]
    separator, *rest = separators
    if not separator:
        return [text[i:i + size] for i in range(0, len(text), size)]
    pieces, current = text.split(separator), ""
    output: list[str] = []
    for piece in pieces:
        candidate = piece if not current else current + separator + piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                output.append(current)
            if len(piece) > size:
                output.extend(_split(piece, size, tuple(rest)))
                current = ""
            else:
                current = piece
    if current:
        output.append(current)
    return output

def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Document]:
    chunks: list[Document] = []
    for doc in documents:
        raw = [part.strip() for part in _split(doc.text, size) if part.strip()]
        for position, part in enumerate(raw):
            text = ((raw[position - 1][-overlap:] + "\n" + part) if overlap and position else part)
            identity = f"{doc.metadata.get('source')}:{doc.metadata.get('page', 0)}:{position}:{text}"
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
            chunks.append(Document(text=text, metadata=metadata))
    return chunks

