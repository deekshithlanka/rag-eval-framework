"""Load help center docs and split them into chunks.

The chunker is paragraph-aware and deterministic so chunk size is the only
thing that changes between chunking experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    category: str
    last_updated: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str

    def as_context(self) -> str:
        return f"[{self.doc_id}] {self.title}\n{self.text}"


_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def load_corpus(directory: str | Path) -> list[Doc]:
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        raw = path.read_text()
        match = _FRONTMATTER.match(raw)
        if not match:
            raise ValueError(f"{path} is missing frontmatter")
        meta = dict(line.split(": ", 1) for line in match.group(1).splitlines())
        body = raw[match.end():].strip()
        body = re.sub(r"^# .*\n+", "", body)  # title is stored separately
        docs.append(Doc(meta["id"], meta["title"], meta["category"], meta["last_updated"], body))
    return docs


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, size: int) -> list[str]:
    """Split text into paragraphs, and any paragraph longer than size into sentences."""
    out = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if len(para) <= size:
            out.append(para)
            continue
        for sent in _SENTENCE.split(para):
            while len(sent) > size:  # very long sentence: hard split
                out.append(sent[:size])
                sent = sent[size:]
            if sent:
                out.append(sent)
    return out


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, size):
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        if tail:
            # start the overlap at a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail}\n\n{piece}" if len(tail) + len(piece) + 2 <= size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_docs(docs: list[Doc], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for doc in docs:
        for i, text in enumerate(chunk_text(doc.text, size, overlap)):
            chunks.append(Chunk(f"{doc.id}#{i}", doc.id, doc.title, text))
    return chunks
