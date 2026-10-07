"""Retrievers: FAISS over Gemini or open-source embeddings, plus a TF-IDF baseline.

LangChain and FAISS are imported lazily so the TF-IDF path (used by tests and
--mock runs) works without them.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

from .corpus import Chunk

CACHE = Path(".cache/index")
EMBED_BATCH = 80      # chunks per embedding batch (free tier: 100 requests per minute)
EMBED_PAUSE_S = 65    # seconds to wait between batches

# Embedding names accepted in configs/experiments.yaml
GEMINI_EMBEDDINGS = {"gemini-embedding-001"}
HF_EMBEDDINGS = {"bge-small-en-v1.5": "BAAI/bge-small-en-v1.5"}
LEXICAL = {"tfidf"}


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class TfidfRetriever:
    """Lexical baseline. Scores are cosine similarity (higher is better)."""

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.as_context() for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def make_embeddings(name: str):
    if name in GEMINI_EMBEDDINGS:
        from dotenv import load_dotenv
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        load_dotenv()
        return GoogleGenerativeAIEmbeddings(model=name)
    if name in HF_EMBEDDINGS:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDINGS[name], encode_kwargs={"normalize_embeddings": True}
        )
    raise ValueError(f"Unknown embedding model: {name}")


class FaissRetriever:
    """FAISS vector store built with LangChain. FAISS returns squared L2
    distance on normalized vectors (lower is better); we convert it to cosine
    similarity with 1 - d/2 so all retrievers report higher-is-better scores."""

    def __init__(self, chunks: list[Chunk], embedding: str, cache_key: str):
        from langchain_community.vectorstores import FAISS

        self.by_id = {c.chunk_id: c for c in chunks}
        embeddings = make_embeddings(embedding)
        fingerprint = hashlib.sha256(
            json.dumps([[c.chunk_id, c.text] for c in chunks]).encode()
        ).hexdigest()[:12]
        path = CACHE / f"{cache_key}__{fingerprint}"
        if path.exists():
            self.store = FAISS.load_local(
                str(path), embeddings, allow_dangerous_deserialization=True  # our own cache
            )
        else:
            # Embed in batches with a pause between them. The Gemini free tier
            # allows 100 embedding requests per minute, and smaller chunk sizes
            # produce more chunks than that.
            texts = [c.as_context() for c in chunks]
            metas = [{"chunk_id": c.chunk_id} for c in chunks]
            batch = EMBED_BATCH if embedding in GEMINI_EMBEDDINGS else len(texts)
            self.store = None
            for start in range(0, len(texts), batch):
                if start:
                    print(f"  embedded {start}/{len(texts)} chunks, pausing {EMBED_PAUSE_S}s for rate limit")
                    time.sleep(EMBED_PAUSE_S)
                part = slice(start, start + batch)
                if self.store is None:
                    self.store = FAISS.from_texts(texts[part], embeddings, metadatas=metas[part], normalize_L2=True)
                else:
                    self.store.add_texts(texts[part], metadatas=metas[part])
            path.parent.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(path))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        hits = self.store.similarity_search_with_score(query, k=k)
        return [(self.by_id[d.metadata["chunk_id"]], 1 - float(s) / 2) for d, s in hits]


def build_retriever(chunks: list[Chunk], embedding: str, cache_key: str) -> Retriever:
    if embedding in LEXICAL:
        return TfidfRetriever(chunks)
    return FaissRetriever(chunks, embedding, cache_key)
