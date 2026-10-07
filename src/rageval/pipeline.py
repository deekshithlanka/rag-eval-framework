"""The RAG pipeline under test: retrieve top-k chunks, then generate an answer."""

from __future__ import annotations

import re
import time

from .config import RunConfig
from .corpus import Chunk
from .llm import LLMResult
from .prompts import ABSTAIN_PHRASE, render
from .retrieval import Retriever


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n---\n\n".join(chunk.as_context() for chunk, _ in hits)


class RAGPipeline:
    def __init__(self, cfg: RunConfig, retriever: Retriever, generator):
        self.cfg = cfg
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> dict:
        start = time.perf_counter()
        hits = self.retriever.search(question, self.cfg.top_k)
        retrieval_s = time.perf_counter() - start
        context = format_context(hits)
        system, user = render(self.cfg.prompt, question, context)
        result: LLMResult = self.generator.complete(system, user)
        retrieved_docs: list[str] = []
        for chunk, _ in hits:
            if chunk.doc_id not in retrieved_docs:
                retrieved_docs.append(chunk.doc_id)
        return {
            "answer": result.text,
            "context": context,
            "retrieved_chunks": [[c.chunk_id, round(s, 4)] for c, s in hits],
            "retrieved_docs": retrieved_docs,
            "input_tokens": 0 if result.cached else result.input_tokens,
            "output_tokens": 0 if result.cached else result.output_tokens,
            # cached calls report the original generation latency
            "latency_s": round(retrieval_s + result.latency_s, 3),
        }


class ExtractiveMockGenerator:
    """Offline stand-in for the LLM. Returns the context sentences that best
    overlap the question, or abstains when overlap is low. Only for smoke tests."""

    _word = re.compile(r"[a-z0-9$%]+")

    def complete(self, system: str, user: str) -> LLMResult:
        context, _, question = user.partition("Customer question:")
        q = {w for w in self._word.findall(question.lower()) if len(w) > 3}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if len(s.strip()) > 30]
        scored = sorted(
            ((len(q & set(self._word.findall(s.lower()))), s) for s in sentences),
            key=lambda x: -x[0],
        )
        if not scored or scored[0][0] < 2:
            return LLMResult(ABSTAIN_PHRASE, latency_s=0.0)
        best = [s for score, s in scored[:2] if score >= 2]
        return LLMResult(" ".join(best), latency_s=0.0)
