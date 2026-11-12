from __future__ import annotations

from loguru import logger

from app.config import settings
from app.models import ChatResponse, RetrievedChunk, ResponseMetadata, RetrievedChunkPreview

from app.security.spotlighting import build_spotlighted_context
from app.security.system_prompt import build_system_prompt

from app.services.embedding_service import embed_texts
from app.services.llm_services import generate 
from app.services.vector_store import search


def _retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    embeddings = embed_texts([question])
    return search(embeddings[0], top_k=top_k)

def _generate(question: str, chunks: list[RetrievedChunk]) -> ChatResponse:
    spotlighted = build_spotlighted_context(chunks)
    system_prompt = build_system_prompt()
    user_message = f"{spotlighted}\n\nQuestion: {question}"
    raw = generate(system_prompt, user_message)
    chunk_previews = [
        RetrievedChunkPreview(text=chunk.text, source=chunk.source, score=chunk.score) for chunk in chunks
    ]

    return ChatResponse(
        answer=raw["text"],
        sources=list({c.source for c in chunks}),
        confidence=0.7,
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=chunk_previews,
        ),
        
    )

def _top_K_from_flags(flags: dict | None = None) -> int:
    if flags is None:
        return 5
    if isinstance(flags, int):
        return flags
    return int(flags.get("top_k", 5))

def run_rag_with_trace(question: str, flags: dict | None = None) -> tuple[ChatResponse, list[RetrievedChunk]]:
    top_k = _top_K_from_flags(flags)
    logger.info(f"Retrieving {top_k} chunks for question: {question}")

    chunks = _retrieve(question, top_k=top_k)
    response = _generate(question, chunks)
    return response, chunks



run_rag_with_trace_no_cache = run_rag_with_trace