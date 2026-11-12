from __future__ import annotations

import os

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


from app.config import settings


METRICS = [
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
]

_GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"


def _is_groq_model(model: str) -> bool:
    return model.startswith("openai/") or "gpt-oss" in model


def _get_ragas_llm():
    """Create a Ragas-compatible LLM using app settings."""
    model = settings.llm_model_grader
    # Match app.services.llm_services: grader ids like openai/gpt-oss-* are Groq.
    if _is_groq_model(model):
        chat = ChatOpenAI(
            model=model,
            api_key=settings.groq_api_key,
            base_url=_GROQ_OPENAI_BASE_URL,
        )
        return LangchainLLMWrapper(chat)

    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    return llm_factory(model)

def _get_ragas_embeddings():
    """Create Ragas-compatible embeddings using app settings."""
    lc_emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return LangchainEmbeddingsWrapper(lc_emb)

def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )

def run(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    ds = build_dataset(rows)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=False,
    )
    return result.to_pandas().to_dict(orient="records")


