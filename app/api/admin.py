import asyncio
from typing import Any 
from fastapi import APIRouter, Depends
from loguru import logger

from app.config import settings

from app.middleware import User, require_admin

router = APIRouter(tags=["admin"])

async def _ping_postgres() -> bool:
    try:
        import psycopg2
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.debug(f"Error pinging PostgreSQL: {e}")
        return False

async def _ping_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, timeout=3)
        client.get_collections()
        return True
    except Exception as e:
        logger.debug(f"Error pinging Qdrant: {e}")
        return False

async def _ping_redis() -> bool:
    try:
        from upstash_redis import Redis
        redis = Redis(url=settings.upstash_redis_rest_url, token=settings.upstash_redis_rest_token)
        redis.ping()
        return True
    except Exception as e:
        logger.debug(f"Error pinging Redis: {e}")
        return False

async def _ping_openai() -> bool:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        client.models.list()
        return True
    except Exception as e:
        logger.debug(f"Error pinging OpenAI: {e}")
        return False

async def _ping_gemini() -> bool:
    try:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        next(client.models.list(), None)
        return True
    except Exception as e:
        logger.debug(f"Error pinging Gemini: {e}")
        return False

async def _ping_tavily() -> bool:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        client.search("health check", max_results=1)
        return True
    except Exception as e:
        logger.debug(f"Error pinging Tavily: {e}")
        return False

@router.get("/admin/health")
async def health_check() -> dict[str, Any]:
    results = await asyncio.gather(
        _ping_postgres(),
        _ping_qdrant(),
        _ping_redis(),
        _ping_openai(),
        _ping_gemini(),
        _ping_tavily(),
        return_exceptions=True,
    )

    postgres_ok = bool(results[0]) if not isinstance(results[0], Exception) else False
    qdrant_ok = bool(results[1]) if not isinstance(results[1], Exception) else False
    redis_ok = bool(results[2]) if not isinstance(results[2], Exception) else False
    openai_ok = bool(results[3]) if not isinstance(results[3], Exception) else False
    gemini_ok = bool(results[4]) if not isinstance(results[4], Exception) else False
    tavily_ok = bool(results[5]) if not isinstance(results[5], Exception) else False

    return {
        "postgres_ok": postgres_ok,
        "qdrant_ok": qdrant_ok,
        "redis_ok": redis_ok,
        "openai_ok": openai_ok,
        "gemini_ok": gemini_ok,
        "tavily_ok": tavily_ok,
    }

