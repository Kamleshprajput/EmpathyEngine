"""
cache.py — Redis prompt/response cache for EmpathyEngine.

Drop this file alongside app.py and import get_cached / set_cached
wherever you call process() in app.py.

Cache key = sha256(normalised_text)  → JSON-serialised result dict
TTL is read from REDIS_TTL env var (default 3600 s = 1 hour).
"""

import os
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

_redis_client = None


def _get_client():
    """Lazy-initialise the Redis client (returns None if Redis is unavailable)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", 6379))
        client = redis.Redis(
            host=host,
            port=port,
            db=0,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.ping()  # Fail fast if Redis is unreachable
        _redis_client = client
        logger.info(f"[cache] Connected to Redis at {host}:{port}")
    except Exception as exc:
        logger.warning(f"[cache] Redis unavailable — running without cache. ({exc})")
        _redis_client = None

    return _redis_client


def _cache_key(text: str) -> str:
    """Deterministic key: strip + lowercase + sha256."""
    normalised = text.strip().lower()
    return "empathy:v1:" + hashlib.sha256(normalised.encode()).hexdigest()


def get_cached(text: str) -> dict | None:
    """
    Return the cached result for *text*, or None on a cache miss / Redis error.
    """
    client = _get_client()
    if client is None:
        return None

    key = _cache_key(text)
    try:
        raw = client.get(key)
        if raw is None:
            logger.debug(f"[cache] MISS  {key[:20]}…")
            return None
        logger.info(f"[cache] HIT   {key[:20]}…")
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"[cache] get error: {exc}")
        return None


def set_cached(text: str, result: dict) -> None:
    """
    Store *result* (must be JSON-serialisable) keyed by *text*.
    Silently swallows errors so a Redis outage never breaks the API.
    """
    client = _get_client()
    if client is None:
        return

    key = _cache_key(text)
    ttl = int(os.environ.get("REDIS_TTL", 3600))
    try:
        client.setex(key, ttl, json.dumps(result))
        logger.debug(f"[cache] SET   {key[:20]}…  TTL={ttl}s")
    except Exception as exc:
        logger.warning(f"[cache] set error: {exc}")


def invalidate(text: str) -> bool:
    """Delete a specific cache entry. Returns True if the key existed."""
    client = _get_client()
    if client is None:
        return False
    key = _cache_key(text)
    try:
        return bool(client.delete(key))
    except Exception:
        return False


def cache_stats() -> dict:
    """Return basic Redis INFO for a /cache/stats endpoint."""
    client = _get_client()
    if client is None:
        return {"status": "unavailable"}
    try:
        info = client.info("stats")
        mem  = client.info("memory")
        return {
            "status": "ok",
            "hits":   info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "used_memory_human": mem.get("used_memory_human"),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}