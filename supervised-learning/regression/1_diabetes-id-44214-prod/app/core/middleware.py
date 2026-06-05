"""
app/core/middleware.py
Production middleware stack — two classes, fully self-contained.

  RateLimitMiddleware  — sliding-window per-IP rate limiter (no Redis needed)
  APIKeyMiddleware     — X-API-Key / Bearer token authentication

Both are opt-in via settings flags so local dev can run without them.
Import in app/main.py:
    from app.core.middleware import RateLimitMiddleware, APIKeyMiddleware
"""

import hashlib
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths that always bypass both rate limiting AND auth
# ---------------------------------------------------------------------------
_PUBLIC_PATHS = {
    "/",
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/monitoring",
    "/drift",
    "/model-info",
}


# ===========================================================================
# 1.  Sliding-window rate limiter
# ===========================================================================

class _SlidingWindow:
    """
    Thread-safe per-key sliding window counter.

    Stores a deque of monotonic timestamps for every client key.
    On each call we evict expired entries, then count what's left.

    Time complexity  : O(expired) amortised per call
    Memory           : O(limit * n_unique_clients)
    """

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit          = limit
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Returns (allowed, remaining_quota).
        Mutates the bucket — calling this IS the "consume one token" action.
        """
        now    = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._buckets[key]
            # Remove timestamps older than the window
            while dq and dq[0] < cutoff:
                dq.popleft()
            count = len(dq)
            if count >= self.limit:
                return False, 0
            dq.append(now)
            return True, self.limit - count - 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter keyed on the client IP address.

    Behaviour
    ---------
    - Allowed requests receive X-RateLimit-Limit and X-RateLimit-Remaining headers.
    - Blocked requests receive HTTP 429 with a Retry-After: 60 header.
    - Public paths (health, docs, metrics …) are never counted or blocked.
    - The raw IP is hashed (SHA-256, first 16 hex chars) before use as a key
      so it is never stored or logged in plain text.

    Configuration (set in Settings / .env)
    ----------------------------------------
    RATE_LIMIT_PER_MINUTE=60   # requests per rolling 60-second window
    RATE_LIMIT_ENABLED=true    # set to false in dev to skip entirely
    """

    def __init__(self, app, limit_per_minute: int = 60):
        super().__init__(app)
        self._limiter = _SlidingWindow(limit=limit_per_minute, window_seconds=60)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Public / ops paths are never rate-limited
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Resolve client IP (works behind reverse proxies)
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP", "")
            or (request.client.host if request.client else "unknown")
        )

        # Hash IP so it is never stored in plain text
        key = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

        allowed, remaining = self._limiter.is_allowed(key)

        if not allowed:
            logger.warning("Rate limit exceeded", extra={"client_key": key})
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
                content={
                    "detail": "Too many requests — please slow down.",
                    "retry_after_seconds": 60,
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(self._limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ===========================================================================
# 2.  API Key authentication middleware
# ===========================================================================

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates an API key on all non-public endpoints.

    Accepted key locations (checked in order)
    ------------------------------------------
    1. Header  X-API-Key: <raw_key>
    2. Header  Authorization: Bearer <raw_key>

    Key storage
    -----------
    Raw keys are NEVER stored.  Only SHA-256 hashes are kept, supplied
    via Settings.valid_key_hashes (derived from API_KEY_HASHES env var,
    comma-separated list of sha256 hex strings).

    How to generate a hash for a new key
    -------------------------------------
        python3 -c "import hashlib; print(hashlib.sha256(b'your-secret-key').hexdigest())"

    Configuration (set in Settings / .env)
    ----------------------------------------
    API_KEYS_ENABLED=true
    API_KEY_HASHES=<hash1>,<hash2>,...

    For local development leave API_KEYS_ENABLED=false (default) — all
    requests are allowed through without a key.
    """

    def __init__(self, app, valid_key_hashes: set[str], enabled: bool = True):
        super().__init__(app)
        self._hashes  = valid_key_hashes
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_key(request: Request) -> Optional[str]:
        """Pull the raw API key out of request headers."""
        # 1. Dedicated header
        key = request.headers.get("X-API-Key")
        if key:
            return key
        # 2. Bearer token
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    # ------------------------------------------------------------------
    # Middleware entry point
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Middleware disabled — pass everything through (local dev mode)
        if not self._enabled:
            return await call_next(request)

        # Public paths never require auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        raw_key = self._extract_key(request)

        # No key provided at all
        if not raw_key:
            logger.warning("Missing API key", extra={"path": request.url.path})
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "Missing API key. "
                        "Provide an X-API-Key header or Authorization: Bearer <key>."
                    )
                },
            )

        # Key provided but does not match any known hash
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if key_hash not in self._hashes:
            logger.warning("Invalid API key", extra={"path": request.url.path})
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        # Valid key — let the request through
        return await call_next(request)
