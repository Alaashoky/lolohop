"""Rate Limiting Configuration for HOPEFX Trading Bot.

Provides a simple in-memory token-bucket rate limiter that can be applied
to API routes or broker API calls to avoid exceeding request quotas.
"""

import time
import threading
from typing import Dict, Optional


class RateLimiter:
    """Token-bucket rate limiter.

    Args:
        rate: Maximum number of requests allowed per *period*.
        period: Length of the measurement window in seconds (default: 1).

    Example::

        limiter = RateLimiter(rate=10, period=1)  # 10 req/s
        if limiter.allow("broker_api"):
            make_api_call()
    """

    def __init__(self, rate: int = 60, period: float = 60.0) -> None:
        self.rate = rate
        self.period = period
        self._buckets: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def allow(self, key: str = "default") -> bool:
        """Check whether a request for *key* is allowed.

        Consumes one token from the bucket for *key*.  Returns ``True``
        when a token is available, ``False`` when the bucket is empty.

        Args:
            key: Identifier for the rate-limited resource (e.g. client IP,
                 API endpoint name).

        Returns:
            ``True`` if the request is permitted, ``False`` otherwise.
        """
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = {"tokens": self.rate, "last_refill": now}
                self._buckets[key] = bucket

            # Refill tokens proportional to elapsed time
            elapsed = now - bucket["last_refill"]
            refill = elapsed * (self.rate / self.period)
            bucket["tokens"] = min(self.rate, bucket["tokens"] + refill)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False

    def reset(self, key: Optional[str] = None) -> None:
        """Reset token bucket(s).

        Args:
            key: Specific bucket to reset.  When ``None`` all buckets are
                 cleared.
        """
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)


class RateLimitingConfiguration:
    """Centralised registry of rate limiters for the trading bot.

    Pre-configured limiters for common resource types::

        rl = RateLimitingConfiguration()
        if rl.broker.allow():
            place_order()
    """

    def __init__(self) -> None:
        # Broker REST API – typically 60–120 requests per minute
        self.broker: RateLimiter = RateLimiter(rate=60, period=60.0)
        # Internal API endpoints
        self.api: RateLimiter = RateLimiter(rate=300, period=60.0)
        # Market-data subscriptions
        self.market_data: RateLimiter = RateLimiter(rate=100, period=1.0)
        # Telegram / notification channels
        self.notifications: RateLimiter = RateLimiter(rate=20, period=60.0)

