"""Prometheus Monitoring for HOPEFX Trading Bot.

Exposes a Prometheus-compatible metrics endpoint via a lightweight HTTP
server.  Requires the ``prometheus_client`` package (included in
``requirements.txt``).
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed – metrics disabled")


class PrometheusMonitoring:
    """Collects and exposes trading-bot metrics for Prometheus scraping.

    Usage::

        monitoring = PrometheusMonitoring(port=8000)
        monitoring.start()
        monitoring.record_trade(pnl=120.50, side="BUY")
    """

    def __init__(self, port: int = 8000) -> None:
        self.port = port
        self._started = False

        if not _PROMETHEUS_AVAILABLE:
            return

        # --- Define metrics ---
        self.trades_total = Counter(
            "hopefx_trades_total",
            "Total number of completed trades",
            ["side"],
        )
        self.open_positions = Gauge(
            "hopefx_open_positions",
            "Current number of open positions",
        )
        self.pnl_total = Gauge(
            "hopefx_pnl_total_usd",
            "Cumulative realised P&L in USD",
        )
        self.trade_pnl = Histogram(
            "hopefx_trade_pnl_usd",
            "Per-trade P&L distribution in USD",
            buckets=[-500, -200, -100, -50, 0, 50, 100, 200, 500, 1000],
        )
        self.order_latency = Histogram(
            "hopefx_order_latency_ms",
            "Order execution latency in milliseconds",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        )
        self.capital = Gauge(
            "hopefx_capital_usd",
            "Current account capital in USD",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Prometheus HTTP server on :attr:`port`.

        Safe to call multiple times – the server is only started once.
        """
        if not _PROMETHEUS_AVAILABLE or self._started:
            return
        start_http_server(self.port)
        self._started = True
        logger.info("Prometheus metrics server started on port %d", self.port)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_trade(self, pnl: float, side: str) -> None:
        """Record a completed trade.

        Args:
            pnl: Realised profit or loss for the trade in USD.
            side: ``"BUY"`` or ``"SELL"``.
        """
        if not _PROMETHEUS_AVAILABLE:
            return
        self.trades_total.labels(side=side).inc()
        self.pnl_total.inc(pnl)
        self.trade_pnl.observe(pnl)

    def set_open_positions(self, count: int) -> None:
        """Update the open-position gauge.

        Args:
            count: Current number of open positions.
        """
        if not _PROMETHEUS_AVAILABLE:
            return
        self.open_positions.set(count)

    def set_capital(self, amount: float) -> None:
        """Update the capital gauge.

        Args:
            amount: Current account capital in USD.
        """
        if not _PROMETHEUS_AVAILABLE:
            return
        self.capital.set(amount)

    def record_latency(self, latency_ms: float) -> None:
        """Record order execution latency.

        Args:
            latency_ms: Execution latency in milliseconds.
        """
        if not _PROMETHEUS_AVAILABLE:
            return
        self.order_latency.observe(latency_ms)

