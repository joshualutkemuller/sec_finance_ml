"""
Kafka Producer — Publishes borrow rate ticks to the Kafka topic.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

In production: connect to internal lending system WebSocket or REST polling.
In simulation: generates synthetic ticks with configurable anomaly injection.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


class BorrowRateTick:
    """Single borrow rate tick message."""
    __slots__ = ("ticker", "rate", "notional_usd", "counterparty_id", "timestamp_utc")

    def __init__(self, ticker: str, rate: float, notional_usd: float,
                 counterparty_id: str, timestamp_utc: str) -> None:
        self.ticker = ticker
        self.rate = rate
        self.notional_usd = notional_usd
        self.counterparty_id = counterparty_id
        self.timestamp_utc = timestamp_utc

    def to_json(self) -> bytes:
        return json.dumps({
            "ticker": self.ticker,
            "rate": self.rate,
            "notional_usd": self.notional_usd,
            "counterparty_id": self.counterparty_id,
            "timestamp_utc": self.timestamp_utc,
        }).encode("utf-8")


class SimulatedTickProducer:
    """
    Produces synthetic borrow rate ticks for testing the streaming pipeline.
    Injects random anomaly spikes at configurable probability.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["simulation"]
        self.kafka_cfg = config["kafka"]
        self.n_tickers = self.cfg.get("n_tickers", 50)
        self.tick_interval_ms = self.cfg.get("tick_interval_ms", 100)
        self.base_mean = self.cfg.get("base_rate_mean", 0.035)
        self.base_std = self.cfg.get("base_rate_std", 0.005)
        self.anomaly_prob = self.cfg.get("anomaly_probability", 0.005)
        self.anomaly_spike_bps = self.cfg.get("anomaly_spike_bps", 200)
        self._tickers = [f"TICK{i:04d}" for i in range(self.n_tickers)]
        self._base_rates = {t: self.base_mean + random.gauss(0, self.base_std)
                            for t in self._tickers}
        self._producer = None

    def start(self, max_ticks: int | None = None) -> None:
        """Start producing ticks to Kafka. Runs until max_ticks or KeyboardInterrupt."""
        self._producer = self._init_kafka()
        topic = self.kafka_cfg["topic_borrow_rates"]
        tick_count = 0
        logger.info("Starting tick producer: %d tickers, %.0fms interval",
                    self.n_tickers, self.tick_interval_ms)

        try:
            while max_ticks is None or tick_count < max_ticks:
                ticker = random.choice(self._tickers)
                rate = self._next_rate(ticker)
                tick = BorrowRateTick(
                    ticker=ticker,
                    rate=rate,
                    notional_usd=round(random.uniform(1e6, 50e6), 0),
                    counterparty_id=f"CP_{random.randint(1, 20):03d}",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                if self._producer:
                    self._producer.send(topic, value=tick.to_json(), key=ticker.encode())
                else:
                    logger.debug("Tick: %s rate=%.5f", ticker, rate)

                tick_count += 1
                time.sleep(self.tick_interval_ms / 1000)

        except KeyboardInterrupt:
            logger.info("Producer stopped after %d ticks", tick_count)
        finally:
            if self._producer:
                self._producer.flush()
                self._producer.close()

    def _next_rate(self, ticker: str) -> float:
        """Generate next rate with optional anomaly spike."""
        base = self._base_rates[ticker]
        if random.random() < self.anomaly_prob:
            # Inject spike anomaly
            spike = self.anomaly_spike_bps / 10_000 * random.choice([-1, 1])
            rate = base + spike
            logger.debug("Injecting anomaly spike on %s: +%.0fbps", ticker, self.anomaly_spike_bps)
        else:
            rate = base + random.gauss(0, self.base_std * 0.1)

        # Random walk
        self._base_rates[ticker] = base + random.gauss(0, self.base_std * 0.01)
        return max(rate, 0.0001)

    def _init_kafka(self):
        try:
            from kafka import KafkaProducer
            servers = os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS",
                self.kafka_cfg.get("bootstrap_servers", "localhost:9092"),
            )
            if servers.startswith("${"):
                logger.warning("KAFKA_BOOTSTRAP_SERVERS not set — running in dry-run mode")
                return None
            return KafkaProducer(
                bootstrap_servers=servers,
                linger_ms=self.kafka_cfg.get("producer_batch_ms", 5),
                acks=1,
            )
        except Exception as exc:
            logger.warning("Kafka unavailable (%s) — dry-run mode", exc)
            return None
