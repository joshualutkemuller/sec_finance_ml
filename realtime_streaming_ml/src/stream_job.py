"""
Flink Streaming Job — Consumes borrow rate ticks, applies RRCF anomaly detection,
and routes anomaly events to the alert sink.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Runs as a standalone Python process (uses kafka-python consumer loop) or
can be submitted to a Flink cluster using PyFlink if available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


class StreamingJob:
    """
    Consumes borrow_rates Kafka topic, runs online RRCF per ticker,
    and publishes anomaly events to alert_sink with < 500ms end-to-end latency.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.kafka_cfg = config["kafka"]
        self.streaming_cfg = config["streaming"]
        self._consumer = None
        self._detector = None
        self._alert_sink = None
        self._velocity_tracker: dict[str, list] = {}

    def run(self) -> None:
        """Start the streaming job. Blocks until interrupted."""
        from online_rrcf import OnlineRRCFDetector
        from alert_sink import AlertSink

        self._detector = OnlineRRCFDetector(self.config)
        self._alert_sink = AlertSink(self.config)
        self._consumer = self._init_consumer()

        if self._consumer is None:
            logger.warning("Kafka consumer unavailable — running in replay/test mode")
            self._run_replay_mode()
            return

        logger.info("Streaming job started — listening on topic: %s",
                    self.kafka_cfg["topic_borrow_rates"])

        try:
            for message in self._consumer:
                t_recv = time.time()
                self._process_message(message.value, t_recv)
        except KeyboardInterrupt:
            logger.info("Streaming job stopped")
        finally:
            self._consumer.close()

    def _process_message(self, raw: bytes, t_recv: float) -> None:
        """Process a single tick message end-to-end."""
        try:
            tick = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed message: %s", exc)
            return

        ticker = tick.get("ticker", "")
        rate = float(tick.get("rate", 0))
        ts = tick.get("timestamp_utc", datetime.now(timezone.utc).isoformat())

        # RRCF update
        result = self._detector.update(ticker, rate)

        # Velocity check
        velocity_alert = self._check_velocity(ticker, rate, ts)

        # Route to alert sink
        if result["is_anomaly"]:
            self._alert_sink.send({
                "alert_type": "RATE_SPIKE",
                "severity": "CRITICAL",
                "ticker": ticker,
                "rate": rate,
                "codisp_score": result["codisp_score"],
                "sigma_multiple": result["sigma_multiple"],
                "timestamp_utc": ts,
                "latency_ms": round((time.time() - t_recv) * 1000, 1),
            })

        if velocity_alert:
            self._alert_sink.send(velocity_alert)

    def _check_velocity(self, ticker: str, rate: float, ts: str) -> dict | None:
        """Detect abnormal rate-of-change velocity."""
        threshold = self.streaming_cfg.get("velocity_threshold_bps_per_sec", 5.0)
        window_s = self.streaming_cfg.get("velocity_window_seconds", 10)

        if ticker not in self._velocity_tracker:
            self._velocity_tracker[ticker] = []

        now = time.time()
        self._velocity_tracker[ticker].append((now, rate))
        # Keep only window
        self._velocity_tracker[ticker] = [
            (t, r) for t, r in self._velocity_tracker[ticker] if now - t <= window_s
        ]

        history = self._velocity_tracker[ticker]
        if len(history) < 2:
            return None

        oldest_t, oldest_r = history[0]
        elapsed = now - oldest_t
        if elapsed < 0.1:
            return None

        velocity_bps_per_sec = abs(rate - oldest_r) / elapsed * 10_000
        if velocity_bps_per_sec > threshold:
            return {
                "alert_type": "RATE_VELOCITY",
                "severity": "WARNING",
                "ticker": ticker,
                "rate": rate,
                "velocity_bps_per_sec": round(velocity_bps_per_sec, 2),
                "timestamp_utc": ts,
            }
        return None

    def _run_replay_mode(self) -> None:
        """Consume from a local CSV file for testing without Kafka."""
        from online_rrcf import OnlineRRCFDetector
        logger.info("Replay mode: generating synthetic ticks")
        det = OnlineRRCFDetector(self.config)
        import random, math
        for i in range(5000):
            ticker = f"TICK{random.randint(0,9):04d}"
            rate = 0.035 + 0.002 * math.sin(i / 100) + random.gauss(0, 0.0005)
            if random.random() < 0.003:
                rate += 0.02  # inject anomaly
            result = det.update(ticker, rate)
            if result["is_anomaly"]:
                logger.warning("Replay anomaly: %s", result)
            time.sleep(0.001)
        logger.info("Replay complete. State: %s", det.state_summary())

    def _init_consumer(self):
        try:
            from kafka import KafkaConsumer
            servers = os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS",
                self.kafka_cfg.get("bootstrap_servers", "localhost:9092"),
            )
            if servers.startswith("${"):
                return None
            return KafkaConsumer(
                self.kafka_cfg["topic_borrow_rates"],
                bootstrap_servers=servers,
                group_id=self.kafka_cfg.get("consumer_group", "streaming_ml"),
                auto_offset_reset=self.kafka_cfg.get("auto_offset_reset", "latest"),
                value_deserializer=lambda v: v,
            )
        except Exception as exc:
            logger.warning("Kafka consumer init failed: %s", exc)
            return None
