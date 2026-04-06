"""
Real-Time Streaming ML — Main Entry Point

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
"""

from __future__ import annotations

import argparse
import logging

import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-Time Streaming ML — Joshua Lutkemuller, MD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["produce", "consume", "replay"],
        default="replay",
        help="produce=emit ticks; consume=run streaming job; replay=offline test",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging(config.get("alerts", {}).get("log_level", "INFO"))
    logger = logging.getLogger("streaming.main")
    logger.info("mode=%s | Real-Time Streaming ML | Joshua Lutkemuller, MD", args.mode)

    if args.mode == "produce":
        from producer import SimulatedTickProducer
        SimulatedTickProducer(config).start()

    elif args.mode == "consume":
        from stream_job import StreamingJob
        StreamingJob(config).run()

    elif args.mode == "replay":
        # Replay mode: no Kafka needed, uses synthetic data
        from stream_job import StreamingJob
        job = StreamingJob(config)
        job._init_consumer = lambda: None
        job._consumer = None
        job._run_replay_mode()


if __name__ == "__main__":
    main()
