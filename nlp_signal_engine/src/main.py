"""
NLP Signal Engine — Main Entry Point

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLP Signal Engine — Joshua Lutkemuller, MD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["ingest", "process", "brief", "full"],
        default="full",
        help="ingest=fetch text sources; process=run NLP; brief=generate morning brief; full=all",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_id = str(uuid.uuid4())
    logger = logging.getLogger("nlp.main")
    logger.info("run_id=%s | mode=%s | NLP Signal Engine", run_id, args.mode)

    from nlp_processor import NLPProcessor, TextRecord
    from morning_brief import MorningBriefGenerator

    records: list[TextRecord] = []

    if args.mode in ("ingest", "full"):
        records.extend(_ingest_edgar(config))
        records.extend(_ingest_fomc(config))
        records.extend(_ingest_news(config))
        logger.info("Ingested %d text records", len(records))

    nlp_df = pd.DataFrame()
    if args.mode in ("process", "full") and records:
        processor = NLPProcessor(config)
        results = processor.process_batch(records)
        nlp_df = processor.to_dataframe(results)

        out_path = Path(config["signals"]["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nlp_df.to_parquet(out_path)
        logger.info("NLP signals saved to %s (%d rows)", out_path, len(nlp_df))

    if args.mode in ("brief", "full"):
        # Load NLP signals if not freshly computed
        if nlp_df.empty:
            sig_path = Path(config["signals"]["output_path"])
            if sig_path.exists():
                nlp_df = pd.read_parquet(sig_path)

        # Macro signals (placeholder — integrate with fred_macro_intelligence in production)
        macro_signals = {
            "current_regime": "Hiking Cycle",
            "sofr": "5.33%",
            "hy_oas": "350bps",
            "curve_10y2y": "-0.42% (inverted)",
            "vix": "18.2",
            "rrp_balance": "$412bn",
            "fomc_hawkishness_index": "0.71",
        }

        model_outputs = {
            "agency_demand_forecast": "+12% vs 5d avg",
            "collateral_quality_index": "0.42 (Neutral)",
            "top_squeeze_risk_ticker": "See risk names above",
        }

        gen = MorningBriefGenerator(config)
        brief = gen.generate(nlp_df, macro_signals, model_outputs)
        logger.info("Morning Brief generated:\n%s", brief[:200] + "...")

    # Audit log
    log_path = Path(config["logging"]["audit_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": args.mode,
            "records_processed": len(records),
        }) + "\n")


def _ingest_edgar(config: dict) -> list:
    from nlp_processor import TextRecord
    records = []
    try:
        from sec_edgar_downloader import Downloader
        import tempfile, os
        dl = Downloader("FSIP", "alerts@yourdomain.com", tempfile.mkdtemp())
        filing_types = config["sec_edgar"].get("filing_types", ["10-K", "10-Q"])
        logger._ = logging.getLogger("nlp.edgar")
        for ft in filing_types:
            try:
                dl.get(ft, "AAPL", limit=1, download_details=True)
            except Exception:
                pass
    except ImportError:
        logging.getLogger("nlp.edgar").warning(
            "sec-edgar-downloader not installed — EDGAR ingestion skipped"
        )
    return records


def _ingest_fomc(config: dict) -> list:
    """Fetch most recent FOMC statement text."""
    from nlp_processor import TextRecord
    import requests
    records = []
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm"
        resp = requests.get(url, timeout=15)
        if resp.ok:
            text = resp.text[:8000]
            records.append(TextRecord(
                doc_id="fomc_latest",
                source="fomc",
                ticker=None,
                date=pd.Timestamp.today(),
                text=text,
            ))
    except Exception as exc:
        logging.getLogger("nlp.fomc").warning("FOMC fetch failed: %s", exc)
    return records


def _ingest_news(config: dict) -> list:
    """Fetch RSS news feeds."""
    from nlp_processor import TextRecord
    import feedparser
    records = []
    for feed_cfg in config.get("news", {}).get("feeds", []):
        try:
            feed = feedparser.parse(feed_cfg["url"])
            for entry in feed.entries[:20]:
                text = getattr(entry, "summary", "") or getattr(entry, "title", "")
                if text:
                    records.append(TextRecord(
                        doc_id=entry.get("id", entry.get("link", "unknown")),
                        source="news",
                        ticker=None,
                        date=pd.Timestamp.today(),
                        text=f"{entry.get('title', '')} {text}",
                    ))
        except Exception as exc:
            logging.getLogger("nlp.news").warning("Feed %s failed: %s", feed_cfg["url"], exc)
    return records


if __name__ == "__main__":
    main()
