"""
FRED Loader — Fetches Federal Reserve Economic Data via the FRED API.

Supports:
  - Bulk fetch of all configured series
  - Local disk caching (Parquet) to minimize API calls
  - Forward-fill and interpolation for series with different release frequencies
  - Holiday/weekend alignment to trading calendar
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FREDLoader:
    """Fetch and cache FRED series, returning a single aligned DataFrame."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["fred"]
        self.cache_dir = Path(self.cfg.get("cache_dir", "data/fred_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_years = int(self.cfg.get("lookback_years", 10))
        self._fred = None  # lazy init

    def _get_fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
            except ImportError as exc:
                raise ImportError("Install fredapi: pip install fredapi") from exc
            api_key = os.environ.get("FRED_API_KEY") or self.cfg.get("api_key", "")
            if not api_key or api_key.startswith("${"):
                raise EnvironmentError("FRED_API_KEY environment variable not set")
            self._fred = Fred(api_key=api_key)
        return self._fred

    def fetch_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch all configured FRED series and return as a merged daily DataFrame.

        Args:
            force_refresh: If True, bypass cache and re-fetch from FRED API.

        Returns:
            DataFrame indexed by date, one column per FRED series (friendly name).
        """
        start_date = datetime.utcnow() - timedelta(days=365 * self.lookback_years)
        all_series: dict[str, pd.Series] = {}

        # Collect all series name→id mappings from config
        series_map: dict[str, str] = {}
        for section_key in ["rate_series", "curve_series", "spread_series",
                             "liquidity_series", "agency_series"]:
            section = self.cfg.get(section_key, {})
            series_map.update(section)

        for friendly_name, fred_id in series_map.items():
            s = self._fetch_series(friendly_name, fred_id, start_date, force_refresh)
            if s is not None:
                all_series[friendly_name] = s

        if not all_series:
            raise RuntimeError("No FRED series could be loaded")

        # Merge all series into a single DataFrame aligned to business days
        df = pd.DataFrame(all_series)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Reindex to business days and forward-fill gaps (weekends, holidays, release lags)
        bday_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(bday_range)
        df = df.ffill(limit=5)  # forward-fill up to 5 consecutive missing days
        df = df.dropna(how="all")

        # Apply lookback window
        cutoff = datetime.utcnow() - timedelta(days=365 * self.lookback_years)
        df = df[df.index >= pd.Timestamp(cutoff)]

        logger.info(
            "FRED data loaded: %d series, %d trading days (%s to %s)",
            len(df.columns),
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df

    def _fetch_series(
        self,
        friendly_name: str,
        fred_id: str,
        start_date: datetime,
        force_refresh: bool,
    ) -> pd.Series | None:
        """Fetch a single FRED series, using cache if available."""
        cache_path = self.cache_dir / f"{fred_id}.parquet"

        if not force_refresh and cache_path.exists():
            cached = pd.read_parquet(cache_path).squeeze()
            # Refresh if stale (older than 1 business day)
            last_date = cached.index.max() if hasattr(cached.index, "max") else pd.Timestamp("1970-01-01")
            if last_date >= pd.Timestamp(datetime.utcnow() - timedelta(days=2)):
                logger.debug("Cache hit: %s (%s)", fred_id, friendly_name)
                return cached

        logger.info("Fetching from FRED: %s (%s)", fred_id, friendly_name)
        try:
            fred = self._get_fred()
            raw = fred.get_series(
                fred_id,
                observation_start=start_date.strftime("%Y-%m-%d"),
            )
            raw.name = friendly_name
            raw = raw.dropna()
            # Cache to disk
            raw.to_frame().to_parquet(cache_path)
            return raw
        except Exception as exc:
            logger.warning("Failed to fetch %s (%s): %s", fred_id, friendly_name, exc)
            # Fall back to cache if it exists
            if cache_path.exists():
                logger.info("Using stale cache for %s", fred_id)
                return pd.read_parquet(cache_path).squeeze()
            return None

    def get_latest_snapshot(self) -> pd.Series:
        """Return the most recent row of all series as a named Series."""
        df = self.fetch_all()
        return df.iloc[-1]
