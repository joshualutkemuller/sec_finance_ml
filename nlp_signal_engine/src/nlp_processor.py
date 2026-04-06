"""
NLP Processor — FinBERT sentiment scoring + spaCy NER pipeline.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Processes raw text from SEC filings, FOMC statements, earnings transcripts,
and news articles into structured sentiment and entity signals.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TextRecord:
    """A single unit of text to be processed."""
    doc_id: str
    source: str          # "sec_10k" | "sec_10q" | "fomc" | "news" | "earnings"
    ticker: str | None
    date: pd.Timestamp
    text: str


@dataclass
class NLPResult:
    """Structured output for a single TextRecord."""
    doc_id: str
    source: str
    ticker: str | None
    date: pd.Timestamp
    # Sentiment
    sentiment_score: float        # -1 (bearish) to +1 (bullish)
    sentiment_label: str          # "positive" | "negative" | "neutral"
    sentiment_confidence: float
    # Entities
    mentioned_tickers: list[str]
    mentioned_organizations: list[str]
    # Finance-specific signals
    risk_keyword_score: float     # 0–1, fraction of risk keywords found
    hawkishness_score: float      # 0–1 for FOMC texts; NaN otherwise
    uncertainty_score: float      # hedging language density
    raw_text_len: int


class NLPProcessor:
    """
    Runs FinBERT + spaCy on a batch of TextRecords.

    FinBERT is a BERT model fine-tuned on financial text and produces
    positive/negative/neutral sentiment with calibrated probabilities.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["nlp"]
        self._finbert_pipe = None
        self._spacy_nlp = None
        self._risk_keywords = self._flatten_keywords(config.get("sec_edgar", {}).get("risk_keywords", {}))
        self._hawkish = set(config.get("fomc", {}).get("hawkish_terms", []))
        self._dovish = set(config.get("fomc", {}).get("dovish_terms", []))

    def process_batch(self, records: list[TextRecord]) -> list[NLPResult]:
        """Process a list of TextRecords and return NLPResults."""
        self._ensure_loaded()
        results = []
        batch_size = self.cfg.get("batch_size", 32)

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            texts = [r.text[:self.cfg.get("max_seq_len", 512) * 4] for r in batch]

            sentiments = self._finbert_sentiment(texts)
            for j, (rec, sent) in enumerate(zip(batch, sentiments)):
                entities = self._extract_entities(rec.text)
                results.append(NLPResult(
                    doc_id=rec.doc_id,
                    source=rec.source,
                    ticker=rec.ticker,
                    date=rec.date,
                    sentiment_score=sent["score"],
                    sentiment_label=sent["label"],
                    sentiment_confidence=sent["confidence"],
                    mentioned_tickers=entities["tickers"],
                    mentioned_organizations=entities["orgs"],
                    risk_keyword_score=self._risk_keyword_score(rec.text),
                    hawkishness_score=self._hawkishness_score(rec.text) if rec.source == "fomc" else float("nan"),
                    uncertainty_score=self._uncertainty_score(rec.text),
                    raw_text_len=len(rec.text),
                ))
            logger.debug("Processed batch %d/%d", i + batch_size, len(records))

        logger.info("NLP processing complete: %d records", len(results))
        return results

    def to_dataframe(self, results: list[NLPResult]) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in results])

    # ── Private ────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._finbert_pipe is None:
            self._load_finbert()
        if self._spacy_nlp is None:
            self._load_spacy()

    def _load_finbert(self) -> None:
        try:
            from transformers import pipeline
            model_id = self.cfg.get("finbert_model", "ProsusAI/finbert")
            self._finbert_pipe = pipeline(
                "text-classification",
                model=model_id,
                tokenizer=model_id,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded: %s", model_id)
        except Exception as exc:
            logger.warning("FinBERT load failed (%s) — falling back to lexicon-based sentiment", exc)
            self._finbert_pipe = "lexicon"

    def _load_spacy(self) -> None:
        try:
            import spacy
            model = self.cfg.get("spacy_model", "en_core_web_lg")
            self._spacy_nlp = spacy.load(model)
            logger.info("spaCy loaded: %s", model)
        except Exception as exc:
            logger.warning("spaCy load failed (%s) — entity extraction disabled", exc)
            self._spacy_nlp = None

    def _finbert_sentiment(self, texts: list[str]) -> list[dict]:
        if self._finbert_pipe == "lexicon":
            return [self._lexicon_sentiment(t) for t in texts]
        try:
            raw = self._finbert_pipe(texts)
            results = []
            for preds in raw:
                label_scores = {p["label"]: p["score"] for p in preds}
                pos = label_scores.get("positive", 0)
                neg = label_scores.get("negative", 0)
                neu = label_scores.get("neutral", 0)
                score = pos - neg
                label = max(label_scores, key=label_scores.get)
                results.append({"score": score, "label": label, "confidence": label_scores[label]})
            return results
        except Exception as exc:
            logger.warning("FinBERT inference failed: %s", exc)
            return [self._lexicon_sentiment(t) for t in texts]

    def _lexicon_sentiment(self, text: str) -> dict:
        """Simple word-count fallback sentiment."""
        positive_words = {"increase", "strong", "growth", "beat", "profit", "positive", "improve"}
        negative_words = {"decline", "loss", "weak", "miss", "concern", "risk", "uncertainty"}
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        pos = len(tokens & positive_words)
        neg = len(tokens & negative_words)
        total = pos + neg + 1
        score = (pos - neg) / total
        label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return {"score": score, "label": label, "confidence": abs(score)}

    def _extract_entities(self, text: str) -> dict:
        tickers, orgs = [], []
        # Regex ticker extraction: 1-5 uppercase letters in parentheses or standalone
        tickers = re.findall(r'\b[A-Z]{1,5}\b', text[:2000])
        tickers = [t for t in tickers if len(t) >= 2 and t not in
                   {"THE", "AND", "FOR", "INC", "LLC", "SEC", "IPO", "CEO", "CFO", "MD",
                    "FOMC", "FED", "USD", "UST", "HY", "IG", "OAS", "ETF", "REITs"}]
        if self._spacy_nlp:
            try:
                doc = self._spacy_nlp(text[:3000])
                orgs = list({ent.text for ent in doc.ents if ent.label_ == "ORG"})
            except Exception:
                pass
        return {"tickers": list(set(tickers))[:20], "orgs": orgs[:10]}

    def _risk_keyword_score(self, text: str) -> float:
        text_lower = text.lower()
        found = sum(1 for kw in self._risk_keywords if kw in text_lower)
        return min(found / max(len(self._risk_keywords), 1), 1.0)

    def _hawkishness_score(self, text: str) -> float:
        text_lower = text.lower()
        hawkish = sum(1 for t in self._hawkish if t in text_lower)
        dovish = sum(1 for t in self._dovish if t in text_lower)
        total = hawkish + dovish
        return hawkish / total if total > 0 else 0.5

    def _uncertainty_score(self, text: str) -> float:
        hedge_words = {"may", "might", "could", "uncertain", "unclear", "possible",
                       "potential", "risk", "concern", "subject to", "contingent"}
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return 0.0
        hedge_count = sum(1 for t in tokens if t in hedge_words)
        return min(hedge_count / len(tokens) * 20, 1.0)

    @staticmethod
    def _flatten_keywords(keyword_dict: dict) -> list[str]:
        kws = []
        for group_kws in keyword_dict.values():
            kws.extend(group_kws)
        return kws
