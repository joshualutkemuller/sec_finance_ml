"""
Securities Lending Rate Anomaly Detector — source package.

This package implements a two-stage anomaly detection pipeline for
securities lending borrow rates:

  1. Isolation Forest  — static, cross-sectional anomaly scoring
  2. LSTM Autoencoder  — temporal, sequence-based reconstruction-error scoring

Modules
-------
data_ingestion      : Load and validate lending rate data from CSV, Databricks or REST API.
feature_engineering : Derive rolling statistics, spreads, momentum and liquidity features.
model               : IsolationForestDetector, LSTMAutoencoder, LSTMDetector,
                      and CombinedAnomalyDetector.
alerts              : Alert dataclasses, severity levels, and multi-channel dispatcher.
powerbi_connector   : Push scored results to the PowerBI Financing Solutions
                      Intelligence Platform.
main                : CLI entry point and end-to-end pipeline orchestration.
"""
