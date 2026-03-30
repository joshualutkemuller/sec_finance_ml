"""
src — Repo & Financing Cost Forecaster package.

Modules
-------
data_ingestion      : FinancingDataLoader — loads and merges all data sources.
feature_engineering : FinancingCostFeatureEngineer — builds lag, calendar, macro features.
model               : MultiHorizonForecaster — trains and serves LightGBM per horizon.
alerts              : ForecastAlertDispatcher — fires email / Teams / log alerts.
powerbi_connector   : PowerBIConnector — pushes forecast rows to PowerBI streaming dataset.
main                : Pipeline orchestration entry point.
"""
