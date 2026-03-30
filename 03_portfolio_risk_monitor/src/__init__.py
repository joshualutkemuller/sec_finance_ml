"""
src — Portfolio Concentration Risk Early Warning Monitor

Package containing all modules for data ingestion, feature engineering,
Random Forest risk scoring, SHAP explainability, alert dispatch, and
PowerBI push integration.

Modules:
    data_ingestion      — PortfolioDataLoader: load and join raw data sources
    feature_engineering — ConcentrationFeatureEngineer: HHI, top-N, rolling features
    model               — RandomForestRiskScorer + RiskLevel enum
    alerts              — RiskAlert dataclass, RiskAlertDispatcher, RiskAlertFactory
    powerbi_connector   — PowerBIConnector: push scored rows to streaming dataset
    main                — Pipeline orchestration entry point
"""
