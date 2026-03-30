"""
src — Short Squeeze Probability Scorer

This package contains all modules for ingesting short interest data,
engineering squeeze-relevant features, scoring securities with XGBoost,
classifying risk tiers, dispatching alerts, and pushing results to PowerBI.

Modules:
    data_ingestion      — Load and merge short interest, borrow, and price data.
    feature_engineering — Build squeeze-signal features from raw data.
    model               — XGBoost scorer and risk tier classification.
    alerts              — Alert dataclass, dispatcher, and factory.
    powerbi_connector   — Push scored results to PowerBI streaming dataset.
    main                — Pipeline orchestration and CLI entry point.
"""
