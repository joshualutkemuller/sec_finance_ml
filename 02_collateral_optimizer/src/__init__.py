"""
Collateral Optimization & Substitution Recommender — source package.

This package contains all modules required to run the collateral optimization
pipeline:

    data_ingestion      — load and merge collateral pool, eligibility schedules,
                          haircut tables, and transactions.
    feature_engineering — compute quality-scoring features from raw data.
    model               — LightGBM quality scorer and CVXPY optimizer.
    alerts              — alert dataclasses, dispatcher, and factory.
    powerbi_connector   — push optimization results to a PowerBI push dataset.
    main                — CLI entry-point and pipeline orchestration.
"""
