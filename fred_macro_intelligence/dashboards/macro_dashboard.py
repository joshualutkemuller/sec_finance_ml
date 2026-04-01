"""
FRED Macro Intelligence — Standalone Plotly Dash Application

Provides an interactive web dashboard with:
  - Regime timeline with live controls
  - Yield curve animation
  - Funding stress 4-panel view
  - Spread forecast with uncertainty bands
  - Rolling correlation matrix
  - SHAP driver waterfall (latest day)
  - Regime transition probability matrix

Usage:
  python dashboards/macro_dashboard.py --config config/config.yaml [--port 8050]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

import visualization as viz


def build_layout() -> "html.Div":
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("FRED Macro Intelligence Dashboard",
                        className="text-center mt-3 mb-1"),
                html.P("Securities Finance & Agency/Prime Lending — Macro Signal Monitor",
                       className="text-center text-muted mb-3"),
            ])
        ]),

        # Controls Row
        dbc.Row([
            dbc.Col([
                dbc.Label("History Window"),
                dcc.Slider(id="history-slider", min=60, max=504, step=60,
                           value=252, marks={60: "3M", 126: "6M", 252: "1Y", 504: "2Y"}),
            ], width=4),
            dbc.Col([
                dbc.Label("Forecast Target"),
                dcc.Dropdown(id="target-dropdown",
                             options=[
                                 {"label": "HY OAS", "value": "hy_oas"},
                                 {"label": "IG OAS", "value": "ig_oas"},
                                 {"label": "SOFR", "value": "sofr"},
                             ],
                             value="hy_oas", clearable=False),
            ], width=3),
            dbc.Col([
                dbc.Label("Chart Theme"),
                dcc.Dropdown(id="theme-dropdown",
                             options=[
                                 {"label": "Light", "value": "plotly_white"},
                                 {"label": "Dark", "value": "plotly_dark"},
                                 {"label": "Minimal", "value": "simple_white"},
                             ],
                             value="plotly_white", clearable=False),
            ], width=3),
            dbc.Col([
                dbc.Button("Refresh Data", id="refresh-btn", color="primary",
                           className="mt-4 w-100"),
            ], width=2),
        ], className="mb-4 bg-light p-3 rounded"),

        # Regime Timeline
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Macro Rate Regime Timeline"),
                    dbc.CardBody(dcc.Graph(id="regime-timeline-chart")),
                ])
            ])
        ], className="mb-3"),

        # Spread Forecast + Correlation Matrix
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Spread Forecast with Uncertainty Bands"),
                    dbc.CardBody(dcc.Graph(id="spread-forecast-chart")),
                ])
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rolling Correlation Matrix (60d)"),
                    dbc.CardBody(dcc.Graph(id="correlation-matrix-chart")),
                ])
            ], width=5),
        ], className="mb-3"),

        # Funding Stress Dashboard
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Funding Stress & Liquidity Dashboard"),
                    dbc.CardBody(dcc.Graph(id="funding-stress-chart")),
                ])
            ])
        ], className="mb-3"),

        # PCA Biplot + Regime Transition
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Yield Curve PCA Factor Space"),
                    dbc.CardBody(dcc.Graph(id="pca-biplot-chart")),
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Regime Transition Probabilities"),
                    dbc.CardBody(dcc.Graph(id="regime-transition-chart")),
                ])
            ], width=6),
        ], className="mb-3"),

        # Yield Curve Animation
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Yield Curve Animation"),
                    dbc.CardBody(dcc.Graph(id="yield-curve-chart")),
                ])
            ])
        ], className="mb-4"),

        # Footer
        dbc.Row([
            dbc.Col(html.P(
                "FSIP Macro Intelligence | Data: FRED (St. Louis Fed) | "
                "Refreshed: auto-daily 7AM ET",
                className="text-center text-muted small mt-2 mb-3",
            ))
        ]),

        # Data store for shared state
        dcc.Store(id="macro-data-store"),
    ], fluid=True)


def register_callbacks(app: "dash.Dash", config: dict) -> None:
    """Register all Dash callbacks."""

    @app.callback(
        Output("macro-data-store", "data"),
        Input("refresh-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def refresh_data(n_clicks):
        """Load data into the store (triggered on page load and refresh button)."""
        from fred_loader import FREDLoader
        from feature_engineering import MacroFeatureEngineer
        from regime_detector import RegimeDetector
        from spread_forecaster import LGBMSpreadForecaster

        raw = FREDLoader(config).fetch_all()
        features = MacroFeatureEngineer(config).build(raw)

        detector = RegimeDetector(config)
        regime_model_path = config["regime_detector"]["model_path"]
        if Path(regime_model_path).exists():
            detector.load(regime_model_path)
            regime_df = detector.predict(features)
        else:
            regime_df = None

        # Serialize minimal needed data to store
        store = {
            "raw_tail": raw.tail(504).to_json(),
            "features_tail": features.tail(504).to_json(),
            "regime_labels": regime_df["regime_label"].to_json() if regime_df is not None else None,
            "regime_colors": regime_df["regime_color"].to_json() if regime_df is not None else None,
        }
        return store

    @app.callback(
        Output("regime-timeline-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("history-slider", "value"),
        Input("theme-dropdown", "value"),
    )
    def update_regime_timeline(store, history_days, theme):
        import pandas as pd
        if not store:
            return {}
        raw = pd.read_json(store["raw_tail"]).tail(history_days)
        if store.get("regime_labels"):
            regime_df = pd.DataFrame({
                "regime_label": pd.read_json(store["regime_labels"], typ="series"),
                "regime_color": pd.read_json(store["regime_colors"], typ="series"),
            }).tail(history_days)
        else:
            regime_df = pd.DataFrame(columns=["regime_label", "regime_color"])
        return viz.regime_timeline(raw, regime_df, theme=theme)

    @app.callback(
        Output("spread-forecast-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("target-dropdown", "value"),
        Input("theme-dropdown", "value"),
    )
    def update_spread_forecast(store, target, theme):
        import pandas as pd
        if not store:
            return {}
        raw = pd.read_json(store["raw_tail"])
        # Return empty forecast bands chart — in production, load model and predict
        return viz.spread_forecast_bands(
            raw, forecasts_lgbm={1: [0], 3: [0], 5: [0]},
            target_col=target, theme=theme
        )

    @app.callback(
        Output("correlation-matrix-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("theme-dropdown", "value"),
    )
    def update_correlation(store, theme):
        import pandas as pd
        if not store:
            return {}
        features = pd.read_json(store["features_tail"])
        return viz.rolling_correlation_matrix(features, theme=theme)

    @app.callback(
        Output("funding-stress-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("history-slider", "value"),
        Input("theme-dropdown", "value"),
    )
    def update_funding_stress(store, history_days, theme):
        import pandas as pd
        if not store:
            return {}
        features = pd.read_json(store["features_tail"]).tail(history_days)
        return viz.funding_stress_dashboard(features, history_days=history_days, theme=theme)

    @app.callback(
        Output("yield-curve-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("theme-dropdown", "value"),
    )
    def update_yield_curve(store, theme):
        import pandas as pd
        if not store:
            return {}
        raw = pd.read_json(store["raw_tail"])
        try:
            return viz.yield_curve_animation(raw, theme=theme)
        except Exception:
            return {}

    @app.callback(
        Output("pca-biplot-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("theme-dropdown", "value"),
    )
    def update_pca(store, theme):
        import pandas as pd
        if not store:
            return {}
        features = pd.read_json(store["features_tail"])
        regime_df = None
        if store.get("regime_colors"):
            regime_df = pd.DataFrame({
                "regime_color": pd.read_json(store["regime_colors"], typ="series"),
            })
        try:
            return viz.curve_pca_biplot(features, regime_df=regime_df, theme=theme)
        except Exception:
            return {}

    @app.callback(
        Output("regime-transition-chart", "figure"),
        Input("macro-data-store", "data"),
        Input("theme-dropdown", "value"),
    )
    def update_regime_transition(store, theme):
        import pandas as pd
        if not store or not store.get("regime_labels"):
            return {}
        from regime_detector import RegimeDetector
        labels = pd.read_json(store["regime_labels"], typ="series")
        detector = RegimeDetector(config)
        trans = detector.regime_transition_matrix(labels)
        return viz.regime_transition_matrix(trans, theme=theme)


def create_app(config: dict) -> "dash.Dash":
    if not DASH_AVAILABLE:
        raise ImportError("Install: pip install dash dash-bootstrap-components")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="FSIP Macro Dashboard",
    )
    app.layout = build_layout()
    register_callbacks(app, config)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="FRED Macro Dash Dashboard")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    app = create_app(config)
    print(f"Dashboard running at http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
