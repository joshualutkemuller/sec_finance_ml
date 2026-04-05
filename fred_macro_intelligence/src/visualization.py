"""
Visualization Library — 10 Plotly chart types for securities finance macro intelligence.

All charts return plotly.graph_objects.Figure objects:
  - Renderable in Jupyter, Dash, or exported as PNG/SVG via kaleido
  - Configured for PowerBI embedding via iframe or Plotly JS
  - Dark and light theme support

Chart Index:
  1.  regime_timeline()           — Rate regime history with SOFR overlay
  2.  yield_curve_animation()     — Animated yield curve morphing over time
  3.  rolling_correlation_matrix()— Rolling 60d correlation heatmap
  4.  spread_forecast_bands()     — Forecast with LSTM uncertainty envelope
  5.  funding_stress_dashboard()  — 4-panel liquidity/stress dashboard
  6.  agency_demand_heatmap()     — Calendar heatmap of lending utilization
  7.  shap_waterfall()            — SHAP feature attribution waterfall
  8.  curve_pca_biplot()          — 3D PCA factor space trajectory
  9.  sensitivity_tornado()       — What-if sensitivity tornado chart
  10. regime_transition_matrix()  — Regime transition probability heatmap
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not installed — visualizations unavailable")


def _check_plotly() -> None:
    if not PLOTLY_AVAILABLE:
        raise ImportError("Install plotly: pip install plotly")


REGIME_COLORS = {
    "Easing / QE": "#2ecc71",
    "Neutral / Normal": "#3498db",
    "Hiking Cycle": "#f39c12",
    "Stress / Crisis": "#e74c3c",
    "Late Cycle / Inversion": "#9b59b6",
}


# ── Chart 1 — Regime Timeline ─────────────────────────────────────────────────

def regime_timeline(
    raw: pd.DataFrame,
    regime_df: pd.DataFrame,
    rate_col: str = "sofr",
    title: str = "Macro Rate Regime Timeline",
    theme: str = "plotly_white",
) -> "go.Figure":
    """Overlay regime color bands on a rate time series.

    Args:
        raw: DataFrame with rate columns (date index).
        regime_df: DataFrame with columns regime_label, regime_color (date index).
        rate_col: Rate column from raw to plot as the main line.
        title: Chart title.
        theme: Plotly template name.
    """
    _check_plotly()
    fig = go.Figure()

    # Add colored background bands per regime segment
    merged = regime_df[["regime_label", "regime_color"]].copy()
    merged["date"] = merged.index
    regime_changes = merged[merged["regime_label"] != merged["regime_label"].shift()].index

    for i, start in enumerate(regime_changes):
        end = regime_changes[i + 1] if i + 1 < len(regime_changes) else merged.index[-1]
        label = merged.loc[start, "regime_label"]
        color = merged.loc[start, "regime_color"]
        fig.add_vrect(
            x0=str(start.date()),
            x1=str(end.date()),
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text=label if (end - start).days > 120 else "",
            annotation_position="top left",
            annotation_font_size=10,
        )

    # Main rate line
    if rate_col in raw.columns:
        fig.add_trace(go.Scatter(
            x=raw.index,
            y=raw[rate_col],
            mode="lines",
            name=rate_col.upper(),
            line=dict(color="#2c3e50", width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" + rate_col.upper() + ": %{y:.3f}%<extra></extra>",
        ))

    # Regime change markers
    for dt in regime_changes:
        if dt in regime_df.index:
            label = regime_df.loc[dt, "regime_label"]
            fig.add_vline(x=str(dt.date()), line_dash="dot", line_color="gray",
                          line_width=1, opacity=0.5)

    fig.update_layout(
        title=title,
        template=theme,
        height=480,
        xaxis_title="Date",
        yaxis_title="Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ── Chart 2 — Yield Curve Animation ──────────────────────────────────────────

def yield_curve_animation(
    raw: pd.DataFrame,
    tenors: list[tuple[str, float]] | None = None,
    resample_freq: str = "W",
    title: str = "Yield Curve Evolution",
    theme: str = "plotly_white",
) -> "go.Figure":
    """Animated yield curve morphing over time with a date slider.

    Args:
        raw: DataFrame with tenor columns and date index.
        tenors: List of (column_name, years_to_maturity) tuples.
                Defaults to standard tenors if present in raw.
        resample_freq: Resample frequency for animation frames (default weekly).
        theme: Plotly template name.
    """
    _check_plotly()

    default_tenors = [
        ("t_1mo", 1/12), ("t_3mo", 0.25), ("t_6mo", 0.5),
        ("t_1yr", 1), ("t_2yr", 2), ("t_5yr", 5),
        ("t_10yr", 10), ("t_30yr", 30),
    ]
    tenors = tenors or [(col, yrs) for col, yrs in default_tenors if col in raw.columns]
    if not tenors:
        raise ValueError("No tenor columns found in raw DataFrame")

    tenor_cols = [t[0] for t in tenors]
    tenor_yrs = [t[1] for t in tenors]

    # Resample to reduce frame count
    curve_data = raw[tenor_cols].resample(resample_freq).last().dropna(how="all")

    frames = []
    for dt, row in curve_data.iterrows():
        y_vals = [row[c] for c in tenor_cols]
        frames.append(go.Frame(
            data=[go.Scatter(
                x=tenor_yrs,
                y=y_vals,
                mode="lines+markers",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=8),
                name=str(dt.date()),
                hovertemplate="<b>%{x:.1f}yr</b>: %{y:.3f}%<extra></extra>",
            )],
            name=str(dt.date()),
        ))

    # Initial frame
    first_row = curve_data.iloc[0]
    fig = go.Figure(
        data=[go.Scatter(
            x=tenor_yrs,
            y=[first_row[c] for c in tenor_cols],
            mode="lines+markers",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=8),
        )],
        frames=frames,
    )

    # Add zero line for inversion visibility
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=title,
        template=theme,
        height=480,
        xaxis=dict(title="Maturity (Years)", tickvals=tenor_yrs,
                   ticktext=[f"{y:.1f}yr" if y >= 1 else f"{int(y*12)}mo" for y in tenor_yrs]),
        yaxis_title="Yield (%)",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0), mode="immediate")],
                        label=f.name, method="animate") for f in frames],
            transition=dict(duration=0),
            x=0.05, y=0, len=0.9,
            currentvalue=dict(prefix="Date: ", visible=True, xanchor="center"),
        )],
    )
    return fig


# ── Chart 3 — Rolling Correlation Matrix ─────────────────────────────────────

def rolling_correlation_matrix(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    window: int = 60,
    title: str = "Rolling 60-Day Correlation Matrix",
    theme: str = "plotly_white",
) -> "go.Figure":
    """Heatmap of rolling correlations between key macro series."""
    _check_plotly()

    default_cols = ["sofr", "hy_oas", "ig_oas", "vix", "curve_10y2y",
                    "rrp_balance_bn", "funding_stress_index"]
    plot_cols = [c for c in (cols or default_cols) if c in features.columns]

    if len(plot_cols) < 2:
        raise ValueError("Need at least 2 columns for correlation matrix")

    corr = features[plot_cols].tail(window).corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
    ))
    fig.update_layout(
        title=f"{title} (window={window}d)",
        template=theme,
        height=520,
        xaxis=dict(tickangle=-30),
    )
    return fig


# ── Chart 4 — Spread Forecast with Uncertainty Bands ─────────────────────────

def spread_forecast_bands(
    historical: pd.DataFrame,
    forecasts_lgbm: dict[int, np.ndarray],
    forecasts_lstm: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    target_col: str = "hy_oas",
    history_days: int = 120,
    title: str = "Spread Forecast with Uncertainty Bands",
    theme: str = "plotly_white",
) -> "go.Figure":
    """Multi-horizon forecast with optional LSTM 90% confidence bands.

    Args:
        historical: Raw DataFrame with target column and date index.
        forecasts_lgbm: {horizon_days: last_N_predictions_array}
        forecasts_lstm: {horizon_days: (mean, lower_90, upper_90)} or None
        target_col: Column to plot.
        history_days: How many historical days to show.
    """
    _check_plotly()
    fig = go.Figure()

    # Historical
    hist = historical[target_col].tail(history_days) if target_col in historical.columns else pd.Series()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            name=f"{target_col.upper()} (Historical)",
            line=dict(color="#2c3e50", width=2),
        ))

    # Extend date axis for forecasts
    last_date = historical.index[-1] if not historical.empty else pd.Timestamp.today()
    forecast_colors = {1: "#3498db", 3: "#f39c12", 5: "#e74c3c", 10: "#9b59b6"}

    for h, preds in forecasts_lgbm.items():
        future_date = last_date + pd.tseries.offsets.BusinessDay(h)
        val = float(preds[-1]) if len(preds) > 0 else np.nan
        fig.add_trace(go.Scatter(
            x=[last_date, future_date],
            y=[hist.iloc[-1] if not hist.empty else np.nan, val],
            mode="lines+markers",
            name=f"LightGBM +{h}d",
            line=dict(color=forecast_colors.get(h, "#95a5a6"), width=2, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
        ))

    # LSTM uncertainty bands
    if forecasts_lstm:
        for h, (mean_arr, lower_arr, upper_arr) in forecasts_lstm.items():
            future_date = last_date + pd.tseries.offsets.BusinessDay(h)
            mean_val = float(mean_arr[-1]) if len(mean_arr) > 0 else np.nan
            lower_val = float(lower_arr[-1]) if len(lower_arr) > 0 else np.nan
            upper_val = float(upper_arr[-1]) if len(upper_arr) > 0 else np.nan

            fig.add_trace(go.Scatter(
                x=[last_date, future_date, future_date, last_date],
                y=[hist.iloc[-1] if not hist.empty else mean_val,
                   upper_val, lower_val,
                   hist.iloc[-1] if not hist.empty else mean_val],
                fill="toself",
                fillcolor=forecast_colors.get(h, "#95a5a6"),
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"LSTM +{h}d 90% CI",
                showlegend=True,
            ))

    fig.update_layout(
        title=f"{title} — {target_col.upper()}",
        template=theme,
        height=480,
        xaxis_title="Date",
        yaxis_title="Spread (bps)" if "oas" in target_col.lower() else "Rate (%)",
        hovermode="x unified",
    )
    return fig


# ── Chart 5 — Funding Stress Dashboard ────────────────────────────────────────

def funding_stress_dashboard(
    features: pd.DataFrame,
    history_days: int = 252,
    theme: str = "plotly_white",
) -> "go.Figure":
    """4-panel funding stress and liquidity dashboard."""
    _check_plotly()
    data = features.tail(history_days)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "TED Spread & Z-Score",
            "RRP Balance vs SOFR",
            "VIX vs HY OAS (Regime-Colored)",
            "Reserve Balances YoY Change",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    # Panel 1: TED Spread
    if "ted_spread" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["ted_spread"],
                                 name="TED Spread", line=dict(color="#e74c3c", width=2)),
                      row=1, col=1)
    if "ted_spread_z60" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["ted_spread_z60"],
                                 name="Z-Score (60d)", line=dict(color="#3498db", width=1.5, dash="dot"),
                                 yaxis="y2"),
                      row=1, col=1)

    # Panel 2: RRP vs SOFR scatter
    if "rrp_balance_bn" in data.columns and "sofr" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["rrp_balance_bn"],
            y=data["sofr"],
            mode="markers",
            marker=dict(
                color=list(range(len(data))),
                colorscale="Viridis",
                size=5,
                colorbar=dict(title="Time", len=0.4, y=0.8),
                showscale=True,
            ),
            name="RRP vs SOFR",
            text=data.index.strftime("%Y-%m-%d"),
            hovertemplate="<b>%{text}</b><br>RRP: $%{x:.0f}bn<br>SOFR: %{y:.3f}%<extra></extra>",
        ), row=1, col=2)

    # Panel 3: VIX vs HY OAS
    if "vix" in data.columns and "hy_oas" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["vix"],
            y=data["hy_oas"],
            mode="markers",
            marker=dict(size=5, color=data.get("funding_stress_index", pd.Series(0, index=data.index)),
                        colorscale="YlOrRd", showscale=False),
            name="VIX vs HY OAS",
            text=data.index.strftime("%Y-%m-%d"),
            hovertemplate="<b>%{text}</b><br>VIX: %{x:.1f}<br>HY OAS: %{y:.0f}bps<extra></extra>",
        ), row=2, col=1)

    # Panel 4: Reserve Balances YoY
    if "reserves_yoy" in data.columns:
        colors = ["#2ecc71" if v >= 0 else "#e74c3c"
                  for v in data["reserves_yoy"].fillna(0)]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data["reserves_yoy"],
            marker_color=colors,
            name="Reserves YoY",
        ), row=2, col=2)

    fig.update_layout(
        title="Funding Stress & Liquidity Dashboard",
        template=theme,
        height=700,
        showlegend=False,
        hovermode="closest",
    )
    return fig


# ── Chart 6 — Agency Demand Calendar Heatmap ──────────────────────────────────

def agency_demand_heatmap(
    demand_signal: pd.Series,
    title: str = "Agency Lending Demand Signal — Calendar View",
    theme: str = "plotly_white",
) -> "go.Figure":
    """GitHub-style calendar heatmap of agency lending demand."""
    _check_plotly()
    df = demand_signal.to_frame(name="value").copy()
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype(int)
    df["day"] = df.index.dayofweek  # 0=Mon

    years = sorted(df["year"].unique())
    fig = make_subplots(rows=len(years), cols=1,
                        subplot_titles=[str(y) for y in years],
                        vertical_spacing=0.06)

    for row_i, year in enumerate(years, 1):
        yr_df = df[df["year"] == year]
        pivot = yr_df.pivot_table(index="day", columns="week", values="value", aggfunc="mean")

        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=["Mon", "Tue", "Wed", "Thu", "Fri"],
            colorscale="RdYlGn",
            zmid=0,
            showscale=(row_i == 1),
            hovertemplate="Week %{x}, %{y}<br>Signal: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Demand\nSignal", len=0.25, y=0.9),
        ), row=row_i, col=1)

    fig.update_layout(
        title=title,
        template=theme,
        height=200 * len(years) + 100,
    )
    return fig


# ── Chart 7 — SHAP Waterfall ──────────────────────────────────────────────────

def shap_waterfall(
    shap_values: pd.Series,
    base_value: float,
    prediction: float,
    title: str = "Macro Driver Attribution (SHAP)",
    theme: str = "plotly_white",
) -> "go.Figure":
    """Waterfall chart showing SHAP feature contributions for a single prediction."""
    _check_plotly()

    # Sort by absolute value
    sv = shap_values.sort_values(key=abs, ascending=True)
    features = sv.index.tolist()
    values = sv.values.tolist()

    cumulative = base_value
    running = [base_value]
    for v in values:
        cumulative += v
        running.append(cumulative)

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(values) + ["total"],
        x=values + [prediction],
        y=features + ["Prediction"],
        connector=dict(line=dict(color="rgb(63,63,63)")),
        decreasing=dict(marker=dict(color="#e74c3c")),
        increasing=dict(marker=dict(color="#2ecc71")),
        totals=dict(marker=dict(color="#3498db")),
        text=[f"{v:+.4f}" for v in values] + [f"{prediction:.4f}"],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        template=theme,
        height=max(400, len(features) * 35 + 100),
        xaxis_title="SHAP Contribution",
        yaxis=dict(automargin=True),
        showlegend=False,
    )
    return fig


# ── Chart 8 — Curve PCA Biplot ────────────────────────────────────────────────

def curve_pca_biplot(
    features: pd.DataFrame,
    regime_df: pd.DataFrame | None = None,
    n_history: int = 504,
    title: str = "Yield Curve PCA Factor Space",
    theme: str = "plotly_white",
) -> "go.Figure":
    """3D scatter of yield curve in PCA factor space, colored by regime."""
    _check_plotly()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    tenor_cols = [c for c in ["t_1mo", "t_3mo", "t_6mo", "t_1yr",
                               "t_2yr", "t_5yr", "t_10yr", "t_30yr"]
                  if c in features.columns]
    if len(tenor_cols) < 3:
        raise ValueError("Need at least 3 tenor columns for PCA biplot")

    data = features[tenor_cols].tail(n_history).dropna()
    X_scaled = StandardScaler().fit_transform(data)
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    exp_var = pca.explained_variance_ratio_

    colors = "rgba(52,152,219,0.7)"
    if regime_df is not None and "regime_color" in regime_df.columns:
        aligned = regime_df["regime_color"].reindex(data.index).fillna("rgba(150,150,150,0.7)")
        colors = aligned.values.tolist()

    fig = go.Figure(go.Scatter3d(
        x=components[:, 0],
        y=components[:, 1],
        z=components[:, 2],
        mode="lines+markers",
        marker=dict(size=3, color=colors, opacity=0.8),
        line=dict(color="rgba(44,62,80,0.3)", width=1),
        text=data.index.strftime("%Y-%m-%d"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"PC1 (Level, {exp_var[0]:.1%}): %{{x:.3f}}<br>"
            f"PC2 (Slope, {exp_var[1]:.1%}): %{{y:.3f}}<br>"
            f"PC3 (Curve, {exp_var[2]:.1%}): %{{z:.3f}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        template=theme,
        height=600,
        scene=dict(
            xaxis_title=f"PC1 Level ({exp_var[0]:.1%})",
            yaxis_title=f"PC2 Slope ({exp_var[1]:.1%})",
            zaxis_title=f"PC3 Curvature ({exp_var[2]:.1%})",
        ),
    )
    return fig


# ── Chart 9 — Sensitivity Tornado ────────────────────────────────────────────

def sensitivity_tornado(
    sensitivity_df: pd.DataFrame,
    target_name: str = "Agency Demand",
    title: str | None = None,
    theme: str = "plotly_white",
) -> "go.Figure":
    """Tornado chart of ±1σ sensitivity of target to each macro input.

    Args:
        sensitivity_df: DataFrame with columns ['feature', 'up_impact', 'dn_impact'].
                        Rows sorted by |up_impact| + |dn_impact| descending.
        target_name: Label for the target variable.
        theme: Plotly template.
    """
    _check_plotly()
    df = sensitivity_df.sort_values(
        by=[c for c in ["up_impact", "dn_impact"] if c in sensitivity_df.columns],
        key=lambda s: s.abs(),
        ascending=True,
    ).tail(15)  # show top 15

    fig = go.Figure()

    if "up_impact" in df.columns:
        fig.add_trace(go.Bar(
            name="+1σ shock",
            x=df["up_impact"],
            y=df["feature"],
            orientation="h",
            marker_color="#2ecc71",
            opacity=0.85,
        ))
    if "dn_impact" in df.columns:
        fig.add_trace(go.Bar(
            name="-1σ shock",
            x=df["dn_impact"],
            y=df["feature"],
            orientation="h",
            marker_color="#e74c3c",
            opacity=0.85,
        ))

    fig.update_layout(
        title=title or f"Sensitivity of {target_name} to ±1σ Macro Shocks",
        template=theme,
        height=max(400, len(df) * 32 + 120),
        barmode="overlay",
        xaxis_title=f"Δ {target_name}",
        yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Chart 10 — Regime Transition Matrix ──────────────────────────────────────

def regime_transition_matrix(
    transition_matrix: pd.DataFrame,
    title: str = "Macro Regime Transition Probabilities",
    theme: str = "plotly_white",
) -> "go.Figure":
    """Heatmap of historical regime transition probabilities."""
    _check_plotly()
    fig = go.Figure(go.Heatmap(
        z=transition_matrix.values,
        x=transition_matrix.columns.tolist(),
        y=transition_matrix.index.tolist(),
        colorscale="Blues",
        zmin=0,
        zmax=1,
        text=transition_matrix.round(3).values,
        texttemplate="%{text:.1%}",
        textfont=dict(size=12),
        hovertemplate=(
            "From: <b>%{y}</b><br>"
            "To: <b>%{x}</b><br>"
            "Probability: %{z:.1%}<extra></extra>"
        ),
        colorbar=dict(title="P(transition)", tickformat=".0%"),
    ))
    fig.update_layout(
        title=title,
        template=theme,
        height=480,
        xaxis=dict(title="Next Regime", tickangle=-30),
        yaxis=dict(title="Current Regime", automargin=True),
    )
    return fig


# ── Convenience: export chart to file ────────────────────────────────────────

def export_chart(fig: "go.Figure", path: str, fmt: str = "png") -> None:
    """Export a Plotly figure to disk.

    Args:
        fig: Plotly figure.
        path: Output file path (include extension).
        fmt: 'png', 'svg', or 'html'.
    """
    _check_plotly()
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        fig.write_html(path)
    else:
        try:
            fig.write_image(path, format=fmt)
        except Exception as exc:
            logger.warning(
                "Static export failed (%s) — try: pip install kaleido. "
                "Falling back to HTML.", exc
            )
            fig.write_html(path.rsplit(".", 1)[0] + ".html")
