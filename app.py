import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# Configuration
# ============================================================
APP_TITLE = "Market Stress Monitor"
PAGE_ICON = "◉"
HORIZON = 3
CLASSIFICATION_THRESHOLD = 0.46
WARNING_THRESHOLD = 0.25

DATA_FILES = {
    "catboost": "catboost_oos_predictions_h3.csv",
    "ablation": "catboost_ablation_oos_predictions_h3.csv",
    "garch": "garch_oos_predictions_h3.csv",
    "shap": "catboost_h3_shap_values.csv",
}

COLORS = {
    "bg": "#0b1020",
    "panel": "#121a31",
    "border": "rgba(255,255,255,0.10)",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "danger": "#fb7185",
    "warning": "#fbbf24",
    "success": "#34d399",
    "garch": "#fb923c",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="DM Sans, sans-serif", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h"),
    margin=dict(l=10, r=10, t=30, b=10),
)

AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.08)",
    linecolor="rgba(255,255,255,0.08)",
    zerolinecolor="rgba(255,255,255,0.08)",
)

FEATURE_LABELS = {
    "vix_log": "VIX level (log)",
    "vix_close": "VIX level",
    "vix_mean_5d": "5-day VIX average",
    "vix_mean_10d": "10-day VIX average",
    "vix_z": "VIX deviation",
    "vix_chg": "VIX daily change",
    "BBB_SPREAD": "BBB credit spread",
    "TED_SPREAD": "TED spread",
    "BBB_SPREAD_chg_5d": "5-day BBB spread change",
    "dxy_close": "US dollar level",
    "dxy_ret": "Dollar return",
    "dxy_vol_7d": "7-day dollar volatility",
    "dxy_vol_14d": "14-day dollar volatility",
    "dxy_ret_z": "Dollar return deviation",
    "es_close": "S&P 500 level",
    "es_ret": "S&P 500 return",
    "es_abs_ret": "Absolute return",
    "es_ret_mean_5d": "5-day average return",
    "es_ret_mean_10d": "10-day average return",
    "es_volume_z": "Volume deviation",
    "es_range": "Intraday range",
    "US10Y": "10-year Treasury yield",
    "US3M": "3-month Treasury yield",
    "US2Y": "2-year Treasury yield",
    "T10Y_IE": "10-year inflation expectations",
    "YIELD_CURVE_SLOPE": "Yield curve slope",
    "FED_RATE_USD": "Fed funds rate",
    "sentiment_mean": "News sentiment",
    "sentiment_std_roll_5": "5-day sentiment dispersion",
    "sentiment_std_roll_10": "10-day sentiment dispersion",
    "stress_ratio": "Stress keyword ratio",
    "stress_ratio_roll_5": "5-day stress keyword ratio",
    "stress_ratio_roll_10": "10-day stress keyword ratio",
}


@dataclass
class Metrics:
    pr_auc: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    prec_at_5: float
    prec_at_10: float


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

.stApp {{
    background:
        radial-gradient(ellipse at 10% 0%,  rgba(34,211,238,0.10), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(167,139,250,0.09), transparent 32%),
        linear-gradient(180deg, {COLORS['bg']} 0%, #0d1428 100%);
    color: {COLORS['text']};
}}
.block-container {{ max-width:1500px; padding-top:1rem; padding-bottom:2rem; }}
[data-testid="stHeader"] {{ background:rgba(0,0,0,0); }}

.card {{
    background: linear-gradient(160deg,rgba(255,255,255,0.055),rgba(255,255,255,0.02));
    border: 1px solid {COLORS['border']};
    border-radius:14px;
    padding:0.95rem 1.1rem;
    backdrop-filter:blur(16px);
    box-shadow:0 12px 36px -20px rgba(0,0,0,0.7);
    transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    height:100%;
}}
.card:hover {{
    transform:translateY(-2px);
    border-color:rgba(255,255,255,0.18);
    box-shadow:0 18px 44px -22px rgba(34,211,238,0.20);
}}
.metric-label {{
    color:{COLORS['muted']};
    font-size:0.63rem;
    text-transform:uppercase;
    letter-spacing:0.12em;
    margin-bottom:0.45rem;
    font-weight:600;
    font-family:'DM Mono',monospace;
}}
.metric-value {{
    font-family:'Syne',sans-serif;
    font-size:1.85rem;
    font-weight:800;
    line-height:1;
    letter-spacing:-0.03em;
}}
.metric-sub {{
    color:{COLORS['muted']};
    font-size:0.65rem;
    margin-top:0.3rem;
    font-family:'DM Mono',monospace;
}}
.section-title {{
    font-size:0.88rem;
    font-weight:700;
    margin:0 0 0.75rem 0;
    display:flex;
    align-items:center;
    gap:0.5rem;
    font-family:'DM Sans',sans-serif;
}}
.section-title::before {{
    content:'';
    width:3px; height:14px;
    border-radius:999px;
    background:linear-gradient(180deg,{COLORS['cyan']},{COLORS['violet']});
    flex-shrink:0;
}}
.note {{
    background:linear-gradient(135deg,rgba(34,211,238,0.08),rgba(167,139,250,0.06));
    border:1px solid rgba(34,211,238,0.18);
    border-radius:12px;
    padding:0.88rem 1rem;
    line-height:1.6;
    font-size:0.82rem;
    backdrop-filter:blur(12px);
}}
.hero {{
    background:linear-gradient(135deg,rgba(34,211,238,0.08),rgba(167,139,250,0.07));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:16px;
    padding:0.88rem 1.15rem 0.75rem 1.15rem;
    margin-bottom:0.85rem;
    backdrop-filter:blur(16px);
}}
.hero-title {{
    font-family:'Syne',sans-serif;
    font-size:1.7rem;
    font-weight:800;
    letter-spacing:-0.04em;
    margin:0;
}}
.hero-sub {{
    color:{COLORS['muted']};
    font-size:0.76rem;
    margin-top:0.28rem;
    font-family:'DM Mono',monospace;
}}
div[data-baseweb="tab-list"] {{
    gap:0.3rem;
    background:rgba(255,255,255,0.03);
    padding:0.3rem;
    border-radius:11px;
    border:1px solid {COLORS['border']};
}}
button[data-baseweb="tab"] {{
    border-radius:9px !important;
    padding:0.42rem 0.85rem !important;
    color:{COLORS['muted']} !important;
    font-weight:600 !important;
    font-family:'DM Sans',sans-serif !important;
    font-size:0.82rem !important;
}}
button[aria-selected="true"] {{
    background:linear-gradient(135deg,rgba(34,211,238,0.16),rgba(167,139,250,0.14)) !important;
    color:{COLORS['text']} !important;
}}
.stDataFrame {{ border:1px solid {COLORS['border']}; border-radius:12px; overflow:hidden; }}
.stButton>button, .stDownloadButton>button {{
    border-radius:9px !important;
    border:1px solid {COLORS['border']} !important;
    background:rgba(255,255,255,0.04) !important;
    color:{COLORS['text']} !important;
    font-weight:600 !important;
    font-family:'DM Sans',sans-serif !important;
}}
hr {{ border-color:rgba(255,255,255,0.07) !important; margin:0.8rem 0 !important; }}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================
def resolve_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for candidate in [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "data", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    return filename


@st.cache_data
def load_data() -> Dict[str, pd.DataFrame]:
    cb = pd.read_csv(resolve_path(DATA_FILES["catboost"]), parse_dates=["date"])
    abl = pd.read_csv(resolve_path(DATA_FILES["ablation"]), parse_dates=["date"])
    gar = pd.read_csv(resolve_path(DATA_FILES["garch"]), parse_dates=["date"])
    shp = pd.read_csv(resolve_path(DATA_FILES["shap"]), parse_dates=["date"])

    gar = gar.rename(columns={"proba": "garch_proba", "pred": "garch_pred"}, errors="ignore")

    for col in shp.columns:
        if col != "date":
            shp[col] = pd.to_numeric(shp[col], errors="coerce")

    return {
        k: v.sort_values("date").reset_index(drop=True)
        for k, v in [("catboost", cb), ("ablation", abl), ("garch", gar), ("shap", shp)]
    }


# ============================================================
# Helpers
# ============================================================
def pretty_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def regime_label(probability: float) -> Tuple[str, str]:
    if probability >= CLASSIFICATION_THRESHOLD:
        return "Stress", COLORS["danger"]
    if probability >= WARNING_THRESHOLD:
        return "Elevated", COLORS["warning"]
    return "Calm", COLORS["success"]


def safe_ap(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else 0.0


def safe_roc(y, p):
    return roc_auc_score(y, p) if y.nunique() > 1 else 0.0


def prec_at_k(y, p, frac):
    n = max(1, int(len(p) * frac))
    top = np.argsort(p.to_numpy())[-n:]
    return float(y.iloc[top].mean())


def compute_metrics(df, proba_col, pred_col) -> Metrics:
    y, p, d = df["y_true"], df[proba_col], df[pred_col]
    return Metrics(
        pr_auc=safe_ap(y, p),
        roc_auc=safe_roc(y, p),
        f1=f1_score(y, d, zero_division=0),
        recall=recall_score(y, d, zero_division=0),
        precision=precision_score(y, d, zero_division=0),
        prec_at_5=prec_at_k(y, p, 0.05),
        prec_at_10=prec_at_k(y, p, 0.10),
    )


def top_shap_drivers(shap_row: pd.Series, n: int = 5) -> pd.Series:
    row = shap_row.drop(labels=["date"], errors="ignore")
    row = pd.to_numeric(row, errors="coerce").dropna()
    return row.reindex(row.abs().sort_values(ascending=False).head(n).index)


def build_explanation(probability: float, shap_row: pd.Series, display_date: str) -> str:
    label, _ = regime_label(probability)
    drivers = top_shap_drivers(shap_row, n=6)
    elevating = [pretty_feature(i) for i, v in drivers.items() if v > 0][:3]
    stabilising = [pretty_feature(i) for i, v in drivers.items() if v < 0][:2]
    parts = [
        f"Signal for {display_date}: <b>{label}</b> regime · predicted stress probability <b>{probability:.1%}</b>."
    ]
    if elevating:
        parts.append(f"Stress-elevating drivers: {', '.join(elevating)}.")
    if stabilising:
        parts.append(f"Stabilising drivers: {', '.join(stabilising)}.")
    best = drivers.abs().idxmax() if not drivers.empty else None
    if best:
        parts.append(f"Most influential feature: <b>{pretty_feature(best)}</b>.")
    return " ".join(parts)


def metric_card(label: str, value: str, color: str | None = None, sub: str | None = None) -> str:
    style = f"color:{color};" if color else ""
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='{style}'>{value}</div>"
        f"{sub_html}</div>"
    )


def outcome_style(val: str) -> str:
    return {
        "Correct alert": "color:#34d399;font-weight:700;",
        "False alarm": "color:#fb7185;font-weight:700;",
        "Missed": "color:#fbbf24;font-weight:700;",
        "Correct negative": "color:#94a3b8;",
    }.get(val, "")


# ============================================================
# Chart builders
# ============================================================
def add_stress_windows(fig: go.Figure, df: pd.DataFrame) -> None:
    in_block, start = False, None
    dates, flags = df["date"].tolist(), df["y_true"].tolist()
    for i, (dt, f) in enumerate(zip(dates, flags)):
        if f == 1 and not in_block:
            start, in_block = dt, True
        elif f == 0 and in_block:
            fig.add_vrect(x0=start, x1=dates[i - 1], fillcolor="rgba(251,113,133,0.07)", line_width=0)
            in_block = False
    if in_block and start is not None:
        fig.add_vrect(x0=start, x1=dates[-1], fillcolor="rgba(251,113,133,0.07)", line_width=0)


def make_timeline(catboost_df, garch_df, highlight: pd.Timestamp | None = None) -> go.Figure:
    fig = go.Figure()
    add_stress_windows(fig, catboost_df)
    fig.add_trace(
        go.Scatter(
            x=catboost_df["date"],
            y=catboost_df["proba"],
            mode="lines",
            name="CatBoost",
            line=dict(color=COLORS["cyan"], width=2.2),
            hovertemplate="%{x|%d %b %Y}<br>CatBoost: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=garch_df["date"],
            y=garch_df["garch_proba"],
            mode="lines",
            name="GARCH",
            line=dict(color=COLORS["garch"], width=1.6, dash="dash"),
            hovertemplate="%{x|%d %b %Y}<br>GARCH: %{y:.3f}<extra></extra>",
        )
    )
    stress_pts = catboost_df[catboost_df["y_true"] == 1]
    fig.add_trace(
        go.Scatter(
            x=stress_pts["date"],
            y=stress_pts["proba"],
            mode="markers",
            name="Actual stress",
            marker=dict(color=COLORS["danger"], size=7, symbol="x"),
            hovertemplate="%{x|%d %b %Y}<br>Actual stress<extra></extra>",
        )
    )
    fig.add_hline(
        y=WARNING_THRESHOLD,
        line_dash="dot",
        line_color=COLORS["warning"],
        line_width=1,
        annotation_text=f"E",
        annotation_position="right",
        annotation_font=dict(color=COLORS["warning"], size=10),
    )
    fig.add_hline(
        y=CLASSIFICATION_THRESHOLD,
        line_dash="dash",
        line_color=COLORS["danger"],
        line_width=1,
        annotation_text=f"C",
        annotation_position="right",
        annotation_font=dict(color=COLORS["danger"], size=10),
    )
    if highlight is not None:
        fig.add_vline(x=highlight, line_dash="dash", line_width=1.8, line_color=COLORS["violet"])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        hovermode="x unified",
        yaxis=dict(title="Stress probability", range=[0, 1.05], **AXIS_STYLE),
        xaxis=dict(title="", **AXIS_STYLE),
    )
    return fig


def make_shap_bar(shap_row: pd.Series, top_n: int = 10) -> go.Figure:
    drivers = top_shap_drivers(shap_row, n=top_n)
    labels = [pretty_feature(x) for x in drivers.index]
    colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in drivers.values]
    fig = go.Figure(
        go.Bar(
            x=drivers.values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, opacity=0.85),
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(title="← stabilises stress  |  elevates stress →", **AXIS_STYLE),
        yaxis=dict(title="", autorange="reversed", **AXIS_STYLE),
    )
    return fig


def make_ablation_chart(comparison_df: pd.DataFrame) -> go.Figure:
    colors = [COLORS["success"] if x > 0 else COLORS["danger"] for x in comparison_df["Delta"]]
    fig = go.Figure(
        go.Bar(
            x=comparison_df["Metric"],
            y=comparison_df["Delta"],
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:+.4f}" for v in comparison_df["Delta"]],
            textposition="outside",
            hovertemplate="%{x}<br>Δ: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=270,
        yaxis=dict(title="Δ metric (Full minus No-Sentiment)", range=[-0.3, 0.3], **AXIS_STYLE),
        xaxis=dict(title="", **AXIS_STYLE),
    )
    return fig


# ============================================================
# Load data
# ============================================================
data = load_data()
catboost_all = data["catboost"]
ablation_all = data["ablation"]
garch_all = data["garch"]
shap_all = data["shap"]

all_dates = sorted(catboost_all["date"].dt.date.unique())

cat_metrics = compute_metrics(catboost_all, "proba", "pred")
garch_metrics = compute_metrics(garch_all, "garch_proba", "garch_pred")
abl_metrics = compute_metrics(ablation_all, "proba", "pred")


# ============================================================
# Header
# ============================================================
h_left, h_right = st.columns([3, 1])

with h_left:
    st.markdown(
        f"""
    <div class='hero'>
        <div class='hero-title'>{APP_TITLE}</div>
        <div class='hero-sub'>E-mini S&amp;P 500 · 2025 out-of-sample · H={HORIZON}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with h_right:
    st.markdown(
        "<div style='height:0.4rem'></div>"
        "<div style='font-size:0.63rem;text-transform:uppercase;letter-spacing:0.12em;"
        "color:#94a3b8;font-family:\"DM Mono\",monospace;margin-bottom:0.4rem;'>"
        "Selected date</div>",
        unsafe_allow_html=True,
    )
    if "selected_day_index" not in st.session_state:
        st.session_state["selected_day_index"] = len(all_dates) - 1

    selected_day = st.selectbox(
        "selected_date",
        options=all_dates,
        index=st.session_state["selected_day_index"],
        format_func=lambda d: d.strftime("%d %b %Y"),
        label_visibility="collapsed",
        key="global_date",
    )
    st.session_state["selected_day_index"] = all_dates.index(selected_day)

# Resolve selected-day data
cb_day = catboost_all[catboost_all["date"].dt.date == selected_day]
shap_day = shap_all[shap_all["date"].dt.date == selected_day]
garch_day = garch_all[garch_all["date"].dt.date == selected_day]

if not cb_day.empty:
    selected_cb = cb_day.iloc[0]
else:
    selected_cb = catboost_all.iloc[-1]

selected_prob = float(selected_cb["proba"])
selected_actual = int(selected_cb["y_true"])
selected_ts = pd.Timestamp(selected_day)
selected_day_str = selected_day.strftime("%d %b %Y")

predicted_regime, predicted_regime_color = regime_label(selected_prob)
actual_label, actual_label_color = (
    ("Stress", COLORS["danger"]) if selected_actual == 1 else ("No stress", COLORS["success"])
)

if not garch_day.empty:
    selected_garch_prob = float(garch_day.iloc[0]["garch_proba"])
    garch_regime, garch_regime_color = regime_label(selected_garch_prob)
    garch_sub = f"prob {selected_garch_prob:.3f}"
else:
    selected_garch_prob = None
    garch_regime, garch_regime_color = "N/A", COLORS["muted"]
    garch_sub = "—"

# Header cards
col1, col2, col3, col4 = st.columns(4)
col1.markdown(metric_card("Predicted state", predicted_regime, predicted_regime_color, sub=f"threshold {CLASSIFICATION_THRESHOLD:.2f}"), unsafe_allow_html=True)
col2.markdown(metric_card("Actual state", actual_label, actual_label_color, sub=selected_day_str), unsafe_allow_html=True)
col3.markdown(metric_card("CatBoost probability", f"{selected_prob:.3f}", predicted_regime_color, sub="selected date"), unsafe_allow_html=True)
col4.markdown(metric_card("GARCH state", garch_regime, garch_regime_color, sub=garch_sub), unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# Tabs
# ============================================================
tab_monitor, tab_alerts, tab_explain, tab_compare = st.tabs(["Monitor", "Alerts", "Explain", "Compare"])


with tab_monitor:
    st.markdown("<div class='section-title'>Probability timeline — full year</div>", unsafe_allow_html=True)
    st.plotly_chart(
        make_timeline(catboost_all, garch_all, highlight=selected_ts),
        use_container_width=True,
        config={"displayModeBar": False},
        key="monitor_timeline",
    )

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        st.markdown("<div class='section-title'>Daily monitoring summary</div>", unsafe_allow_html=True)
        summary = catboost_all[["date", "proba", "pred", "y_true"]].copy()
        summary["state"] = summary["proba"].apply(lambda x: regime_label(float(x))[0])
        summary["date"] = summary["date"].dt.strftime("%d %b %Y")
        summary["proba"] = summary["proba"].map(lambda x: f"{x:.3f}")
        summary.columns = ["Date", "Probability", "Predicted", "Actual", "State"]
        st.dataframe(summary.sort_values("Probability", ascending=False), use_container_width=True, hide_index=True, height=320)
        st.download_button(
            "Download full predictions CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"market_stress_predictions_h{HORIZON}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with mc2:
        n_stress = int(catboost_all["y_true"].sum())
        n_detected = int(((catboost_all["pred"] == 1) & (catboost_all["y_true"] == 1)).sum())
        n_false = int(((catboost_all["pred"] == 1) & (catboost_all["y_true"] == 0)).sum())
        st.markdown(metric_card("Stress events", str(n_stress), COLORS["danger"]), unsafe_allow_html=True)
        st.markdown(metric_card("Detected", str(n_detected), COLORS["success"]), unsafe_allow_html=True)
        st.markdown(metric_card("False alarms", str(n_false), COLORS["warning"]), unsafe_allow_html=True)
        st.markdown(
            f"<div class='note' style='margin-top:0.8rem;font-size:0.78rem;'>"
            f"<b>Reading.</b> Shaded regions = actual stress windows. Purple line = selected date. "
            f"E = elevated zone at {WARNING_THRESHOLD:.2f}. C = classification threshold at {CLASSIFICATION_THRESHOLD:.2f}. "
            f"Use Alerts for prioritisation and Explain for per-signal auditability.</div>",
            unsafe_allow_html=True,
        )


with tab_alerts:
    st.markdown("<div class='section-title'>High-priority alert queue</div>", unsafe_allow_html=True)
    top_k_pct = st.radio(
        "Alert bucket — top K% by predicted probability",
        options=[5, 10, 15, 20],
        index=1,
        horizontal=True,
        key="top_k_radio",
    )
    st.caption(
        f"Top {top_k_pct}% of trading days ranked by CatBoost stress probability. Outcome is colour-coded: green = correct alert, red = false alarm, amber = missed."
    )

    top_n = max(1, int(len(catboost_all) * (top_k_pct / 100)))
    alert_df = catboost_all.nlargest(top_n, "proba").copy()

    shap_idx = shap_all.set_index("date")
    d1_list, d2_list, outcome_list = [], [], []

    for _, row in alert_df.iterrows():
        if row["pred"] == 1 and row["y_true"] == 1:
            outcome_list.append("Correct alert")
        elif row["pred"] == 1 and row["y_true"] == 0:
            outcome_list.append("False alarm")
        elif row["pred"] == 0 and row["y_true"] == 1:
            outcome_list.append("Missed")
        else:
            outcome_list.append("Correct negative")

        if row["date"] in shap_idx.index:
            drivers = top_shap_drivers(shap_idx.loc[row["date"]], n=2)
            names = [pretty_feature(x) for x in drivers.index]
            d1_list.append(names[0] if names else "—")
            d2_list.append(names[1] if len(names) > 1 else "—")
        else:
            d1_list.append("—")
            d2_list.append("—")

    alert_df["Top driver 1"] = d1_list
    alert_df["Top driver 2"] = d2_list
    alert_df["Outcome"] = outcome_list
    alert_df["Suggested action"] = alert_df["proba"].apply(
        lambda x: "Immediate review" if x >= CLASSIFICATION_THRESHOLD else "Watch closely" if x >= WARNING_THRESHOLD else "Monitor"
    )

    disp = alert_df[["date", "proba", "Top driver 1", "Top driver 2", "Outcome", "Suggested action"]].copy()
    disp["date"] = disp["date"].dt.strftime("%d %b %Y")
    disp["proba"] = disp["proba"].round(4)
    disp.columns = ["Date", "Probability", "Top driver 1", "Top driver 2", "Outcome", "Suggested action"]

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    st.download_button(
        "Download alert queue as CSV",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name=f"market_stress_alerts_h{HORIZON}_top{top_k_pct}pct.csv",
        mime="text/csv",
        use_container_width=True,
    )


with tab_explain:
    st.markdown("<div class='section-title'>Per-signal SHAP explanation</div>", unsafe_allow_html=True)
    st.caption(
        f"Showing {selected_day_str} — change the date using the selector in the header. Red bars = features that raise stress probability. Green bars = features that lower it."
    )

    if cb_day.empty or shap_day.empty:
        st.warning("No SHAP data available for the selected date. Try a different date.")
    else:
        exp_row = cb_day.iloc[0]
        shap_row = shap_day.iloc[0]
        exp_prob = float(exp_row["proba"])

        ex_l, ex_r = st.columns([1, 2])
        with ex_l:
            st.markdown(f"<div class='note'>{build_explanation(exp_prob, shap_row, selected_day_str)}</div>", unsafe_allow_html=True)
        with ex_r:
            st.plotly_chart(
                make_shap_bar(shap_row, top_n=10),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"shap_{selected_day}",
            )

        st.markdown("<div class='section-title' style='margin-top:0.9rem;'>Context in full year</div>", unsafe_allow_html=True)
        st.plotly_chart(
            make_timeline(catboost_all, garch_all, highlight=selected_ts),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"explain_tl_{selected_day}",
        )


with tab_compare:
    st.markdown("<div class='section-title'>Full-year model comparison</div>", unsafe_allow_html=True)

    mt = pd.DataFrame(
        {
            "Metric": ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision", "Prec@5%", "Prec@10%"],
            "CatBoost": [cat_metrics.pr_auc, cat_metrics.roc_auc, cat_metrics.f1, cat_metrics.recall, cat_metrics.precision, cat_metrics.prec_at_5, cat_metrics.prec_at_10],
            "GARCH": [garch_metrics.pr_auc, garch_metrics.roc_auc, garch_metrics.f1, garch_metrics.recall, garch_metrics.precision, garch_metrics.prec_at_5, garch_metrics.prec_at_10],
            "No sentiment": [abl_metrics.pr_auc, abl_metrics.roc_auc, abl_metrics.f1, abl_metrics.recall, abl_metrics.precision, abl_metrics.prec_at_5, abl_metrics.prec_at_10],
        }
    )
    mt[["CatBoost", "GARCH", "No sentiment"]] = mt[["CatBoost", "GARCH", "No sentiment"]].round(4)

    def highlight_best(row):
        best = max(row["CatBoost"], row["GARCH"], row["No sentiment"])
        return [
            "",
            *[
                f"color:{COLORS['cyan']};font-weight:700;" if row[c] == best else ""
                for c in ["CatBoost", "GARCH", "No sentiment"]
            ],
        ]

    st.dataframe(mt.style.apply(highlight_best, axis=1).hide(axis="index"), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title' style='margin-top:1rem;'>Sentiment ablation delta</div>", unsafe_allow_html=True)
    st.caption("Positive = full model outperforms no-sentiment model. Negative = removing sentiment helps.")

    ac = pd.DataFrame(
        {
            "Metric": ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision", "Prec@5%", "Prec@10%"],
            "Full": [cat_metrics.pr_auc, cat_metrics.roc_auc, cat_metrics.f1, cat_metrics.recall, cat_metrics.precision, cat_metrics.prec_at_5, cat_metrics.prec_at_10],
            "No sentiment": [abl_metrics.pr_auc, abl_metrics.roc_auc, abl_metrics.f1, abl_metrics.recall, abl_metrics.precision, abl_metrics.prec_at_5, abl_metrics.prec_at_10],
        }
    )
    ac["Delta"] = ac["Full"] - ac["No sentiment"]

    st.plotly_chart(
        make_ablation_chart(ac),
        use_container_width=True,
        config={"displayModeBar": False},
        key="ablation_chart",
    )

    st.markdown(
        f"<div class='note'><b>Interpretation.</b> The dashboard focuses on H={HORIZON} and uses the out-of-sample CatBoost decision threshold of {CLASSIFICATION_THRESHOLD:.2f}. It supports three tasks: monitoring the current signal (Monitor), prioritising the highest-risk dates (Alerts), and explaining per-prediction SHAP drivers (Explain). The ablation view supports the report's conclusion that daily news sentiment adds no reliable incremental value once VIX and credit spread features are already present.</div>",
        unsafe_allow_html=True,
    )
