# =============================================================================
# PHASE 5 — STREAMLIT DASHBOARD (v3)
# Macro-Market Regime Detection System (US)
# =============================================================================

import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Regime Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
NAVY         = "#0D1B2A"
BORDER       = "#E2E8F0"
TEXT_PRIMARY = "#0D1B2A"
TEXT_MUTED   = "#64748B"

REGIME_COLORS = {
    "Risk-On":          "#059669",
    "Risk-Off":         "#DC2626",
    "High-Volatility":  "#EA580C",
    "Inflation-Driven": "#7C3AED",
}
REGIME_BG = {
    "Risk-On":          "#ECFDF5",
    "Risk-Off":         "#FEF2F2",
    "High-Volatility":  "#FFF7ED",
    "Inflation-Driven": "#F5F3FF",
}
REGIME_ICONS = {
    "Risk-On":          "↑",
    "Risk-Off":         "↓",
    "High-Volatility":  "⚡",
    "Inflation-Driven": "🔥",
}
REGIME_INTROS = {
    "Risk-On":          "Conditions favour risk assets. Volatility is contained, credit is accessible, and growth indicators are healthy.",
    "Risk-Off":         "Markets are in a cautious, slow-growth posture. Appetite for risk is subdued and credit conditions are tightening.",
    "High-Volatility":  "Stress conditions detected. Volatility is elevated, credit spreads are wide, and growth indicators are contracting.",
    "Inflation-Driven": "Inflation is the dominant macro force. Price pressures are elevated, shaping both policy and asset class dynamics.",
}
PORTFOLIO_TILTS = {
    "Risk-On":          "Equities, cyclicals, credit",
    "Risk-Off":         "Quality bonds, short duration, defensives",
    "High-Volatility":  "Cash, tail hedges, safe havens",
    "Inflation-Driven": "TIPS, commodities, real assets",
}

FEATURE_DISPLAY = [
    ("S&P 500 Daily Return",          "Equity direction signal"),
    ("Realized Vol (20d)",            "Short-term market volatility"),
    ("Realized Vol (60d)",            "Medium-term market volatility"),
    ("VIX Level",                     "Implied volatility / fear gauge"),
    ("VIX Change (20d)",              "Momentum in fear"),
    ("S&P 500 Drawdown",              "Distance from rolling peak"),
    ("GARCH Conditional Vol",         "Model-based volatility estimate"),
    ("Yield Curve Slope (10Y–2Y)",    "Recession signal / steepness"),
    ("Fed Funds Rate",                "Monetary policy level"),
    ("Fed Funds Change (60d)",        "Hiking or cutting cycle"),
    ("Credit Spread (Baa–10Y)",       "Corporate risk appetite"),
    ("Credit Spread Change (20d)",    "Momentum in credit conditions"),
    ("5Y Breakeven Inflation",        "Market inflation expectations"),
    ("CPI Inflation YoY",             "Actual inflation rate"),
    ("Industrial Production YoY",     "Real economic growth (lagging)"),
    ("Philly Fed Manufacturing",      "Growth sentiment (leading, PMI proxy)"),
]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important;
    max-width: 1400px !important; }

/* ── Header: hide only the decorative Streamlit branding, keep Deploy/Run buttons ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Do NOT hide header — that removes the Deploy/Run/Share buttons */

/* ── Sidebar background via attribute selector (works across Streamlit versions) ── */
[data-testid="stSidebar"] > div:first-child {
    background-color: #0B1526;
    padding-top: 1rem;
}
/* Text colours inside sidebar — target rendered markdown and widgets specifically */
[data-testid="stSidebar"] .stMarkdown p    { color: #475569 !important; font-size: 9px !important;
    text-transform: uppercase; letter-spacing: 0.10em; margin: 0 !important; }
[data-testid="stSidebar"] .stMarkdown span { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stMarkdown div  { color: #CBD5E1 !important; }
[data-testid="stSidebar"] small            { color: #94A3B8 !important; }
/* Expander inside sidebar */
[data-testid="stSidebar"] details {
    border: 1px solid #1E293B !important; border-radius: 6px !important;
    background: transparent !important; }
[data-testid="stSidebar"] summary { color: #64748B !important; font-size: 11px !important; }
[data-testid="stSidebar"] details p,
[data-testid="stSidebar"] details div { color: #CBD5E1 !important; }

/* ── Tabs ── */
/* Tab bar background */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #F1F5F9 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid #E2E8F0 !important;
    margin-bottom: 20px !important;
}
/* Every tab button — force colour with high specificity via :not trick */
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 7px !important;
    padding: 8px 20px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
}
/* Active tab — white pill with dark text */
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
}
/* Hide BaseWeb underline decorations */
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* ── Section labels ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif; font-size: 9px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; color: #94A3B8;
    margin-bottom: 8px; margin-top: 2px;
}

/* ── Cards ── */
.regime-card  { border-radius: 14px; padding: 22px 26px; margin-bottom: 16px;
    border: 1.5px solid; }
.stat-card    { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 8px; }
.explanation-box { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 16px 20px; margin-top: 14px; }
.info-box { background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 8px;
    padding: 11px 15px; font-size: 12px; color: #0369A1; margin-bottom: 16px; }

/* ── Prob bars ── */
.prob-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.prob-label { font-family: 'Space Grotesk', sans-serif; font-size: 11px; font-weight: 600;
    width: 130px; flex-shrink: 0; color: #374151; }
.prob-bar-bg { flex: 1; height: 5px; background: #E2E8F0; border-radius: 3px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 3px; }
.prob-pct  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: #64748B; width: 34px; text-align: right; flex-shrink: 0; }

/* ── Pulse dot ── */
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,255,255,0.5); }
    70%  { box-shadow: 0 0 0 6px rgba(255,255,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}
.live-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    animation: pulse 2s infinite; vertical-align: middle; margin-right: 5px; }

/* ── Misc widgets ── */
[data-testid="stExpander"] { border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important; }
[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 8px; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem !important; font-weight: 500 !important; color: #0F172A !important; }
[data-testid="stMetricLabel"] { font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.65rem !important; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA + MODEL LOADING
# =============================================================================
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"

@st.cache_data
def load_data():
    regime_df   = pd.read_parquet(f"{PROCESSED_DIR}/us_regime_labels.parquet")
    feat_raw    = pd.read_parquet(f"{PROCESSED_DIR}/us_features_daily.parquet")
    feat_raw    = feat_raw[feat_raw.index >= "2003-01-01"].dropna()
    feat_raw    = feat_raw.reindex(regime_df.index)
    raw_prices  = pd.read_parquet(f"{PROCESSED_DIR}/us_combined_daily.parquet")
    raw_prices  = raw_prices[raw_prices.index >= "2003-01-01"]
    stats_df    = pd.read_parquet(f"{PROCESSED_DIR}/us_regime_stats.parquet")
    return regime_df, feat_raw, raw_prices, stats_df

@st.cache_resource
def load_models():
    def pkl(n):
        with open(f"{MODEL_DIR}/{n}.pkl", "rb") as f: return pickle.load(f)
    return {k: pkl(v) for k, v in {
        "hmm":"hmm_model","pca":"pca_model","state_labels":"state_labels",
        "state_means":"state_means_df","feat_meta":"feature_meta",
        "feat_mean":"feat_mean","feat_std":"feat_std",
    }.items()}

regime_df, feat_raw, raw_prices, stats_df = load_data()
M = load_models()
STATE_LABELS   = M["state_labels"]
FEATURE_META   = M["feat_meta"]
feat_mean      = M["feat_mean"]
feat_std       = M["feat_std"]
state_means_df = M["state_means"]

# =============================================================================
# HELPERS
# =============================================================================
def rc(r):  return REGIME_COLORS.get(r, "#888")
def rbg(r): return REGIME_BG.get(r, "#F8FAFC")

def score_contributions(obs, regime_label):
    sid = {v: k for k, v in STATE_LABELS.items()}[regime_label]
    rm  = state_means_df.loc[f"State {sid}"]
    s   = {}
    for col in feat_raw.columns:
        if col not in obs.index or pd.isna(obs[col]): continue
        zt = (obs[col] - feat_mean[col]) / (feat_std[col] + 1e-9)
        zr = (rm[col]  - feat_mean[col]) / (feat_std[col] + 1e-9)
        s[col] = zt * np.sign(zr) if abs(zr) > 0.1 else 0.0
    return pd.Series(s).sort_values(key=abs, ascending=False)

def zdesc(col, val):
    z = (val - feat_mean[col]) / (feat_std[col] + 1e-9)
    if z >  1.5: return "well above historical avg"
    if z >  0.5: return "above historical avg"
    if z < -1.5: return "well below historical avg"
    if z < -0.5: return "below historical avg"
    return "near historical avg"

def zdesc_short(col, val):
    z = (val - feat_mean[col]) / (feat_std[col] + 1e-9)
    if z >  1.5: return "well above avg"
    if z >  0.5: return "above avg"
    if z < -1.5: return "well below avg"
    if z < -0.5: return "below avg"
    return "near avg"

def generate_explanation(obs, regime_label, confidence, top_n=3):
    contribs     = score_contributions(obs, regime_label)
    top_features = contribs.head(top_n).index.tolist()
    intro        = REGIME_INTROS[regime_label]
    conf_str     = f"Model confidence: {confidence:.0%}."
    phrases = []
    for c in top_features:
        label = FEATURE_META.get(c, {"label": c})["label"]
        phrases.append(f"{label} ({zdesc(c, obs[c])})")
    if len(phrases) == 1:
        driven = f"Primarily driven by {phrases[0]}."
    elif len(phrases) == 2:
        driven = f"Driven by {phrases[0]} and {phrases[1]}."
    else:
        driven = f"Driven by {phrases[0]}, {phrases[1]}, and {phrases[2]}."
    tilt = PORTFOLIO_TILTS[regime_label]
    return intro, conf_str, driven, tilt

def plotly_base(fig, height=400, hovermode="x unified"):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=24, b=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=11, color=TEXT_PRIMARY),
        hovermode=hovermode,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#E2E8F0",
                     linewidth=1, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                     tickfont=dict(size=9))
    return fig

# =============================================================================
# DERIVED CURRENT VALUES
# =============================================================================
latest        = regime_df.index[-1]
latest_regime = str(regime_df.loc[latest, "regime"]).strip()
latest_conf   = float(np.asarray(
    pd.to_numeric(regime_df.loc[latest, "confidence"], errors="coerce")
).reshape(()).item())
color  = rc(latest_regime)
icon   = REGIME_ICONS[latest_regime]
days_in_regime = int((regime_df["regime"] == latest_regime).iloc[::-1].cumprod().sum())

# Pre-compute explanation
latest_obs                     = feat_raw.loc[latest]
intro_txt, conf_txt, driven_txt, tilt_txt = generate_explanation(
    latest_obs, latest_regime, latest_conf
)
contribs_latest = score_contributions(latest_obs, latest_regime)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    # Logo / title
    st.markdown(
        "<div style='padding:4px 0 20px'>"
        "<div style='font-family:Space Grotesk;font-size:15px;font-weight:700;"
        "color:#F1F5F9;letter-spacing:-0.01em'>📡 Macro Regime Monitor</div>"
        "<div style='font-size:9px;color:#334155;letter-spacing:0.09em;"
        "text-transform:uppercase;margin-top:2px'>US Market · 2003–present</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Live regime block
    st.markdown(
        f"<div style='background:{color}1A;border:1px solid {color}33;"
        f"border-radius:10px;padding:14px 16px;margin-bottom:16px'>"
        f"<div style='font-size:8px;color:{color};font-family:Space Grotesk;"
        f"font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
        f"<span class='live-dot' style='background:{color}'></span>Current Regime</div>"
        f"<div style='font-size:19px;font-weight:700;color:#F1F5F9;"
        f"font-family:Space Grotesk;letter-spacing:-0.01em;line-height:1.1'>"
        f"{icon} {latest_regime}</div>"
        f"<div style='display:flex;justify-content:space-between;margin-top:10px;"
        f"padding-top:8px;border-top:1px solid {color}22'>"
        f"<div><div style='font-family:JetBrains Mono;font-size:14px;color:{color};"
        f"font-weight:500'>{latest_conf:.0%}</div>"
        f"<div style='font-size:8px;color:#475569;text-transform:uppercase;"
        f"letter-spacing:0.08em'>confidence</div></div>"
        f"<div style='text-align:right'>"
        f"<div style='font-family:JetBrains Mono;font-size:14px;color:#94A3B8'>"
        f"{days_in_regime}d</div>"
        f"<div style='font-size:8px;color:#475569;text-transform:uppercase;"
        f"letter-spacing:0.08em'>in regime</div></div>"
        f"</div></div>",
        unsafe_allow_html=True
    )

    # Regime overview
    st.markdown("<p>Regime breakdown</p>", unsafe_allow_html=True)
    total_days = len(regime_df)
    for reg in ["Risk-On", "Risk-Off", "Inflation-Driven", "High-Volatility"]:
        n   = (regime_df["regime"] == reg).sum()
        pct = n / total_days
        rclr = rc(reg)
        active = " ← now" if reg == latest_regime else ""
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:space-between;"
            f"padding:5px 0;border-bottom:1px solid #1E293B'>"
            f"<div style='font-size:10px;color:#94A3B8;font-family:Inter;display:flex;"
            f"align-items:center;gap:6px'>"
            f"<span style='width:6px;height:6px;border-radius:50%;background:{rclr};"
            f"flex-shrink:0;display:inline-block'></span>{reg}"
            f"<span style='font-size:8px;color:{rclr}'>{active}</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono;font-size:10px;color:#64748B'>"
            f"{pct:.0%}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model spec
    st.markdown("<p>Model specification</p>", unsafe_allow_html=True)
    specs = [
        ("Method",       "Gaussian HMM"),
        ("States",       "4"),
        ("PCA variance", "74.3%"),
        ("K-means agr.", "80.3%"),
        ("Avg conf.",    "99.2%"),
        ("Obs. period",  f"2003–{latest.year}"),
    ]
    for lbl, val in specs:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:4px 0;border-bottom:1px solid #0F172A'>"
            f"<div style='font-size:9px;color:#475569'>{lbl}</div>"
            f"<div style='font-family:JetBrains Mono;font-size:9px;color:#64748B'>"
            f"{val}</div></div>",
            unsafe_allow_html=True
        )

    # Features expandable
    with st.expander(f"  Features  ·  16"):
        for name, desc in FEATURE_DISPLAY:
            st.markdown(
                f"<div style='padding:5px 0;border-bottom:1px solid #1E293B'>"
                f"<div style='font-size:10px;color:#CBD5E1;font-family:Space Grotesk;"
                f"font-weight:500'>{name}</div>"
                f"<div style='font-size:9px;color:#475569;margin-top:1px'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown(
        "<div style='margin-top:20px;font-size:8px;color:#1E293B;line-height:1.6'>"
        "Data: Yahoo Finance · FRED<br>"
        "⚠️ Research only — not investment advice."
        "</div>",
        unsafe_allow_html=True
    )

# =============================================================================
# PAGE HEADER
# =============================================================================
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(
        "<div style='padding:4px 0 2px;background:##0F172A'>"
        "<div style='font-family:Space Grotesk,sans-serif;font-size:24px;font-weight:700;"
        "letter-spacing:-0.03em;color:#0F172A !important;line-height:1.15'>"
        "<b style='color:#FFFFFF'>Macro-Market Regime Detection</b></div>"
        "<div style='font-size:12px;color:#64748B;margin-top:4px;font-family:Inter,sans-serif'>"
        "US equities · 2003–present · Gaussian HMM + PCA · 16 macro-financial features"
        "</div></div>",
        unsafe_allow_html=True
    )
with h_right:
    st.markdown(
        f"<div style='text-align:right'>"
        f"<div style='font-size:8px;color:#94A3B8;font-family:Space Grotesk;"
        f"text-transform:uppercase;letter-spacing:0.10em'>Last observation</div>"
        f"<div style='font-family:JetBrains Mono;font-size:13px;font-weight:500;"
        f"color:#0D1B2A;margin-top:2px'>{latest.strftime('%b %d, %Y')}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<hr style='border:none;border-top:1px solid #E2E8F0;margin:10px 0 4px'>",
    unsafe_allow_html=True
)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Timeline",
    "🔍  Current Regime",
    "📋  Statistics",
    "⚖️  Backtest",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    ctrl1, ctrl2, ctrl3 = st.columns([4, 1, 1])
    with ctrl1:
        year_range = st.slider(
            "Date range", 2003, latest.year, (2003, latest.year),
            key="yr", label_visibility="collapsed"
        )
    with ctrl2:
        show_vix  = st.checkbox("VIX", True)
    with ctrl3:
        show_conf = st.checkbox("Confidence", True)

    start_dt = pd.Timestamp(f"{year_range[0]}-01-01")
    end_dt   = pd.Timestamp(f"{year_range[1]}-12-31")
    mask     = (regime_df.index >= start_dt) & (regime_df.index <= end_dt)
    rdf_s    = regime_df[mask]
    sp500    = raw_prices["SP500"].reindex(rdf_s.index)
    vix_s    = raw_prices["VIX"].reindex(rdf_s.index)

    n_rows  = 1 + int(show_vix) + int(show_conf)
    h_ratios = [0.60] + ([0.23] if show_vix else []) + ([0.17] if show_conf else [])
    h_norm   = [h / sum(h_ratios) for h in h_ratios]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=h_norm, vertical_spacing=0.025)

    # Regime background bands
    seen = set()
    prev_r = None; b_start = None
    for date, row in rdf_s.iterrows():
        r = row["regime"]
        if r != prev_r:
            if prev_r:
                first = prev_r not in seen
                if first: seen.add(prev_r)
                fig.add_vrect(x0=b_start, x1=date, fillcolor=rc(prev_r),
                              opacity=0.12, line_width=0,
                              legendgroup=prev_r, name=prev_r, showlegend=first)
            prev_r = r; b_start = date
    if prev_r:
        first = prev_r not in seen
        fig.add_vrect(x0=b_start, x1=rdf_s.index[-1], fillcolor=rc(prev_r),
                      opacity=0.12, line_width=0, legendgroup=prev_r,
                      name=prev_r, showlegend=first)

    # S&P 500
    fig.add_trace(go.Scatter(x=rdf_s.index, y=sp500, name="S&P 500",
                             line=dict(color="#0D1B2A", width=1.5),
                             showlegend=False), row=1, col=1)

    cur_row = 2
    if show_vix:
        fig.add_trace(go.Scatter(x=rdf_s.index, y=vix_s, name="VIX",
                                 line=dict(color="#DC2626", width=1),
                                 fill="tozeroy",
                                 fillcolor="rgba(220,38,38,0.05)",
                                 showlegend=False), row=cur_row, col=1)
        fig.add_shape(type="line",
                      x0=rdf_s.index[0], x1=rdf_s.index[-1], y0=20, y1=20,
                      line=dict(dash="dot", color="#CBD5E1", width=1),
                      row=cur_row, col=1)  # type: ignore[arg-type]
        fig.update_yaxes(title_text="VIX", title_font_size=9,
                         row=cur_row, col=1)
        cur_row += 1

    if show_conf:
        fig.add_trace(go.Scatter(x=rdf_s.index, y=rdf_s["confidence"],
                                 fill="tozeroy",
                                 line=dict(color="#2563EB", width=0.8),
                                 fillcolor="rgba(37,99,235,0.07)",
                                 name="Confidence", showlegend=False),
                      row=cur_row, col=1)
        fig.add_shape(type="line",
                      x0=rdf_s.index[0], x1=rdf_s.index[-1], y0=0.6, y1=0.6,
                      line=dict(dash="dot", color="#CBD5E1", width=1),
                      row=cur_row, col=1)  # type: ignore[arg-type]
        fig.update_yaxes(title_text="Conf.", title_font_size=9,
                         row=cur_row, col=1, range=[0, 1.05])

    fig.update_yaxes(title_text="S&P 500", title_font_size=9, row=1, col=1)
    plotly_base(fig, height=520 if n_rows == 3 else 420)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                  y=1.03, xanchor="left", x=0, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    # Legend pills
    cols_leg = st.columns(4)
    for i, (reg, clr) in enumerate(REGIME_COLORS.items()):
        with cols_leg[i]:
            pct  = (regime_df["regime"] == reg).mean()
            n_ep = int((regime_df["regime"] != regime_df["regime"].shift()).cumsum()
                       [regime_df["regime"] == reg].nunique())
            st.markdown(
                f"<div style='padding:10px 14px;background:{rbg(reg)};"
                f"border:1px solid {clr}30;border-radius:10px;"
                f"{'border-left:3px solid ' + clr + ';' if reg == latest_regime else ''}'>"
                f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>"
                f"<span style='width:7px;height:7px;border-radius:50%;background:{clr};"
                f"flex-shrink:0;display:inline-block'></span>"
                f"<span style='font-family:Space Grotesk;font-size:12px;"
                f"font-weight:600;color:{clr}'>{reg}</span>"
                f"{'<span style=\"font-size:9px;color:' + clr + ';margin-left:auto\">● now</span>' if reg == latest_regime else ''}"
                f"</div>"
                f"<div style='display:flex;gap:12px'>"
                f"<div><div style='font-family:JetBrains Mono;font-size:13px;"
                f"font-weight:500;color:#374151'>{pct:.0%}</div>"
                f"<div style='font-size:8px;color:#94A3B8;text-transform:uppercase;"
                f"letter-spacing:0.06em'>of history</div></div>"
                f"<div><div style='font-family:JetBrains Mono;font-size:13px;"
                f"font-weight:500;color:#374151'>{n_ep}</div>"
                f"<div style='font-size:8px;color:#94A3B8;text-transform:uppercase;"
                f"letter-spacing:0.06em'>episodes</div></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CURRENT REGIME
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    left, right = st.columns([11, 10], gap="large")

    with left:
        # ── Regime card ──
        st.markdown(
            f"<div class='regime-card' style='background:{rbg(latest_regime)};"
            f"border-color:{color}40'>"

            f"<div style='display:flex;align-items:flex-start;"
            f"justify-content:space-between;gap:12px'>"

            f"<div>"
            f"<div style='font-size:8px;font-family:Space Grotesk;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.12em;color:{color};"
            f"margin-bottom:6px'>Current Regime</div>"
            f"<div style='font-family:Space Grotesk;font-size:28px;font-weight:700;"
            f"color:{color};letter-spacing:-0.02em;line-height:1.05'>"
            f"{icon} {latest_regime}</div>"
            f"<div style='font-size:12px;color:#6B7280;margin-top:6px;font-family:Inter'>"
            f"{intro_txt}</div>"
            f"</div>"

            f"<div style='text-align:right;flex-shrink:0'>"
            f"<div style='font-family:JetBrains Mono;font-size:26px;font-weight:500;"
            f"color:{color};line-height:1'>{latest_conf:.0%}</div>"
            f"<div style='font-size:8px;color:#94A3B8;font-family:Space Grotesk;"
            f"text-transform:uppercase;letter-spacing:0.09em;margin-top:2px'>"
            f"confidence</div>"
            f"<div style='font-family:JetBrains Mono;font-size:11px;color:#94A3B8;"
            f"margin-top:6px'>{days_in_regime} days</div>"
            f"<div style='font-size:8px;color:#94A3B8'>in this regime</div>"
            f"</div></div>"

            f"<hr style='border:none;border-top:1px solid {color}20;margin:14px 0 10px'>"

            f"<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap'>"
            f"<div style='font-size:8px;font-family:Space Grotesk;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.10em;color:#94A3B8'>Portfolio tilt</div>"
            f"<div style='font-size:12px;color:{color};font-weight:600;font-family:Inter'>"
            f"{tilt_txt}</div>"
            f"<div style='margin-left:auto;font-size:9px;color:#94A3B8'>"
            f"Regime-based · not investment advice</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

        # ── Phase 4 explanation ──
        st.markdown("<div class='section-label'>Model Explanation</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='explanation-box'>"
            f"<div style='font-size:13px;color:#374151;line-height:1.75'>"
            f"{REGIME_INTROS[latest_regime]}"
            f"</div>"
            f"<div style='margin-top:10px;font-size:12px;color:#64748B;"
            f"font-family:Inter;line-height:1.7'>"
            f"<span style='font-weight:600;color:#374151'>{conf_txt}</span>"
            f"&nbsp; {driven_txt}"
            f"</div>"
            f"<div style='margin-top:10px;padding:8px 12px;"
            f"background:{color}0D;border-radius:6px;border-left:3px solid {color};"
            f"font-size:11px;color:#64748B'>"
            f"<b style='color:{color}'>Suggested tilt:</b> {tilt_txt}. "
            f"<span style='color:#94A3B8'>Regime-based signal only — not investment advice.</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

        # ── State probabilities ──
        st.markdown("<div class='section-label' style='margin-top:18px'>State Probabilities</div>",
                    unsafe_allow_html=True)
        probs_raw = M["hmm"].predict_proba(
            M["pca"].transform(
                ((latest_obs - feat_mean) / (feat_std + 1e-9)).values.reshape(1, -1)
            )
        )[0]
        prob_dict = {STATE_LABELS[i]: float(p) for i, p in enumerate(probs_raw)}
        for reg, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
            rclr = rc(reg)
            st.markdown(
                f"<div class='prob-row'>"
                f"<div class='prob-label' style='color:{rclr}'>{reg}</div>"
                f"<div class='prob-bar-bg'><div class='prob-bar-fill' "
                f"style='width:{prob*100:.1f}%;background:{rclr}'></div></div>"
                f"<div class='prob-pct'>{prob:.0%}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown(
            "<div style='font-size:9px;color:#94A3B8;margin-top:4px;line-height:1.5'>"
            "⚠ Single forward pass — may differ from the Viterbi-decoded label above "
            "which uses full historical sequence context."
            "</div>",
            unsafe_allow_html=True
        )

    with right:
        # ── Feature contribution chart ──
        st.markdown("<div class='section-label'>Feature Contributions</div>",
                    unsafe_allow_html=True)
        top10  = contribs_latest.head(10)
        labels = [FEATURE_META.get(c, {"label": c})["label"] for c in top10.index]
        vals   = top10.values
        bclrs  = [color if v > 0 else "#D1D5DB" for v in vals]

        fig2 = go.Figure(go.Bar(
            x=vals[::-1], y=labels[::-1], orientation="h",
            marker=dict(color=bclrs[::-1], line=dict(width=0)),
            text=[f"{v:+.2f}" for v in vals[::-1]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=9, color="#94A3B8"),
        ))
        fig2.add_vline(x=0, line_color="#E2E8F0", line_width=1.5)
        plotly_base(fig2, height=320)
        fig2.update_layout(margin=dict(l=0, r=48, t=8, b=0))
        fig2.update_xaxes(title_text="Contribution score", title_font_size=9,
                          showgrid=True, gridcolor="#F1F5F9", zeroline=False)
        fig2.update_yaxes(showgrid=False, tickfont=dict(size=9))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "<div style='font-size:9px;color:#94A3B8;line-height:1.6;margin-top:-8px'>"
            "Colored bars: features elevating in line with this regime's profile. "
            "Grey bars: working against it."
            "</div>",
            unsafe_allow_html=True
        )

        # ── Top 3 driver cards ──
        st.markdown("<div class='section-label' style='margin-top:16px'>Top Drivers</div>",
                    unsafe_allow_html=True)
        for col, score in contribs_latest.head(3).items():
            val   = latest_obs[col]
            meta  = FEATURE_META.get(col, {"label": col, "unit": "", "scale": 1})
            disp  = val * meta["scale"]
            unit  = meta["unit"]
            d     = zdesc_short(col, val)
            sv    = float(score)
            # Accent color for border/value; label text is always dark for readability
            clr2  = color if sv > 0 else "#64748B"
            st.markdown(
                f"<div style='padding:11px 14px;border-radius:8px;"
                f"border:1px solid {clr2}55;margin-bottom:6px;"
                f"background:#F8FAFC;"              # solid light background — always readable
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<div style='font-size:12px;font-weight:600;color:#1E293B;"  # always dark
                f"font-family:Space Grotesk'>{meta['label']}</div>"
                f"<div style='font-size:10px;color:#64748B;margin-top:2px'>{d}</div>"
                f"</div>"
                f"<div style='font-family:JetBrains Mono;font-size:15px;"
                f"font-weight:600;color:{clr2};white-space:nowrap;margin-left:12px'>"
                f"{disp:.2f}<span style='font-size:9px;color:#94A3B8;margin-left:2px'>"
                f"{unit}</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Full feature table ──
    with st.expander("All 16 feature values"):
        rows = []
        for col in feat_raw.columns:
            val  = latest_obs[col]
            meta = FEATURE_META.get(col, {"label": col, "unit": "", "scale": 1})
            disp = val * meta["scale"]
            unit = meta["unit"]
            havg = feat_mean[col] * meta["scale"]
            z    = (val - feat_mean[col]) / (feat_std[col] + 1e-9)
            rows.append({
                "Feature":   meta["label"],
                "Value":     f"{disp:.3f} {unit}".strip(),
                "Hist. Avg": f"{havg:.3f} {unit}".strip(),
                "Z-score":   f"{z:+.2f}",
                "Reading":   zdesc(col, val),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-label'>Regime Summary</div>",
                unsafe_allow_html=True)

    order     = ["Risk-On", "Risk-Off", "Inflation-Driven", "High-Volatility"]
    stat_cols = st.columns(4)
    for i, reg in enumerate(order):
        if reg not in stats_df.index: continue
        row = stats_df.loc[reg]; clr = rc(reg)
        with stat_cols[i]:
            active_border = f"border-top:3px solid {clr};" if reg == latest_regime else ""
            st.markdown(
                f"<div style='background:{rbg(reg)};border:1px solid {clr}28;"
                f"{active_border}border-radius:10px;padding:16px 16px 14px'>"

                f"<div style='font-size:8px;font-family:Space Grotesk;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.11em;color:{clr};"
                f"margin-bottom:10px'>{REGIME_ICONS[reg]} {reg}"
                f"{'  ·  now' if reg == latest_regime else ''}</div>"

                f"<div style='font-family:JetBrains Mono;font-size:22px;"
                f"font-weight:500;color:{clr};line-height:1'>{row['% of History']}</div>"
                f"<div style='font-size:8px;color:#94A3B8;text-transform:uppercase;"
                f"letter-spacing:0.07em;margin-bottom:10px'>of history</div>"

                f"<hr style='border:none;border-top:1px solid {clr}18;margin:8px 0'>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px 4px'>"

                f"<div><div style='font-family:JetBrains Mono;font-size:12px;"
                f"font-weight:500;color:#374151'>{row['Episodes']}</div>"
                f"<div style='font-size:8px;color:#94A3B8'>episodes</div></div>"

                f"<div><div style='font-family:JetBrains Mono;font-size:12px;"
                f"font-weight:500;color:#374151'>{row['Avg Duration (days)']}d</div>"
                f"<div style='font-size:8px;color:#94A3B8'>avg duration</div></div>"

                f"<div><div style='font-family:JetBrains Mono;font-size:12px;"
                f"font-weight:500;color:#374151'>{row['Avg VIX']}</div>"
                f"<div style='font-size:8px;color:#94A3B8'>avg VIX</div></div>"

                f"<div><div style='font-family:JetBrains Mono;font-size:12px;"
                f"font-weight:500;color:#374151'>{row['Avg CPI YoY']}</div>"
                f"<div style='font-size:8px;color:#94A3B8'>avg CPI YoY</div></div>"

                f"</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    left3, right3 = st.columns([5, 4], gap="large")

    with left3:
        st.markdown("<div class='section-label'>Episode History</div>",
                    unsafe_allow_html=True)
        episodes = []
        prev = None; ep_start = None
        for date, row in regime_df.iterrows():
            r = row["regime"]
            if r != prev:
                if prev: episodes.append({"Regime": prev, "Start": ep_start, "End": date})
                prev = r; ep_start = date
        if prev: episodes.append({"Regime": prev, "Start": ep_start,
                                   "End": regime_df.index[-1]})
        ep_df = pd.DataFrame(episodes)
        fig4  = px.timeline(ep_df, x_start="Start", x_end="End", y="Regime",
                             color="Regime", color_discrete_map=REGIME_COLORS)
        fig4.update_traces(marker_line_width=0, opacity=0.88)
        plotly_base(fig4, height=200)
        fig4.update_layout(showlegend=False, margin=dict(l=0, r=0, t=8, b=0),
                           yaxis=dict(categoryorder="array",
                                      categoryarray=["High-Volatility",
                                                     "Inflation-Driven",
                                                     "Risk-Off", "Risk-On"]))
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<div class='section-label' style='margin-top:12px'>Full Table</div>",
                    unsafe_allow_html=True)
        st.dataframe(stats_df, use_container_width=True, height=195)

    with right3:
        st.markdown("<div class='section-label'>Regime Frequency</div>",
                    unsafe_allow_html=True)
        freq = regime_df["regime"].value_counts()
        fig3 = go.Figure(go.Pie(
            labels=freq.index, values=freq.values, hole=0.58,
            marker=dict(colors=[rc(r) for r in freq.index],
                        line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textfont=dict(family="Space Grotesk", size=10),
            hovertemplate="%{label}<br>%{value} days (%{percent})<extra></extra>",
        ))
        fig3.update_layout(
            height=240, margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            annotations=[dict(
                text=f"<b>{len(regime_df)}</b><br><span style='font-size:9px'>days</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, family="JetBrains Mono, monospace")
            )]
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<div class='section-label' style='margin-top:4px'>VIX Distribution by Regime</div>",
                    unsafe_allow_html=True)
        fig_vix = go.Figure()
        for reg in order:
            mask_r   = regime_df["regime"] == reg
            vix_vals = feat_raw.loc[mask_r, "vix_level"].dropna()
            fig_vix.add_trace(go.Box(
                x=vix_vals, name=reg,
                marker_color=rc(reg), line_color=rc(reg),
                fillcolor=rbg(reg), orientation="h",
                boxmean=True, showlegend=False,
            ))
        plotly_base(fig_vix, height=195)
        fig_vix.update_layout(margin=dict(l=0, r=0, t=8, b=0))
        fig_vix.update_xaxes(title_text="VIX level", title_font_size=9)
        st.plotly_chart(fig_vix, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        "<div class='info-box'>"
        "Regime labels are shifted one day forward to avoid look-ahead bias. "
        "Bond proxy: –Δ10Y yield × 7yr duration. "
        "Transaction costs and slippage not modelled — treat as illustrative."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-label'>Equity Allocation per Regime</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: w_ro  = st.slider("Risk-On",          0, 100, 85, 5)
    with c2: w_rof = st.slider("Risk-Off",         0, 100, 40, 5)
    with c3: w_hv  = st.slider("High-Volatility",  0, 100, 20, 5)
    with c4: w_inf = st.slider("Inflation-Driven", 0, 100, 55, 5)

    ALLOC      = {"Risk-On":w_ro/100,"Risk-Off":w_rof/100,
                  "High-Volatility":w_hv/100,"Inflation-Driven":w_inf/100}
    sp500_ret  = feat_raw["sp500_ret"].reindex(regime_df.index)
    bond_ret   = -(raw_prices["DGS10"].reindex(regime_df.index).diff() / 100) * 7
    regime_lag = regime_df["regime"].shift(1).fillna("Risk-Off")
    eq_w       = regime_lag.map(ALLOC)
    strat_ret  = (eq_w * sp500_ret + (1 - eq_w) * bond_ret).dropna()
    bench_ret  = (0.6 * sp500_ret + 0.4 * bond_ret).reindex(strat_ret.index)
    strat_cum  = (1 + strat_ret).cumprod()
    bench_cum  = (1 + bench_ret).cumprod()

    def perf(rets, label):
        ar  = (1 + rets.mean()) ** 252 - 1
        av  = rets.std() * np.sqrt(252)
        sh  = ar / av if av > 0 else 0
        cum = (1 + rets).cumprod()
        dd  = (cum / cum.cummax() - 1).min()
        return {"Strategy":label,"Ann. Return":ar,"Ann. Vol":av,"Sharpe":sh,"Max DD":dd}

    ms = perf(strat_ret, "Regime-Aware")
    mb = perf(bench_ret, "Static 60/40")

    st.markdown("<br>", unsafe_allow_html=True)
    mc = st.columns(4)
    defs = [("Ann. Return","Ann. Return","{:.1%}",True),
            ("Ann. Vol",   "Ann. Vol",   "{:.1%}",False),
            ("Sharpe",     "Sharpe",     "{:.2f}",True),
            ("Max DD",     "Max Drawdown","{:.1%}",False)]
    for i, (key, lbl, fmt, hb) in enumerate(defs):
        sv = ms[key]; bv = mb[key]; delta = sv - bv
        clr2 = "#059669" if (delta > 0) == hb else "#DC2626"
        arrow = "▲" if delta > 0 else "▼"
        with mc[i]:
            st.markdown(
                f"<div class='stat-card'>"
                f"<div style='font-size:8px;font-family:Space Grotesk;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.10em;color:#94A3B8'>{lbl}</div>"
                f"<div style='display:flex;align-items:baseline;gap:7px;margin-top:5px'>"
                f"<div style='font-family:JetBrains Mono;font-size:22px;font-weight:500;"
                f"color:#0D1B2A'>{fmt.format(sv)}</div>"
                f"<div style='font-family:JetBrains Mono;font-size:10px;color:{clr2}'>"
                f"{arrow} {fmt.format(abs(delta))}</div>"
                f"</div>"
                f"<div style='font-size:9px;color:#94A3B8;margin-top:2px'>"
                f"vs {fmt.format(bv)} (60/40)</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='section-label' style='margin-top:20px'>Cumulative Return</div>",
                unsafe_allow_html=True)
    fig5 = go.Figure()
    prev_r = None; b_start = None; added = set()
    for date, row in regime_df.reindex(strat_cum.index).iterrows():
        r = row["regime"]
        if r != prev_r:
            if prev_r:
                sl = prev_r not in added
                if sl: added.add(prev_r)
                fig5.add_vrect(x0=b_start, x1=date, fillcolor=rc(prev_r),
                               opacity=0.07, line_width=0,
                               legendgroup=prev_r, name=prev_r, showlegend=sl)
            prev_r = r; b_start = date
    if prev_r:
        sl = prev_r not in added
        fig5.add_vrect(x0=b_start, x1=strat_cum.index[-1], fillcolor=rc(prev_r),
                       opacity=0.07, line_width=0, legendgroup=prev_r,
                       name=prev_r, showlegend=sl)
    fig5.add_trace(go.Scatter(x=strat_cum.index, y=strat_cum, name="Regime-Aware",
                               line=dict(color="#2563EB", width=2)))
    fig5.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="Static 60/40",
                               line=dict(color="#9CA3AF", width=1.5, dash="dot")))
    plotly_base(fig5, height=320)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div class='section-label'>Drawdown</div>", unsafe_allow_html=True)
    s_dd = (strat_cum / strat_cum.cummax() - 1) * 100
    b_dd = (bench_cum / bench_cum.cummax() - 1) * 100
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=s_dd.index, y=s_dd, name="Regime-Aware",
                               line=dict(color="#2563EB", width=1.5),
                               fill="tozeroy", fillcolor="rgba(37,99,235,0.07)"))
    fig6.add_trace(go.Scatter(x=b_dd.index, y=b_dd, name="Static 60/40",
                               line=dict(color="#9CA3AF", width=1, dash="dot")))
    plotly_base(fig6, height=190)
    fig6.update_yaxes(title_text="Drawdown (%)", title_font_size=9)
    st.plotly_chart(fig6, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    "<hr style='border:none;border-top:1px solid #F1F5F9;margin:28px 0 12px'>"
    "<div style='display:flex;justify-content:space-between;align-items:center'>"
    "<div style='font-size:10px;color:#CBD5E1;font-family:Inter'>"
    "Gaussian HMM · PCA · K-means · GARCH(1,1) · Streamlit · Plotly"
    "</div>"
    "<div style='font-size:10px;color:#CBD5E1'>"
    "Data: Yahoo Finance · FRED &nbsp;|&nbsp; "
    "⚠️ Research and portfolio demonstration only — not investment advice"
    "</div></div>",
    unsafe_allow_html=True
)
