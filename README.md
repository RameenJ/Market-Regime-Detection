# Macro-Market Regime Detection System

A quant-research-style system that classifies the US macro-financial environment into four interpretable regimes — **Risk-On**, **Risk-Off**, **High-Volatility**, and **Inflation-Driven** — using unsupervised machine learning on 16 engineered macro and market features.

> **Live dashboard →** https://market-regime-detection-jp7kvkqycthtim5jbfgmmw.streamlit.app/  
> **Data:** Yahoo Finance · FRED (Federal Reserve Economic Data)  
> **Update cadence:** Automatic daily refresh via GitHub Actions (weekdays, 22:30 UTC)

---

## Why regime detection instead of price forecasting?

Most ML-in-finance projects try to predict tomorrow's price. This project asks a different question: *what kind of market environment are we in right now?*

That question matters because asset class behaviour, risk premia, and strategy performance all vary systematically across macro regimes. The same equity strategy that thrives in a Risk-On environment can destroy capital in a High-Volatility one. Regime classification is closer to how professional macro and quant researchers actually frame market analysis — not "will the market be up tomorrow?" but "what is the governing macro regime, and how should positioning reflect it?"

---

## Regimes

| Regime | Description | Avg VIX | Avg CPI YoY | Avg Credit Spread | Freq. |
|---|---|---|---|---|---|
| 🟢 **Risk-On** | Low vol, tight spreads, expanding economy, healthy sentiment | 14.5 | 3.4% | 1.73pp | 26.6% |
| 🔴 **Risk-Off** | Elevated caution, below-average CPI, wider spreads | 16.8 | 1.7% | 2.61pp | 41.0% |
| 🟠 **High-Volatility** | Crisis / stress — GFC 2008, COVID 2020, acute stress events | 32.5 | 1.9% | 3.55pp | 14.6% |
| 🟣 **Inflation-Driven** | Elevated price pressure, rising breakeven inflation — 2021-2023 era | 20.4 | 4.1% | 2.05pp | 17.8% |

### Historical validation

The model was validated against four known macro events before deployment:

| Event | Date | Model label | Expected |
|---|---|---|---|
| COVID crash | Mar 16, 2020 | High-Volatility ✓ | ✓ |
| GFC peak stress | Oct 15, 2008 | High-Volatility ✓ | ✓ |
| Inflation peak | Jun 13, 2022 | Inflation-Driven ✓ | ✓ |
| Bull market | Jan 26, 2018 | Risk-On ✓ | ✓ |

---

## Methods

### Pipeline overview

```
Raw data (Yahoo Finance + FRED)
        │
        ▼
Phase 1 — Data collection & alignment
        │  Daily market data + monthly macro series
        │  Forward-filled onto trading-day calendar
        │
        ▼
Phase 2 — Feature engineering  (16 features)
        │  Returns, rolling vol, GARCH, drawdown,
        │  yield curve, credit spreads, CPI YoY, PMI proxy
        │
        ▼
Phase 3 — Modelling
        │  PCA (4 components, 74.3% variance)
        │  → K-means (cross-check, k=4, silhouette 0.289)
        │  → Gaussian HMM (4 states, full covariance, 20 restarts)
        │  → Viterbi decode → regime sequence
        │
        ▼
Phase 4 — Explanation engine
        │  Feature contribution scoring
        │  Plain-English "why" narrative generation
        │  Regime statistics
        │
        ▼
Phase 5 — Dashboard  (Streamlit)
        │  Timeline · Current Regime · Statistics · Backtest
        │
        ▼
Phase 6 — Daily refresh  (GitHub Actions)
           New data → re-engineer → Viterbi → commit → redeploy
```

### PCA

16 z-scored features are reduced to 4 principal components, explaining 74.3% of total variance:

| Component | Top features (by loading) | Interpretation |
|---|---|---|
| PC1 (39.3%) | Realized vol (20d, 60d), VIX, GARCH vol, credit spread | Volatility / stress axis |
| PC2 (16.1%) | CPI YoY, Fed Funds, yield curve slope, credit spread Δ | Inflation / policy axis |
| PC3 (10.8%) | VIX change, yield curve slope, INDPRO YoY, Philly Fed | Growth momentum axis |
| PC4 (8.0%) | Drawdown, CPI YoY, VIX change, Fed Funds change | Equity stress / policy shift axis |

### Gaussian HMM

A Gaussian HMM with 4 hidden states and full covariance matrices is the primary regime engine. Full covariance allows each state to model within-regime feature correlations — for example, in a Risk-Off state, VIX and credit spreads tend to rise together.

Key results:

| Metric | Value |
|---|---|
| States | 4 |
| Covariance type | Full |
| Random restarts | 20 (best log-likelihood selected) |
| Converged | Yes |
| Avg log-likelihood per obs | –5.04 nats |
| BIC / n\_obs | 10.19 |
| Avg posterior confidence | 99.2% |
| Low-confidence days (<60%) | 28 of 5,885 |

**Transition matrix** (rows = from, cols = to):

|  | Risk-On | Risk-Off | Inflation-Driven | High-Volatility |
|---|---|---|---|---|
| **Risk-On** | 0.993 | 0.001 | 0.006 | 0.000 |
| **Risk-Off** | 0.000 | 0.996 | 0.001 | 0.003 |
| **Inflation-Driven** | 0.009 | 0.000 | 0.987 | 0.004 |
| **High-Volatility** | 0.003 | 0.011 | 0.000 | 0.986 |

All diagonal values exceed 0.98 — once in a regime, the market strongly tends to stay there. This is the expected property of genuine macro regimes, as opposed to noisy day-to-day signals.

### K-means cross-check

K-means clustering (k=4) was run independently on the PCA-reduced features as a methodological cross-check. Agreement with the HMM-decoded sequence was measured using the Hungarian algorithm (optimal one-to-one state mapping):

- **Agreement: 80.3%** — both methods find broadly the same structure
- The 19.7% disagreement is concentrated in the Inflation-Driven state, where the HMM's temporal memory adds meaningful value over K-means' memoryless assignment

### GARCH volatility

A GARCH(1,1) model with Student-t innovations is fitted on S&P 500 log returns to produce a model-based conditional volatility estimate. This is used as an additional feature alongside simple rolling realized volatility. The key difference: GARCH explicitly models volatility clustering — the tendency for high-volatility days to follow high-volatility days — whereas rolling standard deviation treats each window equally.

---

## Data

### Sources

| Series | Source | FRED ID / Ticker | Frequency |
|---|---|---|---|
| S&P 500 | Yahoo Finance | `^GSPC` | Daily |
| VIX | Yahoo Finance | `^VIX` | Daily |
| CPI (inflation level) | FRED | `CPIAUCSL` | Monthly |
| Federal Funds Rate | FRED | `FEDFUNDS` | Monthly |
| 10Y Treasury yield | FRED | `DGS10` | Daily |
| 2Y Treasury yield | FRED | `DGS2` | Daily |
| 5Y Breakeven inflation | FRED | `T5YIE` | Daily |
| Credit spread (Baa–10Y) | FRED | `BAA10Y` | Daily |
| Industrial Production | FRED | `INDPRO` | Monthly |
| Philly Fed Manufacturing | FRED | `GACDFSA066MSFRBPHI` | Monthly |

### Data availability decisions

Two standard series were unavailable and required substitution. These decisions are documented because they reflect real-world data sourcing constraints that don't appear in textbooks.

**High-yield credit spread (`BAMLH0A0HYM2` → `BAA10Y`)**  
In April 2026, FRED restricted the ICE BofA US High Yield Index spread (`BAMLH0A0HYM2`) to a rolling 3-year window. The full history is now only available through paid commercial licenses (Bloomberg, ICE Data Indices). `BAA10Y` — Moody's Baa corporate bond yield minus the 10Y Treasury — is used as a substitute. It captures the same economic signal (compensation for credit risk above the risk-free rate) using a series that Moody's makes freely available with full history.

**ISM Manufacturing PMI → Philly Fed Manufacturing Index**  
The ISM Manufacturing PMI is the standard leading economic indicator used in regime models. It is not available free: ISM asked FRED to remove all its series in 2016, and the data is now licensed through Bloomberg or Moody's Analytics. The Philadelphia Fed Manufacturing Index (`GACDFSA066MSFRBPHI`) is used as a substitute. It uses the same diffusion-index methodology — percent of respondents reporting increases minus percent reporting decreases — and is closely watched in practice as an early-month signal that tends to lead the ISM release. The series is centered at 0 rather than ISM's 50, but the interpretive logic is identical: above 0 = expansion, below 0 = contraction.

### Feature engineering

Raw series are transformed before modelling. Key transformations:

| Feature | Transformation | Rationale |
|---|---|---|
| S&P 500 | Log returns | Removes scale dependence; better statistical properties than simple returns |
| Realized volatility | Rolling std of log returns × √252 (20d and 60d) | Two windows: 20d captures recent stress, 60d captures the prevailing vol regime |
| GARCH vol | GARCH(1,1) fitted on returns × 100 | Model-based vol with memory; smoother and more theoretically grounded than rolling std |
| Drawdown | (Price / rolling max) – 1 | Captures accumulated distress, not just today's return |
| CPI / INDPRO | YoY % change via `.pct_change(252)` | Level is meaningless for regime detection; rate of change captures inflation/growth |
| Yield curve slope | DGS10 – DGS2 | Inversion (negative) has preceded every US recession since the 1970s |
| Fed Funds change | `.diff(60)` | Distinguishes hiking from cutting cycles; direction often matters more than level |
| Credit spread change | `.diff(20)` | Captures whether conditions are tightening or loosening right now |
| All features | Z-scored using Phase 2 training mean/std | Required before PCA/HMM; prevents large-range features (VIX) dominating small-range ones (CPI) |

**Alignment note:** Monthly macro series (CPI, INDPRO, FEDFUNDS, Philly Fed) are forward-filled onto the daily trading calendar. This reflects what a real market participant would know on any given day — the last published reading — and avoids look-ahead bias that would arise from backward-fill or interpolation.

---

## Features (16 total)

| # | Feature | Signal captured |
|---|---|---|
| 1 | S&P 500 daily return | Equity direction |
| 2 | Realized vol (20d) | Short-term volatility |
| 3 | Realized vol (60d) | Medium-term volatility |
| 4 | VIX level | Implied volatility / market fear |
| 5 | VIX change (20d) | Momentum in fear |
| 6 | S&P 500 drawdown | Distance from rolling peak |
| 7 | GARCH conditional vol | Model-based volatility estimate |
| 8 | Yield curve slope (10Y–2Y) | Recession signal / policy stance |
| 9 | Fed Funds Rate | Monetary policy level |
| 10 | Fed Funds change (60d) | Hiking or cutting cycle |
| 11 | Credit spread (Baa–10Y) | Corporate risk appetite |
| 12 | Credit spread change (20d) | Momentum in credit conditions |
| 13 | 5Y breakeven inflation | Market inflation expectations |
| 14 | CPI inflation YoY | Actual inflation rate |
| 15 | Industrial production YoY | Real economic output (lagging) |
| 16 | Philly Fed Manufacturing | Growth sentiment (leading, PMI proxy) |

---

## Dashboard

Four tabs:

- **Timeline** — S&P 500 and VIX with colour-coded regime bands; year-range slider; confidence panel
- **Current Regime** — Regime card with plain-English explanation, state probabilities, feature contribution chart, top 3 drivers
- **Statistics** — Episode history (Gantt), regime frequency, VIX distribution per regime, full stats table
- **Backtest** — Regime-aware portfolio vs static 60/40; interactive equity allocation sliders; cumulative return and drawdown charts

---

## Backtest

A simple regime-conditional allocation is tested against a static 60/40 benchmark to illustrate regime-aware portfolio construction. Default allocations:

| Regime | Equity weight | Bond weight |
|---|---|---|
| Risk-On | 85% | 15% |
| Risk-Off | 40% | 60% |
| High-Volatility | 20% | 80% |
| Inflation-Driven | 55% | 45% |

**Look-ahead bias:** Regime labels are lagged by one day before computing returns — only the regime label that was available at close of day *t* is used to set the allocation for day *t+1*.

**Bond proxy:** Daily bond return approximated as −Δ(10Y yield) × 7yr duration.

**Limitations:** Transaction costs, slippage, and rebalancing friction are not modelled. This backtest is illustrative of the regime framework's logic, not a tradeable strategy.

---

## Daily Refresh

A GitHub Actions workflow runs every weekday at 22:30 UTC (90 minutes after US market close):

```
Pull new data → Append to parquet → Re-engineer features
→ Z-score with training stats → Viterbi decode (full sequence)
→ Commit updated files → Streamlit Cloud redeploys
```

The fitted HMM and PCA models are **not** re-fitted on each refresh. The frozen models from the original Phase 3 training run are used to decode the extended sequence. This keeps regime boundaries stable over time. Periodic manual re-fitting (e.g. quarterly) is a deliberate decision, not an oversight.

---

## Project Structure

```
├── dashboard.py                        # Streamlit app (Phase 5)
├── refresh.py                          # Daily data refresh script (Phase 6)
├── requirements.txt                    # Python dependencies
├── .github/
│   └── workflows/
│       └── daily_refresh.yml           # GitHub Actions cron workflow
│
├── Phase1_US_Data_Pipeline.ipynb       # Data collection
├── Phase2_Feature_Engineering.ipynb    # Feature engineering + GARCH
├── Phase3_Modelling.ipynb              # PCA + K-means + HMM
├── Phase4_Explanation_Engine.ipynb     # Contribution scoring + narrative
│
├── data/
│   ├── raw/                            # Cached raw API pulls (not committed)
│   └── processed/
│       ├── us_combined_daily.parquet   # Aligned raw series
│       ├── us_features_daily.parquet   # 16 engineered features (raw)
│       ├── us_features_zscored.parquet # Z-scored features (model input)
│       ├── us_regime_labels.parquet    # Viterbi-decoded regime sequence
│       ├── us_regime_stats.parquet     # Per-regime statistics table
│       ├── us_feature_contributions.parquet  # Pre-computed contributions
│       └── refresh_log.json            # Daily refresh run history
│
└── models/
    ├── pca_model.pkl                   # Fitted PCA (4 components)
    ├── hmm_model.pkl                   # Fitted Gaussian HMM (4 states)
    ├── state_labels.pkl                # State ID → regime name mapping
    ├── regime_colors.pkl               # Regime colour palette
    ├── state_means_df.pkl              # Per-state raw feature means
    ├── feature_meta.pkl                # Display names + units
    ├── feat_mean.pkl                   # Training-time feature means (for z-scoring)
    └── feat_std.pkl                    # Training-time feature stds (for z-scoring)
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Market-Regime-Detection.git
cd Market-Regime-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY="your_32_char_key_here"

# 4. Run Phase 1–4 notebooks to generate data/ and models/
#    (or skip if cloning a repo that already has committed data files)

# 5. Launch the dashboard
streamlit run dashboard.py

# 6. Run a manual data refresh
python refresh.py
```

---

## Setup: GitHub Actions Auto-Refresh

1. Push the full repo to GitHub (include `data/processed/` and `models/` — the workflow needs them)
2. Go to **Settings → Actions → General → Workflow permissions → Read and write permissions → Save**
3. Go to **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `FRED_API_KEY`
   - Value: your 32-character FRED API key
4. The workflow triggers automatically at 22:30 UTC Mon–Fri, or manually from the **Actions** tab

---

## Limitations and Known Constraints

- **Model re-fitting:** The HMM is not automatically re-fitted on new data. Regime boundaries are stable but may drift from reality over very long periods without a manual re-fit.
- **Feature lag:** Monthly macro series (CPI, INDPRO) are released 2–3 weeks after the reference month. The most recent available reading is always used; the dashboard does not model release-timing uncertainty.
- **US only:** This version covers US equities and macro data. The project plan includes an EU extension (Euro Stoxx 50, ECB data, HICP) as future work.
- **Backtest limitations:** The regime-aware portfolio backtest is illustrative only. It does not account for transaction costs, slippage, or the difficulty of executing daily rebalancing in practice.
- **Single forward-pass vs Viterbi:** The dashboard's "state probabilities" panel uses a single HMM forward pass on the latest observation. This can differ from the Viterbi-decoded label (which uses full sequence context) on days near a regime transition. The Viterbi label is always shown as the primary regime designation.

---

## Stack

| Component | Library / Service |
|---|---|
| Data | `yfinance`, `fredapi` |
| Feature engineering | `pandas`, `numpy`, `arch` (GARCH) |
| Modelling | `scikit-learn` (PCA, K-means), `hmmlearn` (Gaussian HMM) |
| Dashboard | `streamlit`, `plotly` |
| Deployment | Streamlit Community Cloud |
| Auto-refresh | GitHub Actions |
| Serialisation | `pickle`, `pyarrow` (parquet) |

---

## Disclaimer

This project is built for portfolio and research demonstration purposes. Nothing in this repository constitutes investment advice. All regime classifications, portfolio tilt suggestions, and backtest results are illustrative and should not be used as the basis for financial decisions.

---

*Built by Rameen — BS Artificial Intelligence, pursuing Masters in Data Science*  
