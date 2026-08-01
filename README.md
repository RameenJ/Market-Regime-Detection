# Macro-Market Regime Detection System

A quant-research-style dashboard that classifies the US macro-financial environment
into four distinct regimes using unsupervised machine learning.

## Live Demo
*(add your Streamlit Community Cloud URL here after deployment)*

## Regimes Detected
| Regime | Description |
|---|---|
| 🟢 Risk-On | Low volatility, tight credit spreads, expanding economy |
| 🔴 Risk-Off | Elevated caution, wider spreads, slow-growth conditions |
| 🟠 High-Volatility | Crisis / stress periods (GFC 2008, COVID 2020) |
| 🟣 Inflation-Driven | Elevated CPI, rising breakeven inflation (2021–2023) |

## Methods
- **PCA** (4 components, 74.3% variance explained) for dimensionality reduction
- **Gaussian HMM** (4 states, full covariance, 20 random restarts) as the primary regime engine
- **K-means** (k=4) as an independent cross-check — 80.3% agreement with HMM
- **GARCH(1,1)** for model-based conditional volatility estimation

## Data Sources (all free)
| Series | Source | Description |
|---|---|---|
| S&P 500, VIX | Yahoo Finance | Equity index and implied volatility |
| CPI, INDPRO | FRED | Inflation and industrial production |
| DGS10, DGS2 | FRED | Treasury yields (curve slope) |
| FEDFUNDS | FRED | Federal Funds Rate |
| T5YIE | FRED | 5Y breakeven inflation |
| BAA10Y | FRED | Credit spread (Moody's Baa – 10Y Treasury) |
| GACDFSA066MSFRBPHI | FRED | Philly Fed Manufacturing Index (PMI proxy) |

> **Note on data availability:** FRED restricted `BAMLH0A0HYM2` (ICE BofA HY spread)
> to a rolling 3-year window in April 2026; `BAA10Y` is used instead.
> ISM PMI was removed from FRED in 2016 (licensed/proprietary); the Philly Fed
> Manufacturing Index is used as the standard free substitute.

## Project Structure
```
├── dashboard.py                  # Streamlit app (Phase 5)
├── requirements.txt
├── .streamlit/config.toml
├── Phase1_US_Data_Pipeline.ipynb
├── Phase2_Feature_Engineering.ipynb
├── Phase3_Modelling.ipynb
├── Phase4_Explanation_Engine.ipynb
├── data/
│   ├── raw/                      # Cached raw API pulls
│   └── processed/                # Feature matrices, regime labels, stats
└── models/                       # Fitted HMM, PCA, metadata pickles
```

## Running Locally
```bash
pip install -r requirements.txt
# Run Phases 1–4 notebooks first to generate data/ and models/
streamlit run dashboard.py
```

## Deploying to Streamlit Community Cloud
1. Push this repo to GitHub (include `data/` and `models/` folders)
2. Go to share.streamlit.io → New app → select repo → `dashboard.py`
3. Deploy — Streamlit Cloud installs `requirements.txt` automatically

> ⚠️ **Disclaimer:** For research and portfolio demonstration purposes only.
> Not investment advice.
