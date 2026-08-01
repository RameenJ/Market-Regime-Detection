# =============================================================================
# refresh.py — Daily Data Refresh + Regime Re-decode
# Macro-Market Regime Detection System (US)
# =============================================================================
# What this does:
#   1. Pulls the latest market + macro data from Yahoo Finance and FRED
#   2. Appends new rows to the existing processed parquet files
#   3. Re-engineers features for the full updated history
#   4. Re-runs Viterbi decode on the full updated sequence
#   5. Saves updated regime labels + a JSON refresh log
#
# Run manually:  python refresh.py
# Run on schedule: .github/workflows/daily_refresh.yml (GitHub Actions cron)
#
# Environment variable required: FRED_API_KEY
# =============================================================================

import os
import sys
import json
import pickle
import logging
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from arch import arch_model

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("refresh.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
LOG_PATH      = "data/processed/refresh_log.json"

# ── Config ────────────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
if not FRED_API_KEY:
    log.error("FRED_API_KEY environment variable is not set. Exiting.")
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

MARKET_TICKERS = {"SP500": "^GSPC", "VIX": "^VIX"}

FRED_SERIES = {
    "CPI":        "CPIAUCSL",
    "FEDFUNDS":   "FEDFUNDS",
    "DGS10":      "DGS10",
    "DGS2":       "DGS2",
    "T5YIE":      "T5YIE",
    "BAA10Y":     "BAA10Y",
    "INDPRO":     "INDPRO",
    "PHILLY_FED": "GACDFSA066MSFRBPHI",
}

# Pull this many calendar days back as a buffer for FRED release lags
# (CPI and INDPRO arrive ~3 weeks after the reference month)
LOOKBACK_DAYS = 45

STATE_LABELS = {
    0: "Inflation-Driven",
    1: "Risk-Off",
    2: "Risk-On",
    3: "High-Volatility",
}


# =============================================================================
# STEP 1 — PULL LATEST RAW DATA
# =============================================================================

def pull_market_data(start: str) -> pd.DataFrame:
    frames = {}
    for label, ticker in MARKET_TICKERS.items():
        log.info(f"  Fetching {label} ({ticker})...")
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[label] = df[["Close"]].rename(columns={"Close": label})
    out = pd.concat(frames.values(), axis=1)
    out.columns = list(frames.keys())
    out.index.name = "date"
    return out


def pull_fred_data(start: str) -> pd.DataFrame:
    series_list = []
    for label, series_id in FRED_SERIES.items():
        log.info(f"  Fetching {label} ({series_id})...")
        try:
            s = fred.get_series(series_id, observation_start=start)
            s.index.name = "date"
            s.name = label
            series_list.append(s)
        except Exception as e:
            log.warning(f"  Could not fetch {label}: {e}")
    return pd.concat(series_list, axis=1)


def align_to_trading_days(market_df: pd.DataFrame,
                           macro_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill monthly/irregular macro data onto the market trading calendar."""
    daily_index   = market_df.index
    macro_daily   = macro_df.reindex(
        pd.date_range(macro_df.index.min(), daily_index.max(), freq="D")
    ).ffill()
    macro_daily.index.name = "date"
    macro_aligned = macro_daily.reindex(daily_index).ffill()
    return market_df.join(macro_aligned, how="inner")


def fetch_new_raw_data(last_known_date: pd.Timestamp) -> pd.DataFrame:
    start = (last_known_date - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    log.info(f"Pulling raw data from {start} onward...")
    market_df = pull_market_data(start)
    macro_df  = pull_fred_data(start)
    combined  = align_to_trading_days(market_df, macro_df)
    new_rows  = combined[combined.index > last_known_date].dropna(how="all")
    log.info(f"  {len(new_rows)} new trading day(s) available")
    return new_rows


# =============================================================================
# STEP 2 — ENGINEER FEATURES (full history — rolling windows need it all)
# =============================================================================

def engineer_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors Phase 2 exactly. Must stay in sync if Phase 2 ever changes.
    Always called on the full combined history so rolling windows are correct.
    """
    feat = pd.DataFrame(index=combined_df.index)

    # Market / vol
    feat["sp500_ret"]              = np.log(combined_df["SP500"]).diff()
    feat["realized_vol_20d"]       = feat["sp500_ret"].rolling(20).std() * np.sqrt(252)
    feat["realized_vol_60d"]       = feat["sp500_ret"].rolling(60).std() * np.sqrt(252)
    feat["vix_level"]              = combined_df["VIX"]
    feat["vix_change_20d"]         = combined_df["VIX"].diff(20)
    feat["sp500_drawdown"]         = (combined_df["SP500"] / combined_df["SP500"].cummax()) - 1

    # GARCH conditional vol
    returns_pct = feat["sp500_ret"].dropna() * 100
    garch_fit   = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="t").fit(disp="off")
    feat["garch_vol"] = (garch_fit.conditional_volatility / 100 * np.sqrt(252)
                         ).reindex(feat.index)

    # Rates / curve / credit
    feat["curve_slope_10y2y"]        = combined_df["DGS10"] - combined_df["DGS2"]
    feat["fed_funds_level"]          = combined_df["FEDFUNDS"]
    feat["fed_funds_change_60d"]     = combined_df["FEDFUNDS"].diff(60)
    feat["credit_spread"]            = combined_df["BAA10Y"]
    feat["credit_spread_change_20d"] = combined_df["BAA10Y"].diff(20)
    feat["breakeven_inflation"]      = combined_df["T5YIE"]

    # Inflation / growth
    feat["cpi_yoy"]    = combined_df["CPI"].pct_change(252) * 100
    feat["indpro_yoy"] = combined_df["INDPRO"].pct_change(252) * 100
    feat["philly_fed"] = combined_df["PHILLY_FED"]

    return feat


# =============================================================================
# STEP 3 — Z-SCORE WITH STORED TRAINING STATS
# =============================================================================

def zscore_with_stored_stats(feat: pd.DataFrame,
                              feat_mean: pd.Series,
                              feat_std: pd.Series) -> pd.DataFrame:
    """
    Use the mean/std saved from the original Phase 2 fit.
    Never re-fit the scaler — that would make new inputs incomparable
    to the distribution the PCA and HMM were trained on.
    """
    return (feat - feat_mean) / (feat_std + 1e-9)


# =============================================================================
# MAIN REFRESH ROUTINE
# =============================================================================

def run_refresh():
    log.info("=" * 60)
    log.info("Daily refresh started")
    log.info("=" * 60)

    # ── Check required files exist ────────────────────────────────────────────
    required = [
        f"{PROCESSED_DIR}/us_combined_daily.parquet",
        f"{PROCESSED_DIR}/us_features_daily.parquet",
        f"{PROCESSED_DIR}/us_regime_labels.parquet",
        f"{MODEL_DIR}/pca_model.pkl",
        f"{MODEL_DIR}/hmm_model.pkl",
        f"{MODEL_DIR}/feat_mean.pkl",
        f"{MODEL_DIR}/feat_std.pkl",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        log.error("Required files missing — run Phases 1-3 notebooks first:")
        for p in missing:
            log.error(f"  {p}")
        sys.exit(1)

    # ── Load existing combined data ───────────────────────────────────────────
    log.info("Loading existing data...")
    combined_path = f"{PROCESSED_DIR}/us_combined_daily.parquet"
    combined_existing = pd.read_parquet(combined_path)
    last_date = combined_existing.index.max()
    today     = pd.Timestamp.today().normalize()

    log.info(f"  Last date in dataset : {last_date.date()}")
    log.info(f"  Today                : {today.date()}")

    # ── Skip if nothing to do ─────────────────────────────────────────────────
    if today.weekday() >= 5:
        log.info("Weekend — no market data expected. Exiting cleanly.")
        _write_log(last_date, today, 0, "skipped_weekend")
        return

    if last_date >= today:
        log.info("Dataset already up to date. Exiting cleanly.")
        _write_log(last_date, today, 0, "already_current")
        return

    # ── Pull new raw data ─────────────────────────────────────────────────────
    new_raw = fetch_new_raw_data(last_date)
    if new_raw.empty:
        log.info("No new rows found (market holiday or data not yet published).")
        _write_log(last_date, today, 0, "no_new_data")
        return

    log.info(f"New rows: {new_raw.index.min().date()} → {new_raw.index.max().date()}")

    # ── Append new rows to combined raw ──────────────────────────────────────
    combined_updated = (
        pd.concat([combined_existing, new_raw])
        .loc[~pd.concat([combined_existing, new_raw]).index.duplicated(keep="last")]
        .sort_index()
    )
    combined_updated.to_parquet(combined_path)
    log.info(f"us_combined_daily.parquet updated → {len(combined_updated)} rows total")

    # ── Re-engineer full feature matrix ──────────────────────────────────────
    log.info("Re-engineering feature matrix (full history)...")
    feat_full  = engineer_features(combined_updated)
    core_cols  = [c for c in feat_full.columns if c != "breakeven_inflation"]
    feat_clean = feat_full[feat_full.index >= "2003-01-01"].dropna(subset=core_cols)
    feat_clean.to_parquet(f"{PROCESSED_DIR}/us_features_daily.parquet")
    log.info(f"us_features_daily.parquet updated → {len(feat_clean)} rows")

    # ── Load models + training-time scaler stats ──────────────────────────────
    log.info("Loading models...")
    def pkl(name):
        with open(f"{MODEL_DIR}/{name}.pkl", "rb") as f: return pickle.load(f)

    pca_model = pkl("pca_model")
    hmm_model = pkl("hmm_model")
    feat_mean = pkl("feat_mean")
    feat_std  = pkl("feat_std")

    # ── Z-score using stored stats ────────────────────────────────────────────
    feat_z = zscore_with_stored_stats(feat_clean, feat_mean, feat_std).dropna()
    feat_z.to_parquet(f"{PROCESSED_DIR}/us_features_zscored.parquet")

    # ── Viterbi decode — full updated sequence ────────────────────────────────
    log.info("Running Viterbi decode on full updated sequence...")
    X_pca  = pca_model.transform(feat_z.values)
    states = hmm_model.predict(X_pca)
    probs  = hmm_model.predict_proba(X_pca)

    regime_df = pd.DataFrame({
        "state_id":   states,
        "regime":     [STATE_LABELS[s] for s in states],
        "confidence": probs.max(axis=1),
    }, index=feat_z.index)

    regime_df.to_parquet(f"{PROCESSED_DIR}/us_regime_labels.parquet")
    regime_df.to_csv(f"{PROCESSED_DIR}/us_regime_labels.csv")
    log.info(f"us_regime_labels.parquet updated → {len(regime_df)} rows")

    # ── Summary ───────────────────────────────────────────────────────────────
    latest_regime = regime_df.iloc[-1]["regime"]
    latest_conf   = float(regime_df.iloc[-1]["confidence"])
    latest_date   = regime_df.index[-1]

    log.info("-" * 60)
    log.info(f"Refresh complete")
    log.info(f"  Date    : {latest_date.date()}")
    log.info(f"  Regime  : {latest_regime}")
    log.info(f"  Conf.   : {latest_conf:.1%}")
    log.info(f"  Added   : {len(new_raw)} row(s)")
    log.info("-" * 60)

    _write_log(last_date, latest_date, len(new_raw), "success",
               latest_regime, latest_conf)


def _write_log(prev_date, new_date, n_new, status,
               regime=None, confidence=None):
    """Append a structured entry to the JSON refresh log (keep last 90)."""
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "prev_date":  str(pd.Timestamp(prev_date).date()),
        "new_date":   str(pd.Timestamp(new_date).date()),
        "rows_added": n_new,
        "status":     status,
        "regime":     regime,
        "confidence": round(confidence, 4) if confidence is not None else None,
    }
    history = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    history.append(entry)
    history = history[-90:]
    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    run_refresh()
