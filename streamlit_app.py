import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

#Config

DEFAULT_UNIVERSES = {
    "SP500_ALL": {
        "csv": "universes/sp500_top100.csv",
        "benchmark": "SPY",
        "country": "US",
    },
    "EURO_STOXX_50": {
        "csv": "universes/euro_top50.csv",
        "benchmark": "VGK",
        "country": "EU",
    },
    "CHINA_A50": {
        "csv": "universes/china_a50.csv",
        "benchmark": "KWEB",
        "country": "CN",
    },
    "HANG_SENG": {
        "csv": "universes/hsi.csv",
        "benchmark": "^HSI",
        "country": "HK",
    },
    "SG": {
        "csv": "universes/sg_sti.csv",
        "benchmark": "^STI",
        "country": "SG",
    },
    "INDEXES": {
        "tickers": ["^GSPC","^IXIC","^DJI","^HSI","^N225","^STI","^FTSE","^GDAXI"],
        "benchmark": "SPY",
        "country": "Index",
    }
}

# explicit overrides (if yfinance sector missing)
TICKER_META_OVERRIDES = {}

# Helpers

def compute_beta(ticker_ret: pd.Series, bench_ret: pd.Series, window: int = 252) -> pd.Series:
    # beta = cov(r_i, r_m) / var(r_m)
    cov = ticker_ret.rolling(window, min_periods=60).cov(bench_ret)
    var = bench_ret.rolling(window, min_periods=60).var()
    return cov / var

@st.cache_data(show_spinner=False)
def download_ohlc(tickers, start, end):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )
    return data

@st.cache_data(show_spinner=False)
def fetch_info(tickers):
    out = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.fast_info if hasattr(tk, "fast_info") else {}
            # yfinance "info" can be slow; use it sparingly
            full = {}
            try:
                full = tk.info or {}
            except Exception:
                full = {}
            out[t] = {
                "sector": full.get("sector", None),
                "shortName": full.get("shortName", None) or full.get("longName", None),
            }
        except Exception:
            out[t] = {"sector": None, "shortName": None}
    return out

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    return df

def extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # yf.download returns either single-index cols (single ticker) or multi-index
    if isinstance(raw.columns, pd.MultiIndex):
        sub = raw[ticker].copy()
    else:
        sub = raw.copy()
    sub = normalize_columns(sub)
    sub.index = pd.to_datetime(sub.index)
    return sub

def build_screener(
    universe_name,
    tickers,
    sector_map,
    uni_cfg,
    lookback_days=252,
    beta_window=252,
    overnight_z_threshold=-2.0,
    retracement_min=0.5,
    momentum_days=(5, 20),
    disloc_z_threshold=2.0,   # threshold for overnight residual z
):

    bench = uni_cfg["benchmark"]
    country_bucket = uni_cfg.get("country", None)

    start = (datetime.today() - timedelta(days=int(lookback_days * 2.2))).date()
    end = (datetime.today() + timedelta(days=1)).date()

    all_tickers = list(dict.fromkeys(tickers + [bench]))
    raw = download_ohlc(all_tickers, start, end)

    # --- Benchmark series ---
    bench_df = extract_ticker_frame(raw, bench)
    bench_df = bench_df.sort_index()
    bench_ret = bench_df["close"].pct_change()

    bench_prev_close = bench_df["close"].shift(1)
    bench_overnight = bench_df["open"] / bench_prev_close - 1.0

    # --- Choose ONE as-of date so all tickers align ---
    # Use latest date where benchmark has both prev_close and open (i.e., bench_overnight notna)
    bench_valid_dates = bench_overnight.dropna().index
    if len(bench_valid_dates) == 0:
        return pd.DataFrame(), bench

    asof_dt = bench_valid_dates.max()

    info = fetch_info(tickers)

    rows = []
    for t in tickers:
        df = extract_ticker_frame(raw, t).sort_index()

        # must have asof_dt
        if df.empty or asof_dt not in df.index:
            continue

        # need enough history for rolling stats
        if df["close"].dropna().shape[0] < 80:
            continue

        # --- core returns ---
        ret = df["close"].pct_change()

        prev_close = df["close"].shift(1)
        overnight_ret = df["open"] / prev_close - 1.0

        # overnight absolute z-score (self-normalized)
        on_mu = overnight_ret.rolling(beta_window, min_periods=60).mean()
        on_sd = overnight_ret.rolling(beta_window, min_periods=60).std()
        overnight_z = (overnight_ret - on_mu) / on_sd

        # --- retracement (gap fill strength), only for gap-down ---
        gap = df["open"] - prev_close
        gap_denom = gap.abs().replace(0, np.nan)
        retracement = (df["close"] - df["open"]) / gap_denom
        retracement = retracement.where(gap < 0)

        # --- beta ---
        beta = compute_beta(ret, bench_ret, window=beta_window)
        
        # --- residual = overnight dislocation vs beta ---
        residual = overnight_ret - beta * bench_overnight
        
        resid_mu = residual.rolling(beta_window, min_periods=60).mean()
        resid_sd = residual.rolling(beta_window, min_periods=60).std()
        residual_z = (residual - resid_mu) / resid_sd

        # --- optional: keep close-to-close residual for additional context ---
        expected_cc = beta * bench_ret
        residual_cc = ret - expected_cc

        rc_mu = residual_cc.rolling(beta_window, min_periods=60).mean()
        rc_sd = residual_cc.rolling(beta_window, min_periods=60).std()
        residual_cc_z = (residual_cc - rc_mu) / rc_sd

        # --- momentum ---
        mom_5 = df["close"].pct_change(momentum_days[0])
        mom_20 = df["close"].pct_change(momentum_days[1])

        # --- read values at ONE aligned date ---
        last_dt = asof_dt

        def fval(s):
            v = s.loc[last_dt] if last_dt in s.index else np.nan
            return float(v) if pd.notna(v) else np.nan

        r = {
            "universe": universe_name,
            "date": last_dt.date(),
            "ticker": t,
            "name": info.get(t, {}).get("shortName"),
        
            "sector": (
                TICKER_META_OVERRIDES.get(t, {}).get("sector")
                or sector_map.get(t)
                or info.get(t, {}).get("sector")
            ),
            "country": (
                TICKER_META_OVERRIDES.get(t, {}).get("country")
                or country_bucket
            ),
        
            # prices
            "px_close": fval(df["close"]),
        
            # overnight move diagnostics
            "overnight_ret": fval(overnight_ret),
            "overnight_z": fval(overnight_z),
            "retracement": fval(retracement),
        
            # beta + dislocation (MAIN SIGNAL)
            "beta": fval(beta),
            "residual": fval(residual),        # ← overnight residual vs beta
            "residual_z": fval(residual_z),    # ← THIS is the dislocation z
        
            # momentum confirmation
            "mom_5d": fval(mom_5),
            "mom_20d": fval(mom_20),
        }

        # --- flags ---
        r["flag_overnight_selloff"] = (pd.notna(r["overnight_z"]) and r["overnight_z"] <= overnight_z_threshold)

        r["flag_retracement"] = (pd.notna(r["retracement"]) and r["retracement"] >= retracement_min)
        
        r["flag_beta_dislocation"] = (pd.notna(r["residual_z"]) and abs(r["residual_z"]) >= disloc_z_threshold)
        
        rows.append(r)

    out = pd.DataFrame(rows)

    # spreads: do spreads on overnight dislocation (more aligned with your scatter + “cheap/exp”)
    if not out.empty:
        if out["sector"].notna().any():
            out["sector_residual_z_spread"] = out["residual_z"] - out.groupby("sector")["residual_z"].transform("mean")
        else:
            out["sector_residual_z_spread"] = np.nan

        if out["country"].notna().any():
            out["country_residual_z_spread"] = (out["residual_z"] - out.groupby("country")["residual_z"].transform("mean"))
        else:
            out["country_residual_z_spread"] = np.nan


    return out, bench

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Equity Dislocation Screener", layout="wide")
st.title("Equity Screener — Overnight Selloff + Beta Dislocation + Dip Momentum")

with st.sidebar:
    st.header("Universe")
    uni_name = st.selectbox("Pick universe", list(DEFAULT_UNIVERSES.keys()))
    cfg = DEFAULT_UNIVERSES[uni_name]

    st.header("Parameters")
    lookback = st.slider("History lookback (approx trading days)", 120, 2520, 600, step=60)
    beta_window = st.slider("Beta / z-score window", 60, 504, 252, step=21)

    overnight_thr = st.slider("Z threshold (selloff)", -5.0, -1.0, -2.0, step=0.1)
    retr_min = st.slider("Min retracement (gap fill strength)", 0.0, 2.0, 0.5, step=0.05)

    st.caption("Tip: add new tickers by editing DEFAULT_UNIVERSES in app.py (or load a YAML/CSV).")

st.subheader(f"{uni_name} (benchmark: {cfg['benchmark']})")

sector_map = {}

if "csv" in cfg:
    uni_df = pd.read_csv(cfg["csv"])
    tickers = uni_df["ticker"].astype(str).str.strip().dropna().unique().tolist()
    if "sector" in uni_df.columns:
        sector_map = dict(zip(uni_df["ticker"], uni_df["sector"]))
elif "tickers" in cfg:
    tickers = list(dict.fromkeys(cfg["tickers"]))
else:
    raise ValueError("Universe config must have either csv or tickers")

with st.spinner("Downloading prices & building signals..."):
    df, bench = build_screener(
        universe_name=uni_name,
        tickers=tickers,
        sector_map=sector_map,
        uni_cfg=cfg,
        lookback_days=lookback,
        beta_window=beta_window,
        overnight_z_threshold=overnight_thr,
        retracement_min=retr_min
    )

if df.empty:
    st.warning("No data returned (tickers might be invalid, delisted, or insufficient history).")
    st.stop()

# Ranking view (cheap/exp & dislocations)
c1, c2, c3 = st.columns(3)
c1.metric("Names", len(df))
c2.metric("Overnight selloff flags", int(df["flag_overnight_selloff"].sum()))
c3.metric("Beta dislocation flags", int(df["flag_beta_dislocation"].sum()))

# Filters
st.markdown("### Filters")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    only_selloff = st.checkbox("Only overnight selloff", value=False)
with fcol2:
    only_disloc = st.checkbox("Only beta dislocation", value=False)
with fcol3:
    only_retrace = st.checkbox("Only strong retracement", value=False)

df_view = df.copy()
if only_selloff:
    df_view = df_view[df_view["flag_overnight_selloff"]]
if only_disloc:
    df_view = df_view[df_view["flag_beta_dislocation"]]
if only_retrace:
    df_view = df_view[df_view["flag_retracement"]]

# Table
st.markdown("### Screener table")
show_cols = [
    "ticker","name","sector","country","px_close",
    "overnight_ret","overnight_z","retracement",
    "beta","residual","residual_z",
    "mom_5d","mom_20d",
    "sector_residual_z_spread","country_residual_z_spread",
    "flag_overnight_selloff","flag_retracement","flag_beta_dislocation"
]


# --- column groups ---
PCT_COLS = [
    "overnight_ret",
    "residual",      # residual is a % move
]

ZSCORE_COLS = [
    "overnight_z",
    "residual_z",
    "sector_residual_z_spread",
    "country_residual_z_spread",
]

# retracement is a ratio (0.5 = 50% gap fill). show as %
RETRACE_COLS = ["retracement"]

DECIMAL_COLS = ["beta", "px_close"]  # keep decimals (beta 2dp is fine)

df_view = df_view.sort_values(
    ["flag_beta_dislocation", "residual_z"],
    ascending=[False, True]
)

import numpy as np

def dip_colormap(s: pd.Series):
    """
    For dip-buy lens:
    - more negative values => greener (cheap)
    - more positive values => redder (expensive)
    """
    s = s.astype(float)
    vmax = np.nanmax(np.abs(s.values)) if np.isfinite(s).any() else 1.0
    # Normalize to [-1, 1]
    x = np.clip(s / (vmax if vmax != 0 else 1.0), -1, 1)

    # Map: -1 => green, 0 => white, +1 => red
    # Using simple RGB interpolation
    def to_rgb(val):
        if np.isnan(val):
            return ""
        if val < 0:
            # green-ish
            g = 255
            r = int(255 * (1 + val))   # val is negative, r decreases
            b = int(255 * (1 + val))
        else:
            # red-ish
            r = 255
            g = int(255 * (1 - val))
            b = int(255 * (1 - val))
        return f"background-color: rgb({r},{g},{b});"

    return [to_rgb(v) for v in x.values]

def pnl_colormap(s: pd.Series):
    """
    For returns:
    - positive => greener
    - negative => redder
    """
    s = s.astype(float)
    vmax = np.nanmax(np.abs(s.values)) if np.isfinite(s).any() else 1.0
    x = np.clip(s / (vmax if vmax != 0 else 1.0), -1, 1)

    def to_rgb(val):
        if np.isnan(val):
            return ""
        if val > 0:
            # green
            g = 255
            r = int(255 * (1 - val))
            b = int(255 * (1 - val))
        else:
            # red
            r = 255
            g = int(255 * (1 + val))  # val negative, g decreases
            b = int(255 * (1 + val))
        return f"background-color: rgb({r},{g},{b});"

    return [to_rgb(v) for v in x.values]

df_tbl = df_view[show_cols].copy()

# formatting rules
fmt = {}
for c in PCT_COLS:
    if c in df_tbl.columns:
        fmt[c] = "{:.2%}"          # percent with 2 decimals
for c in ZSCORE_COLS:
    if c in df_tbl.columns:
        fmt[c] = "{:.1f}"          # 1 dp for z-scores
for c in RETRACE_COLS:
    if c in df_tbl.columns:
        fmt[c] = "{:.0%}"          # retracement as whole percent (e.g., 50%)
for c in DECIMAL_COLS:
    if c in df_tbl.columns:
        fmt[c] = "{:.2f}"          # keep decimals

styled = (
    df_tbl
    .style
    .format(fmt, na_rep="—")
)

# apply column-wise coloring
for c in PCT_COLS:
    if c in df_tbl.columns:
        styled = styled.apply(pnl_colormap, subset=[c])

for c in ZSCORE_COLS:
    if c in df_tbl.columns:
        styled = styled.apply(dip_colormap, subset=[c])

for c in RETRACE_COLS:
    if c in df_tbl.columns:
        styled = styled.apply(pnl_colormap, subset=[c])  # higher retracement => greener

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True
)


# Scatter
import altair as alt

st.markdown(
    "### Scatter: Cheap / Expensive vs Momentum  \n"
    "*x = beta-adjusted overnight mispricing (z), y = 20-day momentum*"
)

color_key = "sector" if df_view["sector"].notna().any() else "country"

plot_df = df_view.copy()
plot_df = plot_df.dropna(subset=["residual_z", "mom_20d"])

base = alt.Chart(plot_df).mark_circle(size=70, opacity=0.85).encode(
    x=alt.X("residual_z:Q", title="Beta-adjusted overnight mispricing (z) ← Cheap | Expensive →"),
    y=alt.Y("mom_20d:Q", title="20D momentum"),
    color=alt.Color(f"{color_key}:N", legend=alt.Legend(title=color_key)),
    tooltip=[
        "ticker:N", "name:N", "sector:N", "country:N",
        alt.Tooltip("overnight_ret:Q", format=".2%"),
        alt.Tooltip("overnight_z:Q", format=".1f"),
        alt.Tooltip("residual_z:Q", format=".1f"),
        alt.Tooltip("retracement:Q", format=".2%"),
        alt.Tooltip("mom_20d:Q", format=".2%")
    ]
)

# reference lines at x=0 and y=0
x0 = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeWidth=2).encode(x="x:Q")
y0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=2).encode(y="y:Q")

# quadrant labels (positions are in data coords; tweak if your ranges differ)
labels = pd.DataFrame({
    "x": [-2.5,  1.5, -2.5,  1.5],
    "y": [ 0.15, 0.15, -0.15, -0.15],
    "label": [
        "Cheap + Strong momentum",
        "Expensive + Strong momentum",
        "Cheap + Weak momentum",
        "Expensive + Weak momentum"
    ]
})
txt = alt.Chart(labels).mark_text(align="left", baseline="top", dx=5, dy=-5).encode(
    x="x:Q", y="y:Q", text="label:N"
)

chart = (base + x0 + y0 + txt).properties(height=450).interactive()

st.altair_chart(chart, use_container_width=True)

# Quick "what lost steam" view: weak momentum + negative residual z
st.markdown("### Themes losing steam (low momentum + cheap vs beta)")
losing = df.sort_values(["mom_20d","residual_z"], ascending=[True, True]).head(20)
st.dataframe(losing[show_cols], use_container_width=True)


