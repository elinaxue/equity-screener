import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Streamlit config
# =========================================================
st.set_page_config(page_title="ETF Flow Monitor", layout="wide")

# =========================================================
# Defaults
# =========================================================
PRESET_UNIVERSES = {
    "US Sector ETFs": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"],
    "Broad Equity ETFs": ["SPY", "IVV", "VOO", "QQQ", "IWM", "DIA"],
    "Rates / Bonds": ["SHY", "IEF", "TLT", "LQD", "HYG", "BND", "AGG"],
    "Commodities / Gold": ["GLD", "SLV", "USO", "DBA"],
    "Mixed Asset-Class": ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO"],
    "Flow Study Core": ["SPY", "IVV", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "EWA"],
}

DEFAULT_BUCKET_MAP = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
    "SPY": "US Large Cap",
    "IVV": "US Large Cap",
    "VOO": "US Large Cap",
    "QQQ": "US Growth / Nasdaq",
    "IWM": "US Small Cap",
    "DIA": "US Large Cap",
    "SHY": "Short Duration Bonds",
    "IEF": "Intermediate Treasuries",
    "TLT": "Long Duration Treasuries",
    "LQD": "Investment Grade Credit",
    "HYG": "High Yield Credit",
    "BND": "Core Bonds",
    "AGG": "Core Bonds",
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil",
    "DBA": "Agriculture",
    "EWA": "Australia Equity",
}

TRUE_FLOW_MIN_START = pd.Timestamp("2015-01-01")

# =========================================================
# Helpers
# =========================================================
def parse_ticker_text(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = [x.strip().upper() for x in text.replace("\n", ",").split(",")]
    return [x for x in parts if x]


def group_columns_sum(df: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    if df.empty:
        return df
    return df.T.groupby(mapping).sum(min_count=1).T


def is_ishares(fund_family: str | None, ticker: str) -> bool:
    ff = str(fund_family or "").lower()
    return ("ishares" in ff) or ("blackrock" in ff and ticker.upper().startswith("I"))


def first_trading_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(index=index, data=False)
    first_idx = pd.Series(index=index, data=index).groupby(index.to_period("M")).head(1).index
    s.loc[first_idx] = True
    return s


def assign_month_labels(index: pd.DatetimeIndex, is_ishares_provider: bool) -> pd.PeriodIndex:
    labels = index.to_period("M")
    if is_ishares_provider:
        return labels

    first_mask = first_trading_day_mask(index).values
    labels_series = pd.Series(labels, index=index)
    labels_series.loc[first_mask] = labels_series.loc[first_mask] - 1
    return pd.PeriodIndex(labels_series.values, freq="M")


def safe_round(s, digits=1):
    return pd.to_numeric(s, errors="coerce").round(digits)


def latest_non_na(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[-1] if not s.empty else np.nan


# =========================================================
# Downloads
# =========================================================
@st.cache_data(show_spinner=False, ttl=3600)
def download_market_data(tickers, start_date, end_date):
    if len(tickers) == 0:
        return pd.DataFrame(), pd.DataFrame()

    raw = yf.download(
        list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Adj Close"].copy() if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"].copy()
        volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        close = raw[[price_col]].copy()
        close.columns = [tickers[0]]
        if "Volume" in raw.columns:
            volume = raw[["Volume"]].copy()
            volume.columns = [tickers[0]]
        else:
            volume = pd.DataFrame(index=close.index, columns=[tickers[0]])

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().dropna(how="all").reindex(columns=list(tickers))

    if volume.empty:
        volume = pd.DataFrame(index=close.index, columns=list(tickers))
    else:
        volume.index = pd.to_datetime(volume.index).tz_localize(None)
        volume = volume.sort_index().reindex(columns=list(tickers))

    return close, volume


@st.cache_data(show_spinner=False, ttl=3600)
def download_metadata(tickers):
    rows = []

    for ticker in tickers:
        tk = yf.Ticker(ticker)
        info = {}
        fast_info = {}

        try:
            info = tk.info or {}
        except Exception:
            info = {}

        try:
            fast_info = dict(tk.fast_info) or {}
        except Exception:
            fast_info = {}

        current_price = info.get("currentPrice")
        if current_price is None:
            current_price = fast_info.get("lastPrice")

        current_shares = info.get("sharesOutstanding")
        if current_shares is None:
            current_shares = fast_info.get("shares")

        current_volume = info.get("volume")
        if current_volume is None:
            current_volume = fast_info.get("lastVolume")

        nav_price = info.get("navPrice")
        net_assets = info.get("netAssets")
        fund_family = info.get("fundFamily")
        long_name = info.get("longName") or info.get("shortName") or ticker
        quote_type = info.get("quoteType")
        currency = info.get("currency")

        premium_discount_pct = np.nan
        if pd.notna(current_price) and pd.notna(nav_price) and nav_price not in [0, None]:
            premium_discount_pct = (current_price / nav_price - 1) * 100

        rows.append(
            {
                "ticker": ticker,
                "long_name": long_name,
                "quote_type": quote_type,
                "fund_family": fund_family,
                "currency": currency,
                "current_price": current_price,
                "current_volume": current_volume,
                "current_shares_outstanding": current_shares,
                "nav_price": nav_price,
                "net_assets": net_assets,
                "premium_discount_pct": premium_discount_pct,
                "is_ishares": is_ishares(fund_family, ticker),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def download_shares_history(tickers, start_date, end_date):
    series_list = []
    coverage_rows = []

    for ticker in tickers:
        tk = yf.Ticker(ticker)
        s = None

        try:
            hist_shares = tk.get_shares_full(start=start_date, end=end_date)
        except Exception:
            hist_shares = None

        if hist_shares is not None:
            if isinstance(hist_shares, pd.DataFrame):
                if hist_shares.shape[1] > 0:
                    s = hist_shares.iloc[:, 0].copy()
            else:
                s = pd.Series(hist_shares).copy()

        if s is not None:
            s = pd.Series(s).dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s = s[~s.index.duplicated(keep="last")].sort_index()
                s.name = ticker
                series_list.append(s)
                coverage_rows.append(
                    {
                        "ticker": ticker,
                        "has_shares_history": True,
                        "shares_start": s.index.min(),
                        "shares_end": s.index.max(),
                        "shares_points": int(s.notna().sum()),
                    }
                )
                continue

        coverage_rows.append(
            {
                "ticker": ticker,
                "has_shares_history": False,
                "shares_start": pd.NaT,
                "shares_end": pd.NaT,
                "shares_points": 0,
            }
        )

    shares_hist = pd.concat(series_list, axis=1) if len(series_list) > 0 else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return shares_hist, coverage


# =========================================================
# Optional historical NAV hook
# =========================================================
def build_historical_nav_placeholder(close: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo/yfinance does not reliably provide daily historical NAV series for ETFs.
    Leave this empty for now. If you later source daily NAV elsewhere,
    replace this function so it returns a DataFrame indexed like `close`.
    """
    return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)


# =========================================================
# Flow engine
# =========================================================
def build_flow_dataset(close, shares_hist, metadata, bucket_map, nav_hist=None):
    if close.empty or shares_hist.empty:
        return {}

    if nav_hist is None:
        nav_hist = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    tickers = [c for c in close.columns if c in shares_hist.columns]
    if len(tickers) == 0:
        return {}

    basis_dict = {}
    shares_dict = {}
    daily_flow_dict = {}
    daily_flow_pct_dict = {}
    flow_5d_dict = {}
    flow_20d_dict = {}
    aum_proxy_dict = {}
    mtd_flow_dict = {}
    month_label_dict = {}
    coverage_rows = []

    for ticker in tickers:
        px = close[ticker].dropna()
        sh = shares_hist[ticker].dropna()
        if px.empty or sh.empty:
            continue

        overlap_start = max(px.index.min(), sh.index.min())
        overlap_end = min(px.index.max(), sh.index.max())
        if overlap_start > overlap_end:
            continue

        px_i = px.loc[overlap_start:overlap_end].copy()
        sh_i = sh.loc[:overlap_end].reindex(px_i.index).ffill()

        hist_nav_i = pd.Series(dtype=float)
        if ticker in nav_hist.columns:
            hist_nav_i = pd.to_numeric(nav_hist[ticker], errors="coerce").dropna()
            if not hist_nav_i.empty:
                hist_nav_i = hist_nav_i.loc[overlap_start:overlap_end]

        df = pd.concat([px_i.rename("price"), sh_i.rename("shares")], axis=1)
        if not hist_nav_i.empty:
            df = pd.concat([df, hist_nav_i.rename("hist_nav")], axis=1)
        df = df.dropna(subset=["price", "shares"])

        if len(df) < 20:
            continue

        if "hist_nav" in df.columns and df["hist_nav"].notna().sum() >= 20:
            df["basis"] = df["hist_nav"].where(df["hist_nav"].notna(), df["price"])
            basis_used = "NAV"
        else:
            df["basis"] = df["price"]
            basis_used = "Price proxy"

        df["shares_delta"] = df["shares"].diff()
        df["aum_proxy"] = df["basis"] * df["shares"]
        df["daily_flow_usd"] = df["basis"] * df["shares_delta"]
        df["daily_flow_pct"] = df["daily_flow_usd"] / df["aum_proxy"].shift(1)
        df["flow_5d_usd"] = df["daily_flow_usd"].rolling(5).sum()
        df["flow_20d_usd"] = df["daily_flow_usd"].rolling(20).sum()
        df["flow_5d_pct"] = df["daily_flow_pct"].rolling(5).sum()
        df["flow_20d_pct"] = df["daily_flow_pct"].rolling(20).sum()

        ff = metadata.loc[metadata["ticker"] == ticker, "fund_family"]
        ff = ff.iloc[0] if not ff.empty else None
        ish = is_ishares(ff, ticker)
        month_labels = assign_month_labels(df.index, ish)
        month_label_series = pd.Series(month_labels.astype(str), index=df.index, name=ticker)

        df["month_label"] = month_labels
        df["mtd_flow_usd"] = df.groupby("month_label")["daily_flow_usd"].cumsum()

        month_start_aum = df.groupby("month_label")["aum_proxy"].transform("first")
        df["mtd_flow_pct"] = df["mtd_flow_usd"] / month_start_aum

        basis_dict[ticker] = df["basis"]
        shares_dict[ticker] = df["shares"]
        daily_flow_dict[ticker] = df["daily_flow_usd"]
        daily_flow_pct_dict[ticker] = df["daily_flow_pct"]
        flow_5d_dict[ticker] = df["flow_5d_usd"]
        flow_20d_dict[ticker] = df["flow_20d_usd"]
        aum_proxy_dict[ticker] = df["aum_proxy"]
        mtd_flow_dict[ticker] = df["mtd_flow_usd"]
        month_label_dict[ticker] = month_label_series

        coverage_rows.append(
            {
                "ticker": ticker,
                "bucket": bucket_map.get(ticker, "Other"),
                "basis_used": basis_used,
                "is_ishares": ish,
                "flow_start": df.index.min(),
                "flow_end": df.index.max(),
                "rows": len(df),
                "latest_assigned_month": str(df["month_label"].iloc[-1]),
            }
        )

    if len(daily_flow_dict) == 0:
        return {}

    basis = pd.concat(basis_dict, axis=1)
    shares = pd.concat(shares_dict, axis=1)
    daily_flow = pd.concat(daily_flow_dict, axis=1)
    daily_flow_pct = pd.concat(daily_flow_pct_dict, axis=1)
    flow_5d = pd.concat(flow_5d_dict, axis=1)
    flow_20d = pd.concat(flow_20d_dict, axis=1)
    aum_proxy = pd.concat(aum_proxy_dict, axis=1)
    mtd_flow = pd.concat(mtd_flow_dict, axis=1)
    month_label_df = pd.concat(month_label_dict, axis=1)

    bucket_series = pd.Series({t: bucket_map.get(t, "Other") for t in daily_flow.columns})
    bucket_daily_flow = group_columns_sum(daily_flow, bucket_series)
    bucket_aum = group_columns_sum(aum_proxy, bucket_series)
    bucket_flow_5d = bucket_daily_flow.rolling(5).sum()
    bucket_flow_20d = bucket_daily_flow.rolling(20).sum()
    bucket_daily_flow_pct = bucket_daily_flow / bucket_aum.shift(1)
    bucket_flow_5d_pct = bucket_daily_flow_pct.rolling(5).sum()
    bucket_flow_20d_pct = bucket_daily_flow_pct.rolling(20).sum()

    return {
        "basis": basis,
        "shares": shares,
        "daily_flow": daily_flow,
        "daily_flow_pct": daily_flow_pct,
        "flow_5d": flow_5d,
        "flow_20d": flow_20d,
        "aum_proxy": aum_proxy,
        "mtd_flow": mtd_flow,
        "month_label": month_label_df,
        "bucket_daily_flow": bucket_daily_flow,
        "bucket_aum": bucket_aum,
        "bucket_flow_5d": bucket_flow_5d,
        "bucket_flow_20d": bucket_flow_20d,
        "bucket_daily_flow_pct": bucket_daily_flow_pct,
        "bucket_flow_5d_pct": bucket_flow_5d_pct,
        "bucket_flow_20d_pct": bucket_flow_20d_pct,
        "coverage": pd.DataFrame(coverage_rows),
    }


# =========================================================
# Summaries
# =========================================================
def make_etf_summary(flow_data, metadata, close, bucket_map):
    if not flow_data:
        return pd.DataFrame()

    rows = []
    tickers = list(flow_data["daily_flow"].columns)

    for ticker in tickers:
        df = pd.concat(
            [
                flow_data["basis"][ticker].rename("basis"),
                flow_data["shares"][ticker].rename("shares"),
                flow_data["daily_flow"][ticker].rename("daily_flow_usd"),
                flow_data["daily_flow_pct"][ticker].rename("daily_flow_pct"),
                flow_data["flow_5d"][ticker].rename("flow_5d_usd"),
                flow_data["flow_20d"][ticker].rename("flow_20d_usd"),
                flow_data["mtd_flow"][ticker].rename("mtd_flow_usd"),
                flow_data["month_label"][ticker].rename("month_label"),
                flow_data["aum_proxy"][ticker].rename("aum_proxy"),
            ],
            axis=1,
        ).dropna(subset=["basis", "shares"])

        if df.empty:
            continue

        last_row = df.iloc[-1]
        meta = metadata.loc[metadata["ticker"] == ticker]
        meta_row = meta.iloc[0] if not meta.empty else pd.Series(dtype=object)
        cov = flow_data["coverage"].loc[flow_data["coverage"]["ticker"] == ticker]
        cov_row = cov.iloc[0] if not cov.empty else pd.Series(dtype=object)

        month_label = last_row["month_label"]
        month_start_aum = df.loc[df["month_label"] == month_label, "aum_proxy"].iloc[0]
        mtd_flow_pct = last_row["mtd_flow_usd"] / month_start_aum if pd.notna(month_start_aum) and month_start_aum != 0 else np.nan

        rows.append(
            {
                "ticker": ticker,
                "name": meta_row.get("long_name", ticker),
                "bucket": bucket_map.get(ticker, "Other"),
                "fund_family": meta_row.get("fund_family", np.nan),
                "basis_used": cov_row.get("basis_used", "Price proxy"),
                "flow_date": df.index.max(),
                "assigned_month": str(month_label),
                "basis": last_row["basis"],
                "current_price": meta_row.get("current_price", np.nan),
                "nav_price": meta_row.get("nav_price", np.nan),
                "premium_discount_pct": meta_row.get("premium_discount_pct", np.nan),
                "shares": last_row["shares"],
                "aum_proxy": last_row["aum_proxy"],
                "flow_1d_usd": last_row["daily_flow_usd"],
                "flow_5d_usd": last_row["flow_5d_usd"],
                "flow_20d_usd": last_row["flow_20d_usd"],
                "flow_20d_pct": last_row["flow_20d_usd"] / df["aum_proxy"].shift(20).iloc[-1] if len(df) > 20 and pd.notna(df["aum_proxy"].shift(20).iloc[-1]) and df["aum_proxy"].shift(20).iloc[-1] != 0 else np.nan,
                "mtd_flow_usd": last_row["mtd_flow_usd"],
                "mtd_flow_pct": mtd_flow_pct,
                "flow_start": cov_row.get("flow_start", pd.NaT),
                "flow_end": cov_row.get("flow_end", pd.NaT),
                "rows": cov_row.get("rows", np.nan),
                "is_ishares": cov_row.get("is_ishares", False),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["flow_20d_pct", "flow_20d_usd"], ascending=[False, False]).reset_index(drop=True)


def make_bucket_summary(etf_summary: pd.DataFrame) -> pd.DataFrame:
    if etf_summary.empty:
        return pd.DataFrame()

    rows = []
    for bucket, g in etf_summary.groupby("bucket"):
        rows.append(
            {
                "bucket": bucket,
                "n_etfs": len(g),
                "aum_proxy": g["aum_proxy"].sum(skipna=True),
                "flow_1d_usd": g["flow_1d_usd"].sum(skipna=True),
                "flow_5d_usd": g["flow_5d_usd"].sum(skipna=True),
                "flow_20d_usd": g["flow_20d_usd"].sum(skipna=True),
                "mtd_flow_usd": g["mtd_flow_usd"].sum(skipna=True),
            }
        )

    out = pd.DataFrame(rows)
    out["flow_1d_pct"] = out["flow_1d_usd"] / out["aum_proxy"]
    out["flow_5d_pct"] = out["flow_5d_usd"] / out["aum_proxy"]
    out["flow_20d_pct"] = out["flow_20d_usd"] / out["aum_proxy"]
    out["mtd_flow_pct"] = out["mtd_flow_usd"] / out["aum_proxy"]
    out = out.sort_values(["flow_20d_pct", "flow_20d_usd"], ascending=[False, False]).reset_index(drop=True)
    return out


def format_etf_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["basis"] = safe_round(out["basis"], 2)
    out["current_price"] = safe_round(out["current_price"], 2)
    out["nav_price"] = safe_round(out["nav_price"], 2)
    out["premium_discount_pct"] = safe_round(out["premium_discount_pct"], 2)
    out["aum_bn"] = safe_round(out["aum_proxy"] / 1e9, 1)
    out["flow_1d_mn"] = safe_round(out["flow_1d_usd"] / 1e6, 1)
    out["flow_5d_mn"] = safe_round(out["flow_5d_usd"] / 1e6, 1)
    out["flow_20d_mn"] = safe_round(out["flow_20d_usd"] / 1e6, 1)
    out["mtd_flow_mn"] = safe_round(out["mtd_flow_usd"] / 1e6, 1)
    out["flow_20d_pct"] = safe_round(out["flow_20d_pct"] * 100, 2)
    out["mtd_flow_pct"] = safe_round(out["mtd_flow_pct"] * 100, 2)
    out["shares_mn"] = safe_round(out["shares"] / 1e6, 1)
    return out[
        [
            "ticker",
            "bucket",
            "basis_used",
            "assigned_month",
            "basis",
            "current_price",
            "nav_price",
            "premium_discount_pct",
            "shares_mn",
            "aum_bn",
            "flow_1d_mn",
            "flow_5d_mn",
            "flow_20d_mn",
            "flow_20d_pct",
            "mtd_flow_mn",
            "mtd_flow_pct",
        ]
    ]


def format_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["aum_bn"] = safe_round(out["aum_proxy"] / 1e9, 1)
    out["flow_1d_mn"] = safe_round(out["flow_1d_usd"] / 1e6, 1)
    out["flow_5d_mn"] = safe_round(out["flow_5d_usd"] / 1e6, 1)
    out["flow_20d_mn"] = safe_round(out["flow_20d_usd"] / 1e6, 1)
    out["mtd_flow_mn"] = safe_round(out["mtd_flow_usd"] / 1e6, 1)
    out["flow_1d_pct"] = safe_round(out["flow_1d_pct"] * 100, 2)
    out["flow_5d_pct"] = safe_round(out["flow_5d_pct"] * 100, 2)
    out["flow_20d_pct"] = safe_round(out["flow_20d_pct"] * 100, 2)
    out["mtd_flow_pct"] = safe_round(out["mtd_flow_pct"] * 100, 2)
    return out[
        [
            "bucket",
            "n_etfs",
            "aum_bn",
            "flow_1d_mn",
            "flow_1d_pct",
            "flow_5d_mn",
            "flow_5d_pct",
            "flow_20d_mn",
            "flow_20d_pct",
            "mtd_flow_mn",
            "mtd_flow_pct",
        ]
    ]


def format_coverage(metadata: pd.DataFrame, shares_coverage: pd.DataFrame, flow_coverage: pd.DataFrame) -> pd.DataFrame:
    out = metadata[["ticker", "fund_family", "nav_price", "premium_discount_pct", "is_ishares"]].merge(
        shares_coverage,
        on="ticker",
        how="left",
    )
    if flow_coverage is not None and not flow_coverage.empty:
        out = out.merge(flow_coverage[["ticker", "basis_used", "flow_start", "flow_end", "rows"]], on="ticker", how="left")
    else:
        out["basis_used"] = np.nan
        out["flow_start"] = pd.NaT
        out["flow_end"] = pd.NaT
        out["rows"] = np.nan

    out["nav_price"] = safe_round(out["nav_price"], 2)
    out["premium_discount_pct"] = safe_round(out["premium_discount_pct"], 2)
    return out.sort_values(["has_shares_history", "ticker"], ascending=[False, True]).reset_index(drop=True)


# =========================================================
# Plots
# =========================================================
def plot_bucket_cumulative_flows(bucket_daily_flow: pd.DataFrame):
    if bucket_daily_flow.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    (bucket_daily_flow.fillna(0).cumsum() / 1e9).plot(ax=ax)
    ax.set_title("Bucket cumulative ETF flows")
    ax.set_ylabel("USD bn")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def plot_selected_etf(close_s: pd.Series, flow_s: pd.Series, ticker: str):
    if close_s.empty or flow_s.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    (close_s / close_s.dropna().iloc[0]).plot(ax=ax, label="Price index")
    (flow_s.fillna(0).cumsum() / 1e6).plot(ax=ax, label="Cumulative flow (USD mn)")
    ax.set_title(f"{ticker}: price vs cumulative flow")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# UI
# =========================================================
st.title("ETF Flow Monitor")
st.caption(
    "Flow logic = daily change in shares outstanding × end-of-day basis. "
    "Monthly assignment follows the iShares vs non-iShares convention. "
    "Because Yahoo does not reliably provide historical daily NAV series via yfinance, "
    "historical flow basis currently falls back to price proxy unless you later plug in a NAV history source."
)

with st.sidebar:
    st.header("Inputs")
    preset_name = st.selectbox("Universe preset", list(PRESET_UNIVERSES.keys()), index=4)
    use_custom_only = st.checkbox("Use custom tickers only", value=False)
    custom_ticker_text = st.text_area("Custom tickers (comma-separated)", value="")
    start_date = pd.Timestamp(st.date_input("Start date", value=pd.Timestamp("2024-01-01")))
    end_date = pd.Timestamp(st.date_input("End date", value=pd.Timestamp.today().normalize()))
    drop_no_shares = st.checkbox("Drop ETFs with no shares history", value=True)

preset_tickers = PRESET_UNIVERSES[preset_name]
custom_tickers = parse_ticker_text(custom_ticker_text)
tickers = custom_tickers if use_custom_only else list(dict.fromkeys(preset_tickers + custom_tickers))

if len(tickers) == 0:
    st.warning("Please provide at least one ticker.")
    st.stop()

bucket_map = DEFAULT_BUCKET_MAP.copy()
for t in tickers:
    if t not in bucket_map:
        bucket_map[t] = "Other"

fetch_start = min(TRUE_FLOW_MIN_START, start_date - pd.Timedelta(days=31))
fetch_end = end_date + pd.Timedelta(days=1)

with st.spinner("Downloading ETF data..."):
    close, volume = download_market_data(tuple(tickers), str(fetch_start.date()), str(fetch_end.date()))
    metadata = download_metadata(tuple(tickers))
    shares_hist, shares_coverage = download_shares_history(tuple(tickers), str(TRUE_FLOW_MIN_START.date()), str(fetch_end.date()))
    nav_hist = build_historical_nav_placeholder(close)

if close.empty:
    st.error("Price download failed. Check tickers or date range.")
    st.stop()

if drop_no_shares and not shares_coverage.empty:
    keep = shares_coverage.loc[shares_coverage["has_shares_history"], "ticker"].tolist()
    close = close.reindex(columns=keep)
    metadata = metadata[metadata["ticker"].isin(keep)].reset_index(drop=True)
    shares_hist = shares_hist.reindex(columns=keep) if not shares_hist.empty else shares_hist

flow_data = build_flow_dataset(close, shares_hist, metadata, bucket_map, nav_hist=nav_hist)
etf_summary = make_etf_summary(flow_data, metadata, close, bucket_map)
bucket_summary = make_bucket_summary(etf_summary)
coverage_table = format_coverage(metadata, shares_coverage, flow_data.get("coverage", pd.DataFrame()) if flow_data else pd.DataFrame())

if etf_summary.empty:
    st.warning("No ETF flow dataset could be built from the selected tickers. Usually this means Yahoo did not provide usable shares history.")
    st.dataframe(coverage_table, use_container_width=True, hide_index=True)
    st.stop()

latest_flow_date = pd.to_datetime(etf_summary["flow_date"]).max()
top_bucket = bucket_summary.iloc[0]["bucket"] if not bucket_summary.empty else "N/A"
bottom_bucket = bucket_summary.iloc[-1]["bucket"] if not bucket_summary.empty else "N/A"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest flow date", str(latest_flow_date.date()) if pd.notna(latest_flow_date) else "N/A")
m2.metric("ETFs with flow data", f"{len(etf_summary)}")
m3.metric("Top 20d flow bucket", top_bucket)
m4.metric("Bottom 20d flow bucket", bottom_bucket)

st.markdown("### Bucket flow summary")
st.caption("Sorted by 20d flow %. Top = strongest inflow bucket. Bottom = strongest outflow bucket.")
st.dataframe(format_bucket_summary(bucket_summary), use_container_width=True, hide_index=True)

st.markdown("### ETF flow summary")
st.caption("Basis used is NAV only if you later plug in historical NAV data. With Yahoo alone, this will usually show Price proxy.")
st.dataframe(format_etf_summary(etf_summary), use_container_width=True, hide_index=True)

st.markdown("### Cumulative bucket flows")
plot_bucket_cumulative_flows(flow_data["bucket_daily_flow"])

selected_ticker = st.selectbox("Select ETF", list(etf_summary["ticker"]))
st.markdown(f"### {selected_ticker}: price vs cumulative flow")
plot_selected_etf(
    close[selected_ticker].dropna(),
    flow_data["daily_flow"][selected_ticker].dropna(),
    selected_ticker,
)

with st.expander("Coverage / method"):
    st.markdown(
        """
        **Daily flow logic**
        - Daily flow = end-of-day basis × (shares today − shares yesterday)
        - Basis = historical NAV if available, otherwise price proxy

        **Monthly assignment**
        - iShares: daily flows are assigned to the same calendar month
        - Non-iShares: the first trading day of a month is assigned to the previous month
        - This matches the T=0 vs T+1 convention you described

        **Important**
        - Yahoo/yfinance gives current NAV fields for many ETFs, but not a robust historical daily NAV series
        - So with Yahoo alone, historical flow will usually run on a price proxy basis even though the formula itself is the right ETF-flow formula
        """
    )
    st.dataframe(coverage_table, use_container_width=True, hide_index=True)
