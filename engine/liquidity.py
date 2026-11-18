"""MSCI-style size and liquidity tests for BVB stocks, per quarterly review.

Inputs come from the bvb_data raw cache: one frame per symbol with
date, close, volume (unadjusted), plus the trading calendar.

Definitions (session record, GIMI methodology Nov 2026):
  daily traded value  = close x volume of that day (RON; ATVR is a ratio, no FX needed)
  MTVR (month)        = median daily traded value x days traded / month-end free-float cap
  ATVR 12m / 3m       = mean of the last 12 / 3 monthly MTVRs, x 12 (annualised)
  FOT (quarter)       = days traded / possible trading days in that quarter
  Review cut-offs     = last business day of Mar, Jun, Sep, Dec
Size tests are in USD and use one RON/USD rate per review.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REVIEW_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class Thresholds:
    full_mcap_usd_m: float = 4343.0
    ff_mcap_usd_m: float = 2172.0
    min_fif: float = 0.15
    low_fif_multiplier: float = 1.8      # FIF below min_fif: float cap must reach 1.8x
    atvr_12m: float = 0.15
    atvr_3m: float = 0.15
    fot_3m: float = 0.80                 # in each of the last fot_quarters quarters
    fot_quarters: int = 4
    min_companies: int = 3
    reviews_needed: int = 8

    @classmethod
    def from_csv(cls, path) -> "Thresholds":
        """Last row of config/thresholds.csv (columns = field names)."""
        row = pd.read_csv(path).iloc[-1]
        kw = {f: type(getattr(cls(), f))(row[f]) for f in cls.__dataclass_fields__ if f in row}
        return cls(**kw)


def review_months(start, end) -> list[pd.Period]:
    """Review cut-off months between two dates, inclusive."""
    months = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")
    return [m for m in months if m.month in REVIEW_MONTHS]

def _for_month(value, month):
    """value is a number (same every month) or a Series indexed by month."""
    if isinstance(value, pd.Series):
        return value.get(month, np.nan)
    return value


def shares_by_month(history: pd.DataFrame, months, default: float) -> pd.Series:
    """Share count valid at the end of each month.

    history: rows with effective_date (ex-date) and shares_outstanding for one
    company. Each row applies from its effective_date on. Months before the first
    row take the first row's count. Empty history -> `default` for every month."""
    months = pd.PeriodIndex(months, freq="M")
    if history is None or history.empty:
        return pd.Series(float(default), index=months)
    h = history.assign(effective_date=pd.to_datetime(history["effective_date"]))
    h = h.sort_values("effective_date")
    out = []
    for m in months:
        valid = h[h["effective_date"] <= m.to_timestamp(how="end")]
        row = valid.iloc[-1] if len(valid) else h.iloc[0]
        out.append(float(row["shares_outstanding"]))
    return pd.Series(out, index=months)

def monthly_liquidity(bars: pd.DataFrame, calendar, ff_shares,
                      median_over: str = "traded") -> pd.DataFrame:
    """One row per calendar month: median value, days traded, possible days,
    month-end close and free-float cap (RON), MTVR.

    ff_shares: one number for every month, or a pd.Series indexed by month
    (pd.Period "M") when the share count changed during the period.

    median_over="traded": median over days the stock traded (default).
    median_over="all": zero-value days on the calendar count in the median.
    Month-end close = last traded close on or before the month's last session.
    """
    if median_over not in ("traded", "all"):
        raise ValueError("median_over must be 'traded' or 'all'")
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    b = bars[["date", "close", "volume"]].copy()
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    b = b.drop_duplicates("date", keep="last")
    d = pd.DataFrame({"date": cal}).merge(b, on="date", how="left")
    d["volume"] = d["volume"].fillna(0)
    d["traded"] = d["volume"] > 0
    d["value"] = np.where(d["traded"], d["close"] * d["volume"], 0.0)
    d.loc[~d["traded"], "close"] = np.nan
    d["close"] = d["close"].ffill()
    d["month"] = d["date"].dt.to_period("M")

    rows = []
    for month, g in d.groupby("month", sort=True):
        vals = g.loc[g["traded"], "value"] if median_over == "traded" else g["value"]
        median = float(vals.median()) if len(vals) else 0.0
        days = int(g["traded"].sum())
        close_end = g["close"].iloc[-1]
        ffs = _for_month(ff_shares, month)
        ff_cap = close_end * ffs if pd.notna(close_end) and pd.notna(ffs) else np.nan
        mtvr = median * days / ff_cap if pd.notna(ff_cap) and ff_cap > 0 else np.nan
        rows.append({"month": month, "median_value_ron": median, "days_traded": days,
                     "possible_days": len(g), "close_end_ron": close_end,
                     "ff_cap_ron": ff_cap, "mtvr": mtvr})
    return pd.DataFrame(rows)


def review_liquidity(monthly: pd.DataFrame, review: pd.Period, th: Thresholds = Thresholds()) -> dict:
    """ATVR 12m, ATVR 3m and quarterly FOTs at one review cut-off month.
    Needs 12 months of history; otherwise the ATVR 12m is NaN and the test fails."""
    m = monthly[monthly["month"] <= review].sort_values("month")
    last12, last3 = m.tail(12), m.tail(3)
    atvr12 = last12["mtvr"].mean() * 12 if len(last12) == 12 and last12["mtvr"].notna().all() else np.nan
    atvr3 = last3["mtvr"].mean() * 12 if len(last3) == 3 and last3["mtvr"].notna().all() else np.nan
    fots = []
    for q in range(th.fot_quarters):
        chunk = m.iloc[len(m) - 3 * (q + 1): len(m) - 3 * q] if len(m) >= 3 * (q + 1) else m.iloc[0:0]
        possible = chunk["possible_days"].sum()
        fots.append(chunk["days_traded"].sum() / possible if len(chunk) == 3 and possible else np.nan)
    fot_min = min(fots) if not any(pd.isna(f) for f in fots) else np.nan
    # Daily value needed for ATVR 12m = threshold, keeping days traded and float cap as they are:
    # mean(median_i * days_i / cap_i) * 12 = t  ->  scale today's medians by t / atvr12.
    avg_median = last12["median_value_ron"].mean() if len(last12) else np.nan
    needed = avg_median * th.atvr_12m / atvr12 if pd.notna(atvr12) and atvr12 > 0 else np.nan
    return {
        "review": review, "atvr_12m": atvr12, "atvr_3m": atvr3, "fot_min_q": fot_min,
        "fot_by_quarter": fots, "avg_median_value_ron": avg_median,
        "needed_median_value_ron": needed,
        "value_gap_ron_per_day": max(0.0, needed - avg_median) if pd.notna(needed) else np.nan,
        "pass_atvr_12m": bool(pd.notna(atvr12) and atvr12 >= th.atvr_12m),
        "pass_atvr_3m": bool(pd.notna(atvr3) and atvr3 >= th.atvr_3m),
        "pass_fot": bool(pd.notna(fot_min) and fot_min >= th.fot_3m),
    }


def size_tests(close_ron: float, shares_total: float, fif: float, ron_per_usd: float,
               th: Thresholds = Thresholds()) -> dict:
    """USD size tests at one review. A FIF below th.min_fif raises the float-cap bar."""
    full = close_ron * shares_total / ron_per_usd / 1e6
    ff = full * fif
    ff_needed = th.ff_mcap_usd_m * (th.low_fif_multiplier if fif < th.min_fif else 1.0)
    return {
        "full_mcap_usd_m": full, "ff_mcap_usd_m": ff, "ff_needed_usd_m": ff_needed,
        "ff_gap_usd_m": max(0.0, ff_needed - ff), "full_gap_usd_m": max(0.0, th.full_mcap_usd_m - full),
        "pass_full": bool(full >= th.full_mcap_usd_m), "pass_ff": bool(ff >= ff_needed),
    }


def evaluate_company(symbol: str, bars: pd.DataFrame, calendar, shares_total, fif: float,
                     fx: dict, reviews, th: Thresholds = Thresholds(),
                     median_over: str = "traded") -> pd.DataFrame:
    """All tests for one company at each review. fx maps review Period -> RON per USD.
    A review without an FX rate gets NaN size results and fails the size tests.
    shares_total: one number, or a Series by month (see shares_by_month) when the
    share count changed. The free-float share (FIF) is held constant (flagged
    simplification)."""
    monthly = monthly_liquidity(bars, calendar, shares_total * fif, median_over)
    rows = []
    for rv in reviews:
        rv = pd.Period(rv, "M")
        liq = review_liquidity(monthly, rv, th)
        mrow = monthly[monthly["month"] == rv]
        close = mrow["close_end_ron"].iloc[0] if len(mrow) else np.nan
        rate = fx.get(rv)
        shares_rv = _for_month(shares_total, rv)
        if rate and pd.notna(close) and pd.notna(shares_rv):
            size = size_tests(close, shares_rv, fif, rate, th)
        else:
            size = {"full_mcap_usd_m": np.nan, "ff_mcap_usd_m": np.nan, "ff_needed_usd_m": np.nan,
                    "ff_gap_usd_m": np.nan, "full_gap_usd_m": np.nan, "pass_full": False, "pass_ff": False}
        row = {"symbol": symbol, **liq, **size, "fif": fif}
        row["review"] = str(rv)      # text, ca să meargă în JSON și în Streamlit
        flags = ["pass_full", "pass_ff", "pass_atvr_12m", "pass_atvr_3m", "pass_fot"]
        row["tests_passed"] = sum(row[f] for f in flags)
        row["pass_all"] = row["tests_passed"] == len(flags)
        rows.append(row)
    return pd.DataFrame(rows)


def country_streak(results: pd.DataFrame, th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Per review: companies passing everything, whether the market qualifies
    (at least th.min_companies), and the consecutive streak ending at that review."""
    per = (results.groupby("review")["pass_all"].sum().astype(int)
           .rename("companies_passing").reset_index().sort_values("review"))
    per["market_passes"] = per["companies_passing"] >= th.min_companies
    streak, run = [], 0
    for ok in per["market_passes"]:
        run = run + 1 if ok else 0
        streak.append(run)
    per["streak"] = streak
    per["streak_complete"] = per["streak"] >= th.reviews_needed
    return per.reset_index(drop=True)
