# engine/metrics.py — funcții pure: intră DataFrame-uri, ies DataFrame-uri
import pandas as pd


def free_float_shares(issuers: pd.DataFrame, shareholders: pd.DataFrame) -> pd.DataFrame:
    """FF% = 1 - suma deținerilor strategice. Returnează issuers + ff_shares."""
    strat = (shareholders[shareholders.strategic == 1]
             .groupby("ticker").pct.sum().rename("strategic_pct"))
    out = issuers.set_index("ticker").join(strat)
    out["strategic_pct"] = out["strategic_pct"].fillna(0)
    out["ff_shares"] = out.shares_outstanding * (1 - out.strategic_pct / 100)
    return out.reset_index()


def monthly_table(trades: pd.DataFrame, fx: pd.DataFrame, issuers_ff: pd.DataFrame) -> pd.DataFrame:
    """Tabel lunar per ticker: capitalizări (USD), MTVR, ATVR 12L/3L."""
    t = pd.merge_asof(trades.sort_values("date"), fx.sort_values("date"), on="date")
    t["value_usd"] = t.value_ron / t.ron_per_usd
    t["month"] = t.date.dt.to_period("M")
    traded = t[t.value_usd > 0]
    m = traded.groupby(["ticker", "month"]).agg(
        median_value_usd=("value_usd", "median"),
        days_traded=("value_usd", "size"))
    end = t.groupby(["ticker", "month"]).agg(close_ron=("close_ron", "last"),
                                             fx_end=("ron_per_usd", "last"))
    m = end.join(m).reset_index()
    m[["median_value_usd", "days_traded"]] = m[["median_value_usd", "days_traded"]].fillna(0)
    m = m.merge(issuers_ff[["ticker", "shares_outstanding", "ff_shares"]], on="ticker")
    m["full_mcap_usd"] = m.close_ron * m.shares_outstanding / m.fx_end
    m["ff_mcap_usd"] = m.close_ron * m.ff_shares / m.fx_end
    m["mtvr"] = m.median_value_usd * m.days_traded / m.ff_mcap_usd
    m = m.sort_values(["ticker", "month"])
    g = m.groupby("ticker").mtvr
    m["atvr_12m"] = g.transform(lambda s: s.rolling(12).mean() * 12)
    m["atvr_3m"] = g.transform(lambda s: s.rolling(3).mean() * 12)
    return m


def run_tests(m: pd.DataFrame, th: pd.Series) -> pd.DataFrame:
    """Cele 3 teste MSCI-style, praguri din config/thresholds.csv (milioane USD)."""
    m = m.copy()
    m["pass_full"] = m.full_mcap_usd >= th.full_mcap_usd_m * 1e6
    m["pass_ff"] = m.ff_mcap_usd >= th.ff_mcap_usd_m * 1e6
    m["pass_atvr"] = m.atvr_12m >= th.atvr_min
    m["tests_passed"] = m[["pass_full", "pass_ff", "pass_atvr"]].sum(axis=1)
    m["pass_all"] = m.tests_passed == 3
    return m
