# engine/gap_list.py — lista de deficit: ce îi lipsește fiecărei companii ca să treacă
import pandas as pd


def build(latest: pd.DataFrame, th: pd.Series) -> pd.DataFrame:
    """Pentru companiile care nu trec încă toate testele: deficitul de lichiditate și de free float."""
    if latest.empty:
        return pd.DataFrame()
    g = latest.copy()
    # valoarea mediană zilnică necesară pentru ATVR 15%: 0.15 * CapFF / 252
    g["target_daily_usd"] = th.atvr_min * g.ff_mcap_usd / 252
    g["daily_now_usd"] = g.median_value_usd
    g["liquidity_gap_usd"] = (g.target_daily_usd - g.daily_now_usd).clip(lower=0)
    g["ff_gap_usd"] = (th.ff_mcap_usd_m * 1e6 - g.ff_mcap_usd).clip(lower=0)
    g = g[(g.tests_passed >= 1) & (~g.pass_all)]
    cols = ["ticker", "tests_passed", "atvr_12m", "full_mcap_usd", "ff_mcap_usd",
            "daily_now_usd", "target_daily_usd", "liquidity_gap_usd", "ff_gap_usd"]
    return g[cols].sort_values(["tests_passed", "liquidity_gap_usd"],
                               ascending=[False, True]).reset_index(drop=True)
