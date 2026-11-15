# engine/scenarios.py — pârghiile: modifică datele de intrare, apoi motorul rerulează
# Pârghii legitime: volum (market making, retail), activitate Pilon II, vânzare stat, curs.
import pandas as pd

DEFAULT_LEVERS = {"volume": 0.0, "pension": 0.0, "state_sell": 0.0, "fx": 0.0}


def apply(levers: dict, issuers: pd.DataFrame, shareholders: pd.DataFrame,
          trades: pd.DataFrame, fx: pd.DataFrame):
    lv = {**DEFAULT_LEVERS, **(levers or {})}
    sh, t, f = shareholders.copy(), trades.copy(), fx.copy()

    # volum + activitate Pilon II: scalează valoarea tranzacționată
    scale = (1 + lv["volume"]) * (1 + lv["pension"])
    if not t.empty:
        t["value_ron"] = t["value_ron"] * scale
        t["volume_shares"] = t["volume_shares"] * scale

    # vânzare stat: x% din acțiuni trec din deținere strategică în free float
    sell_pp = lv["state_sell"] * 100
    if sell_pp and not sh.empty:
        is_state = (sh["type"] == "state") & (sh["strategic"] == 1)
        sh.loc[is_state, "pct"] = (sh.loc[is_state, "pct"] - sell_pp).clip(lower=0)

    # șoc de curs RON/USD
    if not f.empty:
        f["ron_per_usd"] = f["ron_per_usd"] * (1 + lv["fx"])

    return issuers, sh, t, f
