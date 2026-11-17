# engine/scenarios.py — pârghiile: modifică datele de intrare, apoi motorul rerulează
# Pârghii legitime: volum (market making, retail), activitate Pilon II, vânzare stat, curs.
import pandas as pd

DEFAULT_LEVERS = {"volume": 0.0, "pension": 0.0, "state_sell": 0.0, "fx": 0.0}

# Cota fondurilor Pilon II din rulajul BVB (ian.–iul. 2025, ASF). O schimbare de y% în
# activitatea lor mută rulajul total cu 0,23 × y. Presupunere editabilă.
PENSION_TRADING_SHARE = 0.23


def apply(levers: dict, bars: dict, shareholders: pd.DataFrame, fx: pd.DataFrame):
    """bars: {simbol: DataFrame[date, close, volume]}. Returnează copii modificate."""
    lv = {**DEFAULT_LEVERS, **(levers or {})}

    # volum + activitate Pilon II: scalează volumul (deci și valoarea) zilelor tranzacționate
    scale = (1 + lv["volume"]) * (1 + PENSION_TRADING_SHARE * lv["pension"])
    new_bars = {}
    for sym, b in bars.items():
        b = b.copy()
        b["volume"] = b["volume"] * scale
        new_bars[sym] = b

    # vânzare stat: x puncte procentuale trec de la stat în free float, la fiecare emitent
    sh = shareholders.copy()
    sell_pp = lv["state_sell"] * 100
    if sell_pp and not sh.empty:
        is_state = sh["type"].astype(str).str.strip().str.lower().isin({"state", "government"})
        # o singură dată per emitent, din cel mai mare pachet de stat (nu din fiecare rând)
        biggest = sh[is_state].groupby("ticker")["pct"].idxmax()
        sh.loc[biggest, "pct"] = (sh.loc[biggest, "pct"] - sell_pp).clip(lower=0)

    # șoc de curs RON/USD (afectează doar testele de mărime, în USD)
    f = fx.copy()
    if not f.empty:
        f["ron_per_usd"] = f["ron_per_usd"] * (1 + lv["fx"])

    return new_bars, sh, f
