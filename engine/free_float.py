# engine/free_float.py — free float și FIF (foreign inclusion factor), în stil MSCI
#
# Intrare: data/shareholders.csv, un rând per acționar numit:
#   ticker, holder, pct (% din totalul acțiunilor), type, strategic, source, as_of
# `type` decide clasificarea (MSCI Free Float Data Methodology, mar. 2026):
#   strategici:   government, company, treasury, bank, employee, officer, private_equity, individual
#   free float:   fund, insurer, broker, pension_pillar2, retail_other
# Coloana `strategic` e doar pentru citire; dacă nu se potrivește cu `type`, primiți eroare.
# Publicul larg nu trebuie listat: free float = 100% minus toate deținerile strategice.
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

STRATEGIC_TYPES = {
    "government", "company", "treasury", "bank", "employee", "officer",
    "private_equity", "individual",
}
FREE_FLOAT_TYPES = {"fund", "insurer", "broker", "pension_pillar2", "retail_other"}
HOLDER_TYPES = STRATEGIC_TYPES | FREE_FLOAT_TYPES
ALIASES = {  # denumiri folosite în ghid -> tipurile de mai sus
    "state": "government", "insider": "officer", "public": "retail_other",
    "pension": "pension_pillar2", "pension_fund": "pension_pillar2",
}


def round_fif(ff: float) -> float:
    """Rotunjire FIF: peste 25% la cel mai apropiat 2,5%; între 5% și 25% la 0,5%;
    sub 5% la 0,1%. Jumătățile se rotunjesc în sus. Decimal evită erorile de float."""
    if not 0 <= ff <= 1:
        raise ValueError(f"free float trebuie să fie o fracție în [0, 1], nu {ff}")
    pct = Decimal(str(ff)) * 100
    if pct > 25:
        step = Decimal("2.5")
    elif pct >= 5:
        step = Decimal("0.5")
    else:
        step = Decimal("0.1")
    rounded = (pct / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step
    return float(min(rounded, Decimal(100)) / 100)


def normalize_types(holders: pd.DataFrame) -> pd.DataFrame:
    h = holders.copy()
    h["type"] = h["type"].astype(str).str.strip().str.lower().replace(ALIASES)
    unknown = sorted(set(h["type"]) - HOLDER_TYPES)
    if unknown:
        raise ValueError(f"valori `type` necunoscute: {unknown}. Permise: {sorted(HOLDER_TYPES)}")
    if "strategic" in h.columns:
        expected = h["type"].isin(STRATEGIC_TYPES).astype(int)
        given = pd.to_numeric(h["strategic"], errors="coerce")
        bad = h[given.notna() & (given != expected)]
        if not bad.empty:
            raise ValueError("coloana `strategic` contrazice `type` la: "
                             + "; ".join(f"{r.ticker}/{r.holder} ({r.type})" for r in bad.itertuples()))
    return h


def free_float_table(holders: pd.DataFrame, tickers=None) -> pd.DataFrame:
    """Per ticker: strategic_pct, ff_raw (fracție) și fif (rotunjit).
    Tickerii din `tickers` fără niciun acționar strategic listat primesc FF 100%."""
    h = normalize_types(holders)
    totals = h.groupby("ticker")["pct"].sum()
    over = totals[totals > 100 + 1e-6]
    if not over.empty:
        raise ValueError(f"deținerile depășesc 100% la: {over.to_dict()}")
    strat = h[h["type"].isin(STRATEGIC_TYPES)].groupby("ticker")["pct"].sum()
    index = sorted(set(totals.index) | set(tickers or []))
    out = pd.DataFrame({"strategic_pct": strat.reindex(index, fill_value=0.0)})
    out.index.name = "ticker"
    out["ff_raw"] = 1 - out["strategic_pct"] / 100
    out["fif"] = out["ff_raw"].map(round_fif)
    return out.reset_index()
