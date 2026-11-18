# engine/pipeline.py — leagă totul: încarcă datele, aplică scenariul, rulează testele
#
# Prețuri și volume, în ordinea priorității:
#   1. data/raw/bvb/<SIMBOL>_RON_raw.csv  (cache-ul din ingest/bvb_data.py, nerajustat)
#   2. data/trades.csv                     (exporturi manuale; close_ron, volume_shares)
# Calendarul de tranzacționare: BET_RON_raw.csv dacă există, altfel zilele în care
# a tranzacționat cel puțin o acțiune.
# Numărul de acțiuni: data/shares_history.csv (dacă există) dă numărul valabil în
# fiecare lună; companiile care lipsesc de acolo folosesc valoarea din issuers.csv.
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import liquidity as liq
from .free_float import free_float_table
from .gap_list import build as build_gap_list
from .scenarios import apply as apply_levers

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "bvb"
CALENDAR_SYMBOL = "BET"
SHARES_HISTORY_COLS = ["ticker", "effective_date", "shares_outstanding", "event", "source"]

def _load_bars(issuers: pd.DataFrame) -> dict:
    """{simbol: DataFrame[date, close, volume]} pentru simbolurile din issuers.csv (+ BET)."""
    bars = {}
    for sym in list(issuers["ticker"]) + [CALENDAR_SYMBOL]:
        p = RAW_DIR / f"{sym}_RON_raw.csv"
        if p.exists():
            bars[sym] = pd.read_csv(p, parse_dates=["date"])[["date", "close", "volume"]]
    trades_path = ROOT / "data" / "trades.csv"
    if trades_path.exists():
        t = pd.read_csv(trades_path, parse_dates=["date"])
        for sym, g in t.groupby("ticker"):
            if sym not in bars:  # cache-ul automat are prioritate
                bars[sym] = (g.rename(columns={"close_ron": "close", "volume_shares": "volume"})
                             [["date", "close", "volume"]].reset_index(drop=True))
    return bars


def _calendar(bars: dict) -> pd.DatetimeIndex:
    if CALENDAR_SYMBOL in bars and not bars[CALENDAR_SYMBOL].empty:
        return pd.DatetimeIndex(bars[CALENDAR_SYMBOL]["date"]).sort_values()
    days = set()
    for sym, b in bars.items():
        if sym != CALENDAR_SYMBOL:
            days.update(b.loc[b["volume"] > 0, "date"])
    return pd.DatetimeIndex(sorted(days))

def _load_shares_history() -> pd.DataFrame:
    p = ROOT / "data" / "shares_history.csv"
    if not p.exists():
        return pd.DataFrame(columns=SHARES_HISTORY_COLS)
    h = pd.read_csv(p)
    missing = [c for c in SHARES_HISTORY_COLS if c not in h.columns]
    if missing:
        raise ValueError(f"shares_history.csv: lipsesc coloanele {missing}")
    h["effective_date"] = pd.to_datetime(h["effective_date"])
    return h


def shares_for(sym: str, history: pd.DataFrame, months, current: float):
    """Numărul de acțiuni pentru un simbol: o valoare constantă (dacă nu are istoric)
    sau o serie pe luni. Ultimul rând din istoric trebuie să fie egal cu issuers.csv."""
    h = history[history["ticker"] == sym]
    if h.empty:
        return float(current)
    latest = h.sort_values("effective_date")["shares_outstanding"].iloc[-1]
    if int(latest) != int(current):
        raise ValueError(f"{sym}: ultimul rând din shares_history.csv ({int(latest)}) "
                         f"diferă de issuers.csv ({int(current)})")
    return liq.shares_by_month(h, months, current)
def _load():
    th = liq.Thresholds.from_csv(ROOT / "config" / "thresholds.csv")
    issuers = pd.read_csv(ROOT / "data" / "issuers.csv")
    shareholders = pd.read_csv(ROOT / "data" / "shareholders.csv")
    fx_path = ROOT / "data" / "fx_usd.csv"
    fx = (pd.read_csv(fx_path, parse_dates=["date"]) if fx_path.exists()
          else pd.DataFrame(columns=["date", "ron_per_usd"]))
    return th, issuers, shareholders, _load_bars(issuers), fx, _load_shares_history()


def fx_at_reviews(fx: pd.DataFrame, reviews) -> dict:
    """Cursul BNR din ultima zi publicată până la sfârșitul fiecărei luni de review."""
    f = fx.sort_values("date")
    out = {}
    for rv in reviews:
        upto = f[f["date"] <= rv.to_timestamp(how="end")]
        if not upto.empty:
            out[rv] = float(upto["ron_per_usd"].iloc[-1])
    return out


def complete_reviews(calendar: pd.DatetimeIndex) -> list:
    """Lunile de review (mar., iun., sep., dec.) încheiate în datele disponibile."""
    if len(calendar) == 0:
        return []
    last = calendar.max()
    reviews = liq.review_months(calendar.min(), last)
    return [rv for rv in reviews if rv.to_timestamp(how="end").normalize() <= last
            or rv < pd.Period(last, "M")]


def _empty(th, last_fx):
    e = pd.DataFrame()
    return SimpleNamespace(tests=e, gap_list=build_gap_list(e), history=e, streak=e,
                           review=None, n_pass_all=0, country_streak=0, fx=last_fx, th=th)


def run(levers: dict | None = None) -> SimpleNamespace:
    th, issuers, shareholders, bars, fx, shares_history = _load()
    bars, shareholders, fx = apply_levers(levers or {}, bars, shareholders, fx)
    last_fx = float(fx.sort_values("date")["ron_per_usd"].iloc[-1]) if not fx.empty else float("nan")

    symbols = [s for s in issuers["ticker"] if s in bars]
    if not symbols:
        return _empty(th, last_fx)

    calendar = _calendar(bars)
    reviews = complete_reviews(calendar)
    if not reviews:
        return _empty(th, last_fx)
    fx_rv = fx_at_reviews(fx, reviews)

    ff = free_float_table(shareholders, tickers=symbols).set_index("ticker")
    shares = issuers.set_index("ticker")["shares_outstanding"]

    months = pd.period_range(calendar.min(), calendar.max(), freq="M")
    frames = [liq.evaluate_company(sym, bars[sym], calendar,
                                   shares_for(sym, shares_history, months, float(shares[sym])),
                                   float(ff.loc[sym, "fif"]), fx_rv, reviews, th)
              for sym in symbols]
    history = pd.concat(frames, ignore_index=True)
    streak = liq.country_streak(history, th)

    latest_review = str(reviews[-1])
    latest = history[history["review"] == latest_review].reset_index(drop=True)
    return SimpleNamespace(
        tests=latest, gap_list=build_gap_list(latest), history=history, streak=streak,
        review=latest_review, n_pass_all=int(latest["pass_all"].sum()),
        country_streak=int(streak["streak"].iloc[-1]), fx=last_fx, th=th)
