# engine/pipeline.py — leagă totul: încarcă datele, aplică scenariul, rulează testele
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import metrics
from .scenarios import apply as apply_levers
from .gap_list import build as build_gap_list

ROOT = Path(__file__).resolve().parents[1]


def _load():
    th = pd.read_csv(ROOT / "config" / "thresholds.csv").iloc[-1]
    issuers = pd.read_csv(ROOT / "data" / "issuers.csv")
    shareholders = pd.read_csv(ROOT / "data" / "shareholders.csv")
    trades = pd.read_csv(ROOT / "data" / "trades.csv", parse_dates=["date"])
    fx_path = ROOT / "data" / "fx_usd.csv"
    if fx_path.exists():
        fx = pd.read_csv(fx_path, parse_dates=["date"])
    else:
        fx = pd.DataFrame(columns=["date", "ron_per_usd"])
    return th, issuers, shareholders, trades, fx


def run(levers: dict | None = None) -> SimpleNamespace:
    th, issuers, shareholders, trades, fx = _load()
    issuers, shareholders, trades, fx = apply_levers(levers or {}, issuers, shareholders, trades, fx)

    last_fx = float(fx.sort_values("date").ron_per_usd.iloc[-1]) if not fx.empty else float("nan")

    if trades.empty or fx.empty or issuers.empty:
        empty = pd.DataFrame()
        return SimpleNamespace(tests=empty, gap_list=empty, monthly=empty,
                               n_pass_all=0, country_streak=0, fx=last_fx, th=th)

    ff = metrics.free_float_shares(issuers, shareholders)
    m = metrics.monthly_table(trades, fx, ff)
    m = metrics.run_tests(m, th)

    latest = m[m.month == m.month.max()].reset_index(drop=True)
    gap = build_gap_list(latest, th)

    # streak: evaluări consecutive în care >= min_companies companii trec toate testele
    by_month = m.groupby("month").pass_all.sum().sort_index()
    streak = 0
    for n_pass in by_month:
        streak = streak + 1 if n_pass >= th.min_companies else 0

    return SimpleNamespace(tests=latest, gap_list=gap, monthly=m,
                           n_pass_all=int(latest.pass_all.sum()),
                           country_streak=int(streak), fx=last_fx, th=th)
