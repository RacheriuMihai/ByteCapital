# tests/test_engine.py — motorul MSCI-style: free float, ATVR, FOT, mărime, streak, pipeline
import math

import numpy as np
import pandas as pd
import pytest

from engine import free_float as ffm
from engine import liquidity as liq

CAL = pd.bdate_range("2025-01-01", "2025-12-31")  # 261 zile, 12 luni complete
DEC = pd.Period("2025-12", "M")


def bars(close=4.0, volume=300_000, days=CAL):
    return pd.DataFrame({"date": days, "close": close, "volume": volume})


# ---------------------------------------------------------------- free float / FIF

@pytest.mark.parametrize("ff, fif", [
    (0.3374, 0.325), (0.3376, 0.35), (0.2512, 0.25), (0.25, 0.25), (0.1237, 0.125),
    (0.1224, 0.12), (0.05, 0.05), (0.0333, 0.033), (0.0049, 0.005), (1.0, 1.0), (0.0, 0.0),
])
def test_round_fif(ff, fif):
    assert ffm.round_fif(ff) == pytest.approx(fif)


def test_round_fif_rejects_percent():
    with pytest.raises(ValueError):
        ffm.round_fif(33.7)


def test_free_float_table_aliases_and_missing_ticker():
    h = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA"], "holder": ["Stat", "Pilon II", "Fond"],
                      "pct": [60.0, 10.0, 5.0], "type": ["state", "pension_pillar2", "fund"],
                      "strategic": [1, 0, 0]})
    t = ffm.free_float_table(h, tickers=["AAA", "BBB"]).set_index("ticker")
    assert t.loc["AAA", "ff_raw"] == pytest.approx(0.40)   # doar statul e strategic
    assert t.loc["AAA", "fif"] == pytest.approx(0.40)
    assert t.loc["BBB", "fif"] == pytest.approx(1.0)       # niciun strategic listat


def test_free_float_table_catches_bad_input():
    base = {"ticker": ["AAA"], "holder": ["X"], "pct": [50.0]}
    with pytest.raises(ValueError, match="necunoscute"):
        ffm.free_float_table(pd.DataFrame({**base, "type": ["oligarh"]}))
    with pytest.raises(ValueError, match="contrazice"):
        ffm.free_float_table(pd.DataFrame({**base, "type": ["fund"], "strategic": [1]}))
    with pytest.raises(ValueError, match="100%"):
        ffm.free_float_table(pd.DataFrame({"ticker": ["AAA", "AAA"], "holder": ["A", "B"],
                                           "pct": [70.0, 40.0], "type": ["company", "bank"]}))


# ---------------------------------------------------------------- lichiditate

def test_atvr_sanity_rule_of_thumb():
    # FF cap 2 mld. RON, valoare 1,2 mil. RON/zi în fiecare zi -> ATVR ~15-16%
    m = liq.monthly_liquidity(bars(), CAL, ff_shares=500_000_000)
    expected = 12 * np.mean([1.2e6 * d / 2e9 for d in m["days_traded"]])
    r = liq.review_liquidity(m, DEC)
    assert r["atvr_12m"] == pytest.approx(expected)
    assert r["atvr_12m"] == pytest.approx(0.1566, abs=0.001)
    assert r["pass_atvr_12m"] and r["pass_atvr_3m"] and r["pass_fot"]
    assert r["value_gap_ron_per_day"] == 0


def test_more_free_float_dilutes_atvr():
    m1 = liq.monthly_liquidity(bars(), CAL, ff_shares=500_000_000)
    m2 = liq.monthly_liquidity(bars(), CAL, ff_shares=700_000_000)
    a1 = liq.review_liquidity(m1, DEC)["atvr_12m"]
    a2 = liq.review_liquidity(m2, DEC)["atvr_12m"]
    assert a2 == pytest.approx(a1 * 500 / 700)


def test_fot_fails_when_trading_every_other_day():
    half = bars(days=CAL[::2])
    m = liq.monthly_liquidity(half, CAL, ff_shares=500_000_000)
    r = liq.review_liquidity(m, DEC)
    assert r["fot_min_q"] == pytest.approx(0.5, abs=0.03)
    assert not r["pass_fot"]


def test_median_over_all_counts_zero_days():
    b = bars()
    b.loc[b.index % 3 == 0, "volume"] = 0           # o zi din trei fără tranzacții
    m_tr = liq.monthly_liquidity(b, CAL, 5e8, median_over="traded")
    m_all = liq.monthly_liquidity(b, CAL, 5e8, median_over="all")
    assert (m_tr["median_value_ron"] == 1.2e6).all()
    assert (m_all["median_value_ron"] == 1.2e6).all()   # mediana nu se mișcă la 1/3 zerouri
    b.loc[b.index % 3 != 2, "volume"] = 0           # acum 2/3 zerouri
    m_all = liq.monthly_liquidity(b, CAL, 5e8, median_over="all")
    assert (m_all["median_value_ron"] == 0).all()


def test_month_end_close_carried_from_last_trade():
    b = bars()
    b.loc[b["date"] == CAL[CAL.month == 1][-1], "volume"] = 0      # ultima zi din ian. fără trade
    b.loc[b["date"] == CAL[CAL.month == 1][-2], "close"] = 5.0
    m = liq.monthly_liquidity(b, CAL, 1)
    assert m.loc[0, "close_end_ron"] == 5.0


def test_atvr_needs_twelve_months():
    m = liq.monthly_liquidity(bars(), CAL, 5e8)
    r = liq.review_liquidity(m, pd.Period("2025-09", "M"))
    assert math.isnan(r["atvr_12m"]) and not r["pass_atvr_12m"]
    assert r["pass_atvr_3m"]


def test_value_gap_brings_atvr_to_threshold():
    m = liq.monthly_liquidity(bars(volume=150_000), CAL, 5e8)   # jumătate din volum
    r = liq.review_liquidity(m, DEC)
    assert not r["pass_atvr_12m"]
    scaled = m.assign(mtvr=m["mtvr"] * r["needed_median_value_ron"] / r["avg_median_value_ron"])
    assert liq.review_liquidity(scaled, DEC)["atvr_12m"] == pytest.approx(0.15)


# ---------------------------------------------------------------- mărime

def test_size_tests_and_low_fif_multiplier():
    s = liq.size_tests(close_ron=4.0, shares_total=1e9, fif=0.5, ron_per_usd=1.0)
    assert s["full_mcap_usd_m"] == pytest.approx(4000) and not s["pass_full"]
    s = liq.size_tests(4.0, 1e9, 0.5, ron_per_usd=0.9)
    assert s["pass_full"] and s["ff_mcap_usd_m"] == pytest.approx(4000 / 0.9 * 0.5) and s["pass_ff"]
    low = liq.size_tests(30.0, 1e9, fif=0.10, ron_per_usd=1.0)          # FF cap 3.000 mil.
    assert low["ff_needed_usd_m"] == pytest.approx(2172 * 1.8) and not low["pass_ff"]
    ok = liq.size_tests(30.0, 1e9, fif=0.15, ron_per_usd=1.0)
    assert ok["ff_needed_usd_m"] == pytest.approx(2172) and ok["pass_ff"]


def test_thresholds_from_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("valid_from,full_mcap_usd_m,ff_mcap_usd_m,min_fif,low_fif_multiplier,atvr_12m,"
                 "atvr_3m,fot_3m,fot_quarters,min_companies,reviews_needed,source\n"
                 "2026-11-01,4343,2172,0.15,1.8,0.15,0.15,0.8,4,3,8,x\n")
    th = liq.Thresholds.from_csv(p)
    assert th == liq.Thresholds()


# ---------------------------------------------------------------- streak

def test_country_streak():
    reviews = ["2025-03", "2025-06", "2025-09", "2025-12", "2026-03"]
    passing = [3, 4, 2, 3, 3]
    rows = [{"review": r, "symbol": f"S{i}", "pass_all": i < n}
            for r, n in zip(reviews, passing) for i in range(5)]
    s = liq.country_streak(pd.DataFrame(rows))
    assert list(s["companies_passing"]) == passing
    assert list(s["streak"]) == [1, 2, 0, 1, 2]


def test_review_months():
    assert [str(p) for p in liq.review_months("2025-01-15", "2025-12-01")] == \
        ["2025-03", "2025-06", "2025-09", "2025-12"]


# ---------------------------------------------------------------- pipeline cap-coadă

@pytest.fixture
def repo(tmp_path, monkeypatch):
    from engine import pipeline
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw" / "bvb").mkdir(parents=True)
    (tmp_path / "config" / "thresholds.csv").write_text(
        "valid_from,full_mcap_usd_m,ff_mcap_usd_m,min_fif,low_fif_multiplier,atvr_12m,atvr_3m,"
        "fot_3m,fot_quarters,min_companies,reviews_needed,source\n"
        "2026-11-01,4343,2172,0.15,1.8,0.15,0.15,0.8,4,3,8,test\n")
    # 3 companii care trec tot + una cu volum prea mic (trece mărimea, pică ATVR)
    pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD"], "name": list("ABCD"),
                  "shares_outstanding": [2e9] * 4, "pension_pct": [10.0] * 4}
                 ).to_csv(tmp_path / "data" / "issuers.csv", index=False)
    pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD"], "holder": ["Stat"] * 4,
                  "pct": [50.0] * 4, "type": ["state"] * 4, "strategic": [1] * 4,
                  "source": ["test"] * 4, "as_of": ["2025-12-31"] * 4}
                 ).to_csv(tmp_path / "data" / "shareholders.csv", index=False)
    days = pd.bdate_range("2025-01-01", "2026-01-15")
    for sym, vol in [("AAA", 800_000), ("BBB", 800_000), ("CCC", 800_000), ("DDD", 100_000)]:
        bars(close=5.0, volume=vol, days=days).to_csv(
            tmp_path / "data" / "raw" / "bvb" / f"{sym}_RON_raw.csv", index=False)
    pd.DataFrame({"date": days, "ron_per_usd": 1.0}).to_csv(tmp_path / "data" / "fx_usd.csv", index=False)
    pd.DataFrame(columns=["date", "ticker", "close_ron", "volume_shares", "value_ron"]
                 ).to_csv(tmp_path / "data" / "trades.csv", index=False)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "RAW_DIR", tmp_path / "data" / "raw" / "bvb")
    return pipeline


def test_pipeline_end_to_end(repo):
    res = repo.run()
    assert res.review == "2025-12"                     # ian. 2026 nu e trimestru încheiat
    t = res.tests.set_index("symbol")
    assert t.loc["AAA", "full_mcap_usd_m"] == pytest.approx(10_000)
    assert t.loc["AAA", "ff_mcap_usd_m"] == pytest.approx(5_000)
    assert res.n_pass_all == 3
    assert list(res.gap_list["symbol"]) == ["DDD"]
    assert res.gap_list.loc[0, "value_gap_ron_per_day"] > 0
    # 2025-03..09 nu au 12 luni de ATVR, deci streak-ul începe abia în decembrie
    assert list(res.streak["streak"]) == [0, 0, 0, 1]
    assert res.country_streak == 1


def test_pipeline_levers(repo):
    base = repo.run()
    sold = repo.run({"state_sell": 0.20})              # statul vinde 20 pp -> FF 70%
    a0 = base.tests.set_index("symbol").loc["AAA"]
    a1 = sold.tests.set_index("symbol").loc["AAA"]
    assert a1["fif"] == pytest.approx(0.70)
    assert a1["atvr_12m"] == pytest.approx(a0["atvr_12m"] * 50 / 70)
    more = repo.run({"volume": 1.0}).tests.set_index("symbol").loc["DDD"]
    assert more["atvr_12m"] == pytest.approx(base.tests.set_index("symbol").loc["DDD", "atvr_12m"] * 2)


def test_pipeline_manual_trades_fallback(repo):
    root = repo.ROOT
    (root / "data" / "raw" / "bvb" / "DDD_RON_raw.csv").unlink()
    days = pd.bdate_range("2025-01-01", "2026-01-15")
    pd.DataFrame({"date": days, "ticker": "DDD", "close_ron": 5.0, "volume_shares": 100_000,
                  "value_ron": 500_000.0}).to_csv(root / "data" / "trades.csv", index=False)
    assert "DDD" in set(repo.run().tests["symbol"])


def test_api_records_have_no_nan():
    from api.main import _records
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [[0.9, float("nan")], [1.0]]})
    assert _records(df) == [{"a": 1.0, "b": [0.9, None]}, {"a": None, "b": [1.0]}]


# ---------------------------------------------------------------- istoric nr actiuni

def test_shares_by_month():
    h = pd.DataFrame({"effective_date": ["2025-01-01", "2025-04-08"],
                      "shares_outstanding": [100, 300]})
    months = pd.period_range("2024-12", "2025-06", freq="M")
    s = liq.shares_by_month(h, months, default=999)
    # dec. 2024 e inainte de primul rand -> ia primul rand; aprilie e dupa data ex -> 300
    assert list(s) == [100, 100, 100, 100, 300, 300, 300]
    empty = liq.shares_by_month(h.iloc[0:0], months, default=999)
    assert (empty == 999).all()


def test_split_keeps_free_float_cap_continuous():
    # split 3:1 pe 8 aprilie 2025: pretul scade de la 30 la 10, volumul se tripleaza
    b = bars(close=30.0, volume=100_000)
    after = b["date"] >= "2025-04-08"
    b.loc[after, "close"] = 10.0
    b.loc[after, "volume"] = 300_000
    months = pd.period_range("2025-01", "2025-12", freq="M")
    h = pd.DataFrame({"effective_date": ["2025-01-01", "2025-04-08"],
                      "shares_outstanding": [100_000_000, 300_000_000]})
    shares = liq.shares_by_month(h, months, default=300_000_000)
    good = liq.monthly_liquidity(b, CAL, shares * 0.5)
    assert good["ff_cap_ron"].round().nunique() == 1         # 1,5 mld. RON în fiecare lună
    # cu nr de acțiuni de azi pentru tot istoricul, ian.-mar. ar iesi de 3 ori prea mari
    wrong = liq.monthly_liquidity(b, CAL, 300_000_000 * 0.5)
    assert wrong.loc[0, "ff_cap_ron"] == pytest.approx(3 * good.loc[0, "ff_cap_ron"])
    assert wrong.loc[0, "mtvr"] == pytest.approx(good.loc[0, "mtvr"] / 3)


def test_size_tests_use_share_count_of_the_review_month():
    h = pd.DataFrame({"effective_date": ["2025-01-01", "2025-07-01"],
                      "shares_outstanding": [1e9, 2e9]})
    months = pd.period_range("2025-01", "2025-12", freq="M")
    shares = liq.shares_by_month(h, months, default=2e9)
    fx = {pd.Period("2025-06", "M"): 1.0, pd.Period("2025-12", "M"): 1.0}
    r = liq.evaluate_company("AAA", bars(close=5.0), CAL, shares, 0.5, fx,
                             ["2025-06", "2025-12"]).set_index("review")
    assert r.loc["2025-06", "full_mcap_usd_m"] == pytest.approx(5_000)
    assert r.loc["2025-12", "full_mcap_usd_m"] == pytest.approx(10_000)


def test_pipeline_reads_shares_history(repo):
    (repo.ROOT / "data" / "shares_history.csv").write_text(
        "ticker,effective_date,shares_outstanding,event,source\n"
        "AAA,2025-01-01,1000000000,start,test\n"
        "AAA,2025-07-01,2000000000,actiuni gratuite,test\n")
    h = repo.run().history.set_index(["symbol", "review"])
    assert h.loc[("AAA", "2025-06"), "full_mcap_usd_m"] == pytest.approx(5_000)
    assert h.loc[("AAA", "2025-12"), "full_mcap_usd_m"] == pytest.approx(10_000)
    assert h.loc[("BBB", "2025-06"), "full_mcap_usd_m"] == pytest.approx(10_000)  # fara istoric


def test_pipeline_rejects_history_that_disagrees_with_issuers(repo):
    (repo.ROOT / "data" / "shares_history.csv").write_text(
        "ticker,effective_date,shares_outstanding,event,source\n"
        "AAA,2025-01-01,1500000000,start,test\n")              # issuers.csv spune 2e9
    with pytest.raises(ValueError, match="diferă de issuers.csv"):
        repo.run()


def test_pipeline_rejects_history_without_columns(repo):
    (repo.ROOT / "data" / "shares_history.csv").write_text("ticker,date,shares\nAAA,2025-01-01,1\n")
    with pytest.raises(ValueError, match="lipsesc coloanele"):
        repo.run()
