# tests/test_metrics.py — verificări de sanitate pentru motor
import pandas as pd
import pytest

from engine import metrics


def _fixture():
    """Companie sintetică: FF cap 2 mld. USD, mediană 1,2 mil. USD/zi -> ATVR ~15%."""
    dates = pd.bdate_range("2025-01-01", periods=252)  # ~12 luni x 21 zile
    trades = pd.DataFrame({
        "date": dates,
        "ticker": "TST",
        "close_ron": 4.0,
        "volume_shares": 300_000,
        "value_ron": 1_200_000.0,
    })
    fx = pd.DataFrame({"date": dates, "ron_per_usd": 1.0})
    issuers = pd.DataFrame({"ticker": ["TST"], "name": ["Test SA"],
                            "shares_outstanding": [1_000_000_000], "pension_pct": [5.0]})
    shareholders = pd.DataFrame({
        "ticker": ["TST"], "holder": ["Statul roman"], "pct": [50.0],
        "type": ["state"], "strategic": [1],
        "source": ["test"], "as_of": ["2025-12-31"],
    })
    th = pd.Series({"full_mcap_usd_m": 3937, "ff_mcap_usd_m": 1969,
                    "atvr_min": 0.15, "min_companies": 3, "reviews_needed": 8})
    return trades, fx, issuers, shareholders, th


def test_free_float():
    trades, fx, issuers, shareholders, th = _fixture()
    ff = metrics.free_float_shares(issuers, shareholders)
    assert ff.ff_shares.iloc[0] == 500_000_000  # 50% strategic -> FF 50%


def test_atvr_sanity():
    # median 1.2M USD/zi * 252 zile / FF cap 2 mld. USD = ~15.1% ATVR
    trades, fx, issuers, shareholders, th = _fixture()
    ff = metrics.free_float_shares(issuers, shareholders)
    m = metrics.monthly_table(trades, fx, ff)
    atvr = m.atvr_12m.dropna().iloc[-1]
    assert atvr == pytest.approx(0.1512, abs=0.005)
    assert m.ff_mcap_usd.iloc[-1] == pytest.approx(2e9)
    assert m.full_mcap_usd.iloc[-1] == pytest.approx(4e9)


def test_three_tests():
    trades, fx, issuers, shareholders, th = _fixture()
    ff = metrics.free_float_shares(issuers, shareholders)
    m = metrics.run_tests(metrics.monthly_table(trades, fx, ff), th)
    last = m.dropna(subset=["atvr_12m"]).iloc[-1]
    # full 4 mld >= 3.937 mld, FF 2 mld >= 1.969 mld, ATVR 15.1% >= 15% -> 3/3
    assert bool(last.pass_all)


def test_fewer_strategic_means_more_ff_less_atvr():
    trades, fx, issuers, shareholders, th = _fixture()
    shareholders2 = shareholders.copy()
    shareholders2["pct"] = 30.0  # statul vinde 20pp -> FF creste
    ff1 = metrics.free_float_shares(issuers, shareholders)
    ff2 = metrics.free_float_shares(issuers, shareholders2)
    m1 = metrics.monthly_table(trades, fx, ff1)
    m2 = metrics.monthly_table(trades, fx, ff2)
    assert m2.ff_mcap_usd.iloc[-1] > m1.ff_mcap_usd.iloc[-1]
    # fara volum suplimentar, ATVR scade cand FF creste
    assert m2.atvr_12m.dropna().iloc[-1] < m1.atvr_12m.dropna().iloc[-1]
