"""Mocked tests for bvb_data. No network. Run: pytest"""
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

from ingest import bvb_data as bd

BUC = ZoneInfo("Europe/Bucharest")

# First 10 bars of the real TLV capture (2025-07-08 ..), ajust=1.
TLV = {
    "t": [1751932800, 1752019200, 1752105600, 1752192000, 1752451200,
          1752537600, 1752624000, 1752710400, 1752796800, 1753056000],
    "o": [22.3122, 22.3563, 22.6209, 22.9590, 22.9296, 23.4587, 23.7674, 24.2083, 24.7152, 24.4180],
    "h": [22.4445, 22.6797, 23.0031, 22.9737, 23.4881, 23.7674, 23.7968, 24.9948, 24.7501, 24.4705],
    "l": [22.2828, 22.3563, 22.6209, 22.8120, 22.9296, 23.4587, 23.6351, 24.0335, 24.2083, 24.0335],
    "c": [22.3416, 22.5768, 22.9296, 22.9296, 23.4146, 23.7674, 23.7380, 24.7152, 24.4705, 24.4530],
    "v": [474958, 294563, 196173, 256938, 1136525, 614776, 658701, 1544702, 325726, 665721],
    "s": "ok", "nextTime": None,
}


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def make_client(responses, **kw):
    sleeps = []
    kw.setdefault("min_interval", 0)
    kw.setdefault("backoff", 0)
    session = FakeSession(responses)
    client = bd.BVBClient(token="TESTTOKEN", session=session, sleep=sleeps.append, **kw)
    return client, session, sleeps


def bars(dates, close=10.0, volume=1000):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": close, "high": close, "low": close,
        "close": close, "volume": volume,
    }).astype({"volume": "int64"})


class ParseTests(unittest.TestCase):
    def test_parse_fixture(self):
        df = bd.parse_history(TLV)
        self.assertEqual(list(df.columns), bd.COLUMNS)
        self.assertEqual(len(df), 10)
        self.assertEqual(df["date"].iloc[0], pd.Timestamp("2025-07-08"))
        self.assertEqual(df["date"].iloc[-1], pd.Timestamp("2025-07-21"))
        self.assertAlmostEqual(df["close"].iloc[0], 22.3416)
        self.assertEqual(int(df["volume"].iloc[4]), 1136525)
        self.assertEqual(df["volume"].dtype, "int64")

    def test_no_data_is_empty(self):
        df = bd.parse_history({"s": "no_data", "nextTime": 123})
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), bd.COLUMNS)

    def test_error_status_raises(self):
        with self.assertRaises(bd.BVBError):
            bd.parse_history({"s": "error", "errmsg": "bad symbol"})

    def test_length_mismatch_raises(self):
        bad = dict(TLV, v=TLV["v"][:5])
        with self.assertRaises(bd.BVBError):
            bd.parse_history(bad)

    def test_non_midnight_stamp_raises(self):
        bad = dict(TLV, t=[t + 3600 for t in TLV["t"]])
        with self.assertRaises(bd.BVBError):
            bd.parse_history(bad)


class ClientTests(unittest.TestCase):
    def test_missing_token(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BVB_WAPI_TOKEN", None)
            with self.assertRaises(bd.BVBAuthError):
                bd.BVBClient(session=FakeSession([]))

    def test_request_and_headers(self):
        client, session, _ = make_client([FakeResponse(200, TLV)])
        df = client.history_daily("TLV", date(2025, 7, 8), date(2025, 7, 21))
        self.assertEqual(len(df), 10)
        url, params = session.calls[0]
        self.assertEqual(url, "https://wapi.bvb.ro/api/history")
        self.assertEqual(params["from"], 1751932800 - 1)
        self.assertEqual(params["to"], 1753056000)
        self.assertEqual((params["rs"], params["ajust"], params["currencyCode"]), ("1D", 0, "RON"))
        self.assertEqual(session.headers["X-SK-API"], "TESTTOKEN")
        self.assertEqual(session.headers["Origin"], "https://m.bvb.ro")
        self.assertTrue(session.headers["Referer"].startswith("https://m.bvb.ro"))
        self.assertIn("Mozilla", session.headers["User-Agent"])

    def test_adjusted_flag(self):
        client, session, _ = make_client([FakeResponse(200, TLV)])
        client.history_daily("TLV", date(2025, 7, 8), date(2025, 7, 21), adjusted=True)
        self.assertEqual(session.calls[0][1]["ajust"], 1)

    def test_retry_then_success(self):
        client, session, sleeps = make_client(
            [FakeResponse(429), FakeResponse(401), FakeResponse(200, TLV)])
        df = client.history_daily("TLV", date(2025, 7, 8), date(2025, 7, 21))
        self.assertEqual(len(df), 10)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_network_error_retried(self):
        import requests
        client, session, _ = make_client([requests.ConnectionError("boom"), FakeResponse(200, TLV)])
        self.assertEqual(len(client.history_daily("TLV", date(2025, 7, 8), date(2025, 7, 21))), 10)
        self.assertEqual(len(session.calls), 2)

    def test_persistent_401_is_auth_error(self):
        client, session, _ = make_client([FakeResponse(401, text="Authorization has been denied")],
                                         max_retries=2)
        with self.assertRaises(bd.BVBAuthError) as cm:
            client.history_daily("TLV", date(2025, 7, 8), date(2025, 7, 21))
        self.assertEqual(len(session.calls), 3)
        self.assertNotIn("TESTTOKEN", str(cm.exception))

    def test_404_not_retried(self):
        client, session, _ = make_client([FakeResponse(404, text="No HTTP resource")])
        with self.assertRaises(bd.BVBError):
            client.history_daily("TLV", date(2025, 7, 8), date(2025, 7, 21))
        self.assertEqual(len(session.calls), 1)

    def test_bars_outside_range_dropped(self):
        client, _, _ = make_client([FakeResponse(200, TLV)])
        df = client.history_daily("TLV", date(2025, 7, 10), date(2025, 7, 16))
        self.assertEqual(df["date"].min(), pd.Timestamp("2025-07-10"))
        self.assertEqual(df["date"].max(), pd.Timestamp("2025-07-16"))
        self.assertEqual(len(df), 5)

    def test_truncation_detected(self):
        # 10 bars returned, cap is 5, oldest bar is after the window start: bars were cut off
        client, _, _ = make_client([FakeResponse(200, TLV)], countback=5)
        with self.assertRaises(bd.BVBError):
            client.history_daily("TLV", date(2025, 7, 1), date(2025, 7, 21))

    def test_cap_hit_but_start_covered_is_fine(self):
        # API ignored 'from' and returned older bars too: window start is covered
        client, _, _ = make_client([FakeResponse(200, TLV)], countback=5)
        df = client.history_daily("TLV", date(2025, 7, 10), date(2025, 7, 21))
        self.assertEqual(len(df), 8)    # 10,11,14,15,16,17,18,21 July

    def test_long_range_is_chunked(self):
        client, session, _ = make_client([FakeResponse(200, {"s": "no_data"})])
        client.history_daily("TLV", date(2024, 1, 1), date(2026, 11, 13))
        self.assertGreaterEqual(len(session.calls), 3)
        starts = [c[1]["from"] + 1 for c in session.calls]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(session.calls[-1][1]["to"], bd._utc_ts(date(2026, 11, 13)))

    def test_symbol_info(self):
        client, session, _ = make_client([FakeResponse(200, {"name": "TLV", "session": "0700-1830"})])
        self.assertEqual(client.symbol_info("TLV")["session"], "0700-1830")
        self.assertEqual(session.calls[0][0], "https://wapi.bvb.ro/api/symbols")


class SessionTests(unittest.TestCase):
    def test_monday_morning_gives_friday(self):
        self.assertEqual(bd.last_complete_session(datetime(2026, 11, 16, 10, 0, tzinfo=BUC)), date(2026, 11, 13))

    def test_monday_evening_gives_monday(self):
        self.assertEqual(bd.last_complete_session(datetime(2026, 11, 16, 19, 30, tzinfo=BUC)), date(2026, 11, 16))

    def test_sunday_gives_friday(self):
        self.assertEqual(bd.last_complete_session(datetime(2026, 11, 15, 12, 0, tzinfo=BUC)), date(2026, 11, 13))

    def test_tuesday_after_close_before_cutoff(self):
        self.assertEqual(bd.last_complete_session(datetime(2026, 11, 17, 18, 45, tzinfo=BUC)), date(2026, 11, 16))

    def test_naive_rejected(self):
        with self.assertRaises(ValueError):
            bd.last_complete_session(datetime(2026, 11, 16, 10, 0))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = bd.DailyCache(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        df = bd.parse_history(TLV)
        self.cache.upsert("TLV", df)
        back = self.cache.load("TLV")
        self.assertEqual(len(back), 10)
        self.assertAlmostEqual(back["close"].iloc[3], 22.9296)
        self.assertEqual(int(back["volume"].iloc[0]), 474958)
        self.assertEqual(back["date"].iloc[0], pd.Timestamp("2025-07-08"))

    def test_upsert_counts_mismatches_and_appends(self):
        first = bars(["2025-07-08", "2025-07-09", "2025-07-10"])
        self.cache.upsert("X", first)
        second = bars(["2025-07-09", "2025-07-10", "2025-07-11"])
        second.loc[second["date"] == "2025-07-10", "volume"] = 999       # revised
        merged, mism = self.cache.upsert("X", second)
        self.assertEqual(mism, 1)
        self.assertEqual(len(merged), 4)
        self.assertEqual(int(merged.loc[merged["date"] == "2025-07-10", "volume"].iloc[0]), 999)
        self.assertEqual(merged["date"].is_monotonic_increasing, True)

    def test_identical_overlap_no_mismatch(self):
        a = bars(["2025-07-08", "2025-07-09"])
        self.cache.upsert("X", a)
        _, mism = self.cache.upsert("X", a)
        self.assertEqual(mism, 0)

    def test_load_missing_is_empty(self):
        self.assertTrue(self.cache.load("NOPE").empty)

    def test_trading_days_prefers_calendar_symbol(self):
        self.cache.upsert("BET", bars(["2025-07-08", "2025-07-09"]))
        self.cache.upsert("TLV", bars(["2025-07-08", "2025-07-10"]))
        days = bd.trading_days(self.cache, ["TLV"])
        self.assertEqual([d.date() for d in days], [date(2025, 7, 8), date(2025, 7, 9)])

    def test_trading_days_fallback_union(self):
        self.cache.upsert("TLV", bars(["2025-07-08", "2025-07-10"]))
        self.cache.upsert("SNP", bars(["2025-07-09", "2025-07-10"]))
        days = bd.trading_days(self.cache, ["TLV", "SNP"])
        self.assertEqual(len(days), 3)


class UpdateTests(unittest.TestCase):
    class FakeClient:
        def __init__(self, frame):
            self.frame, self.calls = frame, []

        def history_daily(self, symbol, start, end, adjusted=False, currency="RON"):
            self.calls.append((symbol, start, end, adjusted))
            f = self.frame
            return f[(f["date"] >= pd.Timestamp(start)) & (f["date"] <= pd.Timestamp(end))].reset_index(drop=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = bd.DailyCache(self.tmp.name)
        self.now = datetime(2026, 11, 16, 19, 30, tzinfo=BUC)       # Monday after close
        days = pd.bdate_range("2024-01-02", "2026-11-16")
        self.frame = bars(days)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_sync_fetches_from_start_raw(self):
        client = self.FakeClient(self.frame)
        r = bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        self.assertEqual(client.calls[0], ("TLV", date(2024, 1, 1), date(2026, 11, 16), False))
        self.assertEqual(r["new_rows"], len(self.frame))
        self.assertEqual(r["last"], date(2026, 11, 16))

    def test_second_sync_uses_overlap(self):
        client = self.FakeClient(self.frame)
        bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        client.calls.clear()
        later = datetime(2026, 11, 17, 19, 30, tzinfo=BUC)
        client.frame = bars(pd.bdate_range("2024-01-02", "2026-11-17"))
        r = bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=later)
        self.assertEqual(client.calls[0][1], date(2026, 11, 16) - timedelta(days=bd.OVERLAP_DAYS))
        self.assertEqual(r["new_rows"], 1)
        self.assertEqual(r["mismatches"], 0)

    def test_up_to_date_still_refreshes_overlap_only(self):
        client = self.FakeClient(self.frame)
        bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        client.calls.clear()
        r = bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        self.assertEqual(r["new_rows"], 0)
        self.assertEqual(len(client.calls), 1)

    def test_earlier_start_triggers_full_refetch(self):
        client = self.FakeClient(self.frame)
        bd.update_symbol(client, self.cache, "TLV", date(2025, 1, 1), now=self.now)
        client.calls.clear()
        bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        self.assertEqual(client.calls[0][1], date(2024, 1, 1))

    def test_revision_reported(self):
        client = self.FakeClient(self.frame)
        bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        revised = self.frame.copy()
        revised.loc[revised["date"] == "2026-11-12", "volume"] = 5
        client.frame = revised
        r = bd.update_symbol(client, self.cache, "TLV", date(2024, 1, 1), now=self.now)
        self.assertEqual(r["mismatches"], 1)


class AdjustmentTests(unittest.TestCase):
    def test_detects_step(self):
        dates = pd.bdate_range("2025-01-01", periods=6)
        raw = bars(dates, close=10.0)
        adj = raw.copy()
        adj.loc[:2, "close"] = 9.0          # earlier bars rescaled by 0.9
        res = bd.adj_ratio_changes(raw, adj)
        self.assertEqual(res["rows"], 6)
        self.assertEqual(len(res["steps"]), 1)
        d, before, after = res["steps"][0]
        self.assertEqual(d, dates[3].date())
        self.assertAlmostEqual(before, 0.9)
        self.assertAlmostEqual(after, 1.0)
        self.assertFalse(res["volume_differs"])

    def test_constant_ratio_no_steps(self):
        raw = bars(pd.bdate_range("2025-01-01", periods=4))
        self.assertEqual(bd.adj_ratio_changes(raw, raw)["steps"], [])

    def test_volume_difference_flagged(self):
        raw = bars(pd.bdate_range("2025-01-01", periods=4))
        adj = raw.copy()
        adj.loc[0, "volume"] = 1
        self.assertTrue(bd.adj_ratio_changes(raw, adj)["volume_differs"])


class DotenvTests(unittest.TestCase):
    def test_loads_and_does_not_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, ".env")
            with open(p, "w", encoding="utf-8-sig") as f:
                f.write("# c\nBVB_TEST_A='quoted'\nBVB_TEST_B=plain==x\nBVB_TEST_C=new\n")
            with mock.patch.dict(os.environ, {"BVB_TEST_C": "existing"}, clear=False):
                os.environ.pop("BVB_TEST_A", None)
                os.environ.pop("BVB_TEST_B", None)
                self.assertTrue(bd.load_dotenv(p))
                self.assertEqual(os.environ["BVB_TEST_A"], "quoted")
                self.assertEqual(os.environ["BVB_TEST_B"], "plain==x")
                self.assertEqual(os.environ["BVB_TEST_C"], "existing")
                os.environ.pop("BVB_TEST_A")
                os.environ.pop("BVB_TEST_B")

    def test_missing_file(self):
        self.assertFalse(bd.load_dotenv("/nonexistent/.env"))


if __name__ == "__main__":
    unittest.main()
