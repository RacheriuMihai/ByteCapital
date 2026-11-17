"""BVB wapi client and cached daily-bar loader.

Pulls daily OHLCV from https://wapi.bvb.ro/api/history, caches RAW
(unadjusted, ajust=0) bars per symbol as CSV, and keeps them current.

Why raw: MSCI daily traded value is shares x close of that day. Back-adjusted
history (ajust=1) is rewritten whenever a corporate action lands, so it cannot
be cached append-only. Raw prices and volumes are what was actually traded.

Setup:
    pip install pandas requests tzdata
    copy .env.example to .env and put your token in BVB_WAPI_TOKEN
    python ingest/bvb_data.py validate
    python ingest/bvb_data.py sync
    python ingest/bvb_data.py adjcheck SNP

The token is a credential. Keep it in .env (git-ignored), never in code.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests

BASE_URL = "https://wapi.bvb.ro/api"
SITE_ORIGIN = "https://m.bvb.ro"
DEFAULT_START = date(2024, 1, 1)      # oldest MSCI review in the streak uses Jan-Dec 2024
OVERLAP_DAYS = 10                     # re-fetched on every sync to catch revisions
CALENDAR_SYMBOL = "BET"               # trading calendar; falls back to union of shares
COUNTBACK = 400                       # larger than any 1-year window of daily bars
WINDOW_DAYS = 365                     # calendar days requested per call
CLOSE_CUTOFF = dtime(19, 0)           # session ends 18:30 Bucharest; wait until 19:00
RETRY_STATUSES = {401, 429, 500, 502, 503, 504}
DEFAULT_SYMBOLS = ["TLV", "SNP", "H2O", "SNG", "BRD", "SNN", "TGN", "EL", "FP", "DIGI"]
COLUMNS = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


class BVBError(Exception):
    """Any failure talking to or parsing the BVB datafeed."""


class BVBAuthError(BVBError):
    """Token missing, expired, or the browser-like headers were rejected."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _utc_ts(d: date) -> int:
    """Unix seconds of 00:00 UTC on date d (how BVB stamps daily bars)."""
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _bucharest() -> ZoneInfo:
    try:
        return ZoneInfo("Europe/Bucharest")
    except ZoneInfoNotFoundError as exc:
        raise BVBError("Time zone database missing. On Windows run: pip install tzdata") from exc


def last_complete_session(now: datetime | None = None) -> date:
    """Latest date whose daily bar is final: today after 19:00 Bucharest on a
    weekday, otherwise the previous weekday. Holidays are not modelled; asking
    for a holiday just returns no bar for it."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(_bucharest())
    d = local.date()
    if not (local.weekday() < 5 and local.time() >= CLOSE_CUTOFF):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def load_dotenv(path: str | os.PathLike = ".env") -> bool:
    """Minimal .env reader. Existing non-empty environment variables win."""
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = value
    return True


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "volume": pd.Series(dtype="int64"),
    })


def parse_history(data: dict) -> pd.DataFrame:
    """Turn a /history JSON payload into a DataFrame[date, open, high, low,
    close, volume], oldest first, one row per date."""
    status = data.get("s")
    if status == "no_data":
        return _empty_frame()
    if status != "ok":
        raise BVBError(f"unexpected history status {status!r}: {data.get('errmsg')}")
    keys = ["t", "o", "h", "l", "c", "v"]
    missing = [k for k in keys if k not in data]
    if missing:
        raise BVBError(f"history payload missing arrays: {missing}")
    lengths = {k: len(data[k]) for k in keys}
    if len(set(lengths.values())) != 1:
        raise BVBError(f"history arrays differ in length: {lengths}")
    if lengths["t"] == 0:
        return _empty_frame()
    t = pd.Series(data["t"], dtype="int64")
    if (t % 86400 != 0).any():
        raise BVBError("daily bar not stamped at 00:00 UTC; refusing to guess the trading date")
    df = pd.DataFrame({
        "date": pd.to_datetime(t, unit="s").dt.normalize(),
        "open": pd.Series(data["o"], dtype="float64"),
        "high": pd.Series(data["h"], dtype="float64"),
        "low": pd.Series(data["l"], dtype="float64"),
        "close": pd.Series(data["c"], dtype="float64"),
        "volume": pd.Series(data["v"]).astype("int64"),
    })
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def count_mismatches(old: pd.DataFrame, new: pd.DataFrame) -> int:
    """Number of dates present in both frames where any OHLCV value differs."""
    if old.empty or new.empty:
        return 0
    m = old.merge(new, on="date", suffixes=("_o", "_n"))
    if m.empty:
        return 0
    diff = m["volume_o"] != m["volume_n"]
    for col in PRICE_COLS:
        diff = diff | ((m[f"{col}_o"] - m[f"{col}_n"]).abs() > 1e-6)
    return int(diff.sum())


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------

class BVBClient:
    def __init__(self, token: str | None = None, session=None, min_interval: float = 1.0,
                 max_retries: int = 4, timeout: float = 30, backoff: float = 2.0,
                 countback: int = COUNTBACK, sleep=time.sleep):
        token = (token or os.environ.get("BVB_WAPI_TOKEN", "")).strip()
        if not token:
            raise BVBAuthError(
                "BVB_WAPI_TOKEN is not set. Put it in .env (see .env.example). "
                "Re-capture it from any /api/history request in the Network tab of bvb.ro.")
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({
            "X-SK-API": token,
            "Origin": SITE_ORIGIN,
            "Referer": SITE_ORIGIN + "/",
            "User-Agent": os.environ.get("BVB_USER_AGENT") or DEFAULT_UA,
            "Accept": "application/json",
        })
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff = backoff
        self.countback = countback
        self._sleep = sleep
        self._last_call = None

    # -- transport ---------------------------------------------------------
    def _throttle(self):
        if self._last_call is not None and self.min_interval > 0:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: dict) -> dict:
        url = f"{BASE_URL}/{path}"
        last_status, last_err = None, None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_status, last_err = None, str(exc)
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise BVBError(f"non-JSON response from {path}") from exc
                last_status = resp.status_code
                last_err = resp.text[:200]
                if resp.status_code not in RETRY_STATUSES:
                    raise BVBError(f"HTTP {resp.status_code} from {path}: {last_err}")
            if attempt < self.max_retries:
                self._sleep(self.backoff * (2 ** attempt))
        attempts = self.max_retries + 1
        if last_status == 401:
            raise BVBAuthError(
                f"401 Unauthorized after {attempts} attempts. The token expired, or the "
                "Origin/Referer/User-Agent headers were rejected. Re-capture X-SK-API "
                "from a /api/history request on bvb.ro and update .env.")
        raise BVBError(f"{path} failed after {attempts} attempts (status {last_status}): {last_err}")

    # -- endpoints ---------------------------------------------------------
    def symbol_info(self, symbol: str) -> dict:
        return self._get("symbols", {"symbol": symbol})

    def _history_chunk(self, symbol: str, start: date, end: date, adjusted: bool,
                       currency: str) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "from": _utc_ts(start) - 1,       # 'from' is exclusive on the wire
            "to": _utc_ts(end),               # 'to' is inclusive
            "rs": "1D",
            "ajust": int(adjusted),
            "countback": self.countback,
            "currencyCode": currency,
        }
        df = parse_history(self._get("history", params))
        # countback caps the number of bars. If we hit the cap and the oldest bar
        # is later than the window start, the start of the window was cut off.
        if len(df) >= self.countback and df["date"].iloc[0] > pd.Timestamp(start):
            raise BVBError(f"{symbol}: response truncated by countback={self.countback} "
                           f"({start}..{end}); use a shorter window")
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df.loc[mask].reset_index(drop=True)

    def history_daily(self, symbol: str, start: date, end: date, adjusted: bool = False,
                      currency: str = "RON") -> pd.DataFrame:
        """Daily bars for start..end inclusive (trading days only)."""
        if start > end:
            return _empty_frame()
        parts, s = [], start
        while s <= end:
            e = min(s + timedelta(days=WINDOW_DAYS), end)
            parts.append(self._history_chunk(symbol, s, e, adjusted, currency))
            s = e + timedelta(days=1)
        parts = [p for p in parts if not p.empty]
        if not parts:
            return _empty_frame()
        out = pd.concat(parts, ignore_index=True)
        return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

class DailyCache:
    """One CSV per symbol: <root>/<SYMBOL>_<CCY>_raw.csv (unadjusted bars)."""

    def __init__(self, root: str | os.PathLike = "data/raw/bvb"):
        self.root = Path(root)

    def path(self, symbol: str, currency: str = "RON") -> Path:
        return self.root / f"{symbol}_{currency}_raw.csv"

    def load(self, symbol: str, currency: str = "RON") -> pd.DataFrame:
        p = self.path(symbol, currency)
        if not p.exists():
            return _empty_frame()
        df = pd.read_csv(p, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def upsert(self, symbol: str, new: pd.DataFrame, currency: str = "RON"):
        """Merge new bars over the cache (new wins on equal dates).
        Returns (merged_frame, number_of_overlapping_dates_whose_values_changed)."""
        old = self.load(symbol, currency)
        mismatches = count_mismatches(old, new)
        if old.empty:
            merged = new.copy()
        elif new.empty:
            merged = old
        else:
            merged = pd.concat([old[~old["date"].isin(new["date"])], new], ignore_index=True)
        merged = merged.sort_values("date").reset_index(drop=True)
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(symbol, currency)
        tmp = p.with_suffix(".tmp")
        merged.to_csv(tmp, index=False, date_format="%Y-%m-%d")
        os.replace(tmp, p)
        return merged, mismatches


def update_symbol(client, cache: DailyCache, symbol: str, start: date = DEFAULT_START,
                  now: datetime | None = None, currency: str = "RON") -> dict:
    """Bring one symbol's raw cache up to the last complete session."""
    end = last_complete_session(now)
    existing = cache.load(symbol, currency)
    if existing.empty or existing["date"].min().date() > start + timedelta(days=7):
        fetch_start = start                       # nothing cached, or history requested further back
    else:
        fetch_start = max(start, existing["date"].max().date() - timedelta(days=OVERLAP_DAYS))
    last = existing["date"].max().date() if not existing.empty else None
    if fetch_start > end:
        return {"symbol": symbol, "new_rows": 0, "mismatches": 0, "last": last, "fetched_from": fetch_start}
    fetched = client.history_daily(symbol, fetch_start, end, adjusted=False, currency=currency)
    if fetched.empty:
        return {"symbol": symbol, "new_rows": 0, "mismatches": 0, "last": last, "fetched_from": fetch_start}
    new_rows = len(set(fetched["date"]) - set(existing["date"]))
    merged, mismatches = cache.upsert(symbol, fetched, currency)
    return {"symbol": symbol, "new_rows": new_rows, "mismatches": mismatches,
            "last": merged["date"].max().date(), "fetched_from": fetch_start}


def trading_days(cache: DailyCache, symbols, calendar_symbol: str = CALENDAR_SYMBOL,
                 currency: str = "RON") -> pd.DatetimeIndex:
    """Trading calendar: the index's dates if cached, else every date on which
    at least one of the given shares traded."""
    cal = cache.load(calendar_symbol, currency)
    if not cal.empty:
        return pd.DatetimeIndex(cal["date"]).sort_values()
    days = set()
    for s in symbols:
        days.update(cache.load(s, currency)["date"])
    return pd.DatetimeIndex(sorted(days))


def adj_ratio_changes(raw: pd.DataFrame, adj: pd.DataFrame, tol: float = 1e-4) -> dict:
    """Compare raw and back-adjusted bars for the same range.
    Returns the dates where adj_close/raw_close steps (corporate actions) and
    whether adjusted volumes differ from raw volumes."""
    m = raw.merge(adj, on="date", suffixes=("_raw", "_adj"))
    if m.empty:
        return {"rows": 0, "steps": [], "volume_differs": False}
    ratio = m["close_adj"] / m["close_raw"]
    steps = []
    prev = ratio.iloc[0]
    for d, r in zip(m["date"].iloc[1:], ratio.iloc[1:]):
        if abs(r / prev - 1) > tol:
            steps.append((d.date(), float(prev), float(r)))
        prev = r
    return {"rows": len(m), "steps": steps, "volume_differs": bool((m["volume_raw"] != m["volume_adj"]).any())}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _make_client(args) -> BVBClient:
    return BVBClient(min_interval=args.interval)


def cmd_validate(args) -> int:
    client = _make_client(args)
    ok = True
    info = client.symbol_info("TLV")
    print(f"[ok] symbols: {info.get('name')} - {info.get('description')}, "
          f"session {info.get('session')}, tz {info.get('timezone')}, {info.get('currency_code')}")
    s, e = date(2025, 7, 7), date(2025, 7, 23)
    raw = client.history_daily("TLV", s, e, adjusted=False)
    adj = client.history_daily("TLV", s, e, adjusted=True)
    print(f"[{'ok' if len(raw) == 12 else 'CHECK'}] TLV {s}..{e}: raw {len(raw)} bars, adjusted {len(adj)} bars "
          "(expected 12, matches the capture)")
    if len(raw) != 12:
        ok = False
    if not raw.empty:
        print(f"      first raw bar : {raw.iloc[0].to_dict()}")
        print(f"      first adj close: {adj['close'].iloc[0] if not adj.empty else 'n/a'}")
    end = last_complete_session()
    recent = client.history_daily("TLV", end - timedelta(days=14), end)
    last = recent["date"].max().date() if not recent.empty else None
    print(f"[ok] TLV last bar {last}; last complete session is {end}")
    try:
        bet = client.history_daily(CALENDAR_SYMBOL, end - timedelta(days=14), end)
        print(f"[{'ok' if not bet.empty else 'CHECK'}] {CALENDAR_SYMBOL} calendar: {len(bet)} bars in the last 14 days")
        if bet.empty:
            print("      -> index symbol returned nothing; sync will fall back to a union of share dates")
    except BVBAuthError:
        raise
    except BVBError as exc:
        print(f"[CHECK] {CALENDAR_SYMBOL} not available ({exc}); sync will fall back to a union of share dates")
    print("validate:", "PASS" if ok else "see CHECK lines")
    return 0 if ok else 1


def cmd_sync(args) -> int:
    client = _make_client(args)
    cache = DailyCache(args.root)
    symbols = [s.upper() for s in (args.symbols or DEFAULT_SYMBOLS)]
    if CALENDAR_SYMBOL not in symbols:
        symbols.append(CALENDAR_SYMBOL)
    start = date.fromisoformat(args.start)
    failures = 0
    for sym in symbols:
        try:
            r = update_symbol(client, cache, sym, start)
        except BVBAuthError:
            raise
        except BVBError as exc:
            failures += 1
            print(f"{sym:6s} FAILED: {exc}")
            continue
        flag = "" if r["mismatches"] == 0 else f"  WARNING: {r['mismatches']} overlapping bars changed since last sync"
        print(f"{sym:6s} +{r['new_rows']:<5d} rows  last {r['last']}  (fetched from {r['fetched_from']}){flag}")
    days = trading_days(cache, [s for s in symbols if s != CALENDAR_SYMBOL])
    print(f"trading calendar: {len(days)} days" + (f", {days.min().date()}..{days.max().date()}" if len(days) else ""))
    return 1 if failures else 0


def cmd_adjcheck(args) -> int:
    client = _make_client(args)
    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end) if args.end else last_complete_session()
    sym = args.symbol.upper()
    raw = client.history_daily(sym, s, e, adjusted=False)
    adj = client.history_daily(sym, s, e, adjusted=True)
    res = adj_ratio_changes(raw, adj)
    print(f"{sym} {s}..{e}: {res['rows']} bars compared (raw {len(raw)}, adjusted {len(adj)})")
    if not res["steps"]:
        print("adjusted close / raw close is constant: no corporate action visible in this range")
    for d, before, after in res["steps"]:
        print(f"  step at {d}: ratio {before:.6f} -> {after:.6f}")
    print("volumes " + ("DIFFER between raw and adjusted" if res["volume_differs"] else "are identical in raw and adjusted"))
    print("MSCI value = shares x close on the day, so the cache keeps raw close and raw volume.")
    return 0


def main(argv=None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="BVB wapi daily-bar loader")
    p.add_argument("--interval", type=float, default=1.0, help="min seconds between requests")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check auth, endpoints and the capture window")
    sp = sub.add_parser("sync", help="update the raw cache")
    sp.add_argument("--symbols", nargs="*", help="default: %s" % " ".join(DEFAULT_SYMBOLS))
    sp.add_argument("--start", default=DEFAULT_START.isoformat())
    sp.add_argument("--root", default="data/raw/bvb")
    sa = sub.add_parser("adjcheck", help="compare raw vs back-adjusted bars for one symbol")
    sa.add_argument("symbol")
    sa.add_argument("--start", default=DEFAULT_START.isoformat())
    sa.add_argument("--end")
    args = p.parse_args(argv)
    handler = {"validate": cmd_validate, "sync": cmd_sync, "adjcheck": cmd_adjcheck}[args.cmd]
    try:
        return handler(args)
    except BVBError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
