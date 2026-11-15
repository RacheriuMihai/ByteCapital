# ingest/bnr.py — preia cursul RON/USD de la BNR (XML anual, fără namespace fixat)
from pathlib import Path

import requests
import xml.etree.ElementTree as ET
import pandas as pd

URLS = [
    "https://curs.bnr.ro/files/xml/years/nbrfxrates{year}.xml",  # host nou (2026)
    "https://www.bnr.ro/files/xml/years/nbrfxrates{year}.xml",   # host vechi, fallback
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ByteCapital/1.0"}


def local(tag):
    return tag.split("}")[-1]  # ignoră namespace-ul (BNR l-a schimbat în 2026)


def _download(year: int) -> bytes:
    last_err = None
    for url in URLS:
        u = url.format(year=year)
        try:
            r = requests.get(u, headers=HEADERS, timeout=30)
            r.raise_for_status()
            body = r.content.lstrip()
            low = body[:80].lower()
            if not body.startswith(b"<") or b"<html" in low or b"<!doctype" in low:
                raise ValueError(f"Răspuns non-XML de la {u}: {body[:150]!r}")
            return body
        except Exception as e:  # încearcă următorul URL
            last_err = e
    raise RuntimeError(f"Nu am putut descărca XML-ul BNR pentru {year}") from last_err


def fetch_usd(year: int) -> pd.DataFrame:
    xml = _download(year)
    rows = []
    for cube in ET.fromstring(xml).iter():
        if local(cube.tag) != "Cube":
            continue
        for rate in cube:
            if local(rate.tag) == "Rate" and rate.get("currency") == "USD":
                mult = float(rate.get("multiplier", 1))
                rows.append({"date": cube.get("date"), "ron_per_usd": float(rate.text) / mult})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    fx = pd.concat([fetch_usd(y) for y in (2024, 2025, 2026)], ignore_index=True)
    fx = fx.drop_duplicates("date").sort_values("date")
    fx.to_csv("data/fx_usd.csv", index=False)
    print(f"OK: {len(fx)} zile, ultima {fx.date.max().date()} la {fx.ron_per_usd.iloc[-1]:.4f} RON/USD")