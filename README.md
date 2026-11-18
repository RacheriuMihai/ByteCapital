# ByteCapital — Road to Emerging Market

MSCI-style Emerging Market eligibility tool for BVB-listed companies: size tests
(full and free-float market cap, FIF), liquidity tests (12- and 3-month ATVR,
quarterly frequency of trading), scenario levers, a liquidity gap list and the
8-review country streak. Not affiliated with MSCI.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                   # then put the BVB token in BVB_WAPI_TOKEN
python ingest/bvb_data.py validate     # checks the token and the endpoints
python ingest/bvb_data.py sync         # daily bars -> data/raw/bvb/<SYMBOL>_RON_raw.csv
python ingest/bnr.py                   # BNR RON/USD rates -> data/fx_usd.csv
pytest                                 # engine + BVB client tests (no network)
streamlit run app/streamlit_app.py     # dashboard
```

Optional API: `uvicorn api.main:app --reload` (docs at /docs).

## Data

| File | Content | Updated |
|---|---|---|
| `data/raw/bvb/<SYMBOL>_RON_raw.csv` | raw (unadjusted) daily close and volume | `ingest/bvb_data.py sync`; not committed |
| `data/trades.csv` | daily trades per ticker, manual BVB exports | fallback, used only for symbols with no cache file |
| `data/fx_usd.csv` | BNR RON/USD rates | automatic (GitHub Actions, daily) |
| `data/issuers.csv` | shares outstanding, Pillar II stake | manual |
| `data/shares_history.csv` | share count changes since 2024 (bonus shares, splits, cancellations), from the ex-date | manual, checked with `bvb_data.py adjcheck` |
| `data/shareholders.csv` | named holders, `type`, source and date | manual, from issuer reports and BVB "Structura actionariat" |
| `config/thresholds.csv` | MSCI-style thresholds (GIMI, Nov 2026) | manual, with source |

**BVB data access.** `ingest/bvb_data.py` reads BVB's datafeed with a token.
BVB has agreed verbally; the written consent is pending. Until it arrives, keep
syncing to once a day, keep `data/raw/bvb/` out of git, and check that the
written terms cover this access. The token lives only in `.env` (git-ignored).

**Free float.** `type` in `shareholders.csv` decides the classification:

- strategic: `government` (or `state`), `company`, `treasury`, `bank`, `employee`,
  `officer`, `private_equity`, `individual`
- free float: `fund`, `insurer`, `broker`, `pension_pillar2`, `retail_other`

Free float = 100% minus all strategic stakes, so the public need not be listed.
The `strategic` column is for reading only; the engine raises an error if it
contradicts `type`. FIF is rounded the MSCI way (2.5% steps above 25%, 0.5% from
5% to 25%, 0.1% below 5%).

## Method

- Daily traded value = close × volume (RON). ATVR is a ratio, so no FX is needed.
- Monthly ratio = median daily value (days traded) × days traded ÷ month-end free-float cap.
- ATVR 12m / 3m = mean of the last 12 / 3 monthly ratios × 12. Both must be ≥ 15%.
- Frequency of trading ≥ 80% in each of the last 4 quarters.
- Size: full cap ≥ USD 4,343 mm, free-float cap ≥ USD 2,172 mm (× 1.8 if FIF < 0.15),
  at the BNR rate of the review month.
- Reviews: cut-offs in March, June, September, December. The market qualifies at a
  review when at least 3 companies pass everything; the target is 8 in a row.

Share count per month comes from `data/shares_history.csv` (the count valid at
month end); companies not listed there keep the value from `issuers.csv`, and the
last row of a company's history must equal `issuers.csv`. Simplification: the
free-float share (FIF) is held at its current value for the whole history.