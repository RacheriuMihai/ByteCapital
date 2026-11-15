# ByteCapital — Road to Emerging Market

MSCI-style Emerging Market eligibility tool for BVB-listed companies: free float,
market cap and ATVR liquidity tests, scenario levers, and a liquidity gap list.
Built on public data (BVB manual exports, BNR FX). Not affiliated with MSCI.

## Run

```bash
pip install -r requirements.txt
python ingest/bnr.py                   # fetch BNR RON/USD rates -> data/fx_usd.csv
pytest                                 # sanity checks for the engine
streamlit run app/streamlit_app.py     # dashboard
```

Optional API: `uvicorn api.main:app --reload` (docs at /docs).

## Data

| File | Content | Updated |
|---|---|---|
| `data/fx_usd.csv` | BNR RON/USD rates | automatic (GitHub Actions, daily) |
| `data/trades.csv` | daily trades per ticker | manual BVB exports |
| `data/shareholders.csv` | large holders + strategic flag | manual, from issuer reports |
| `data/issuers.csv` | shares outstanding, Pillar II stake | manual |
| `config/thresholds.csv` | MSCI-style 2026 thresholds | manual, with source |

BVB terms do not allow automated collection from bvb.ro — trade data is
downloaded manually. Classification rule: state, controlling shareholders,
companies, banks and insiders are strategic (`strategic=1`); funds, pension
funds and the public are non-strategic.
