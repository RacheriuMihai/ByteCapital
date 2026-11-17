# api/main.py — FastAPI (opțional; Streamlit folosește engine/ direct)
import math

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from engine import pipeline

app = FastAPI(title="Road to EM API")


class Levers(BaseModel):
    volume: float = 0
    pension: float = 0
    state_sell: float = 0
    fx: float = 0


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON nu acceptă NaN: valorile lipsă devin null (și în listele FOT pe trimestre)."""
    def clean(v):
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, float) and math.isnan(v):
            return None
        return v.item() if hasattr(v, "item") else v
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/scenario")
def scenario(levers: Levers):
    res = pipeline.run(levers.model_dump())
    return {"review": res.review,
            "companies_passing": res.n_pass_all,
            "country_streak": res.country_streak,
            "tests": _records(res.tests),
            "gap_list": _records(res.gap_list),
            "streak": _records(res.streak)}
