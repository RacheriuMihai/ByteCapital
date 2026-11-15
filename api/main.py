# api/main.py — FastAPI (opțional; Streamlit folosește engine/ direct)
from fastapi import FastAPI
from pydantic import BaseModel

from engine import pipeline

app = FastAPI(title="Road to EM API")


class Levers(BaseModel):
    volume: float = 0
    pension: float = 0
    state_sell: float = 0
    fx: float = 0


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/scenario")
def scenario(levers: Levers):
    res = pipeline.run(levers.model_dump())
    return {"tests": res.tests.to_dict("records"),
            "gap_list": res.gap_list.to_dict("records")}
