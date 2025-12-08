from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from model import predict  # uses existing predict(home_id, away_id)

app = FastAPI(title="Premier League Predictor API")


class MatchRequest(BaseModel):
    home_team_id: int
    away_team_id: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/predict")
async def predict_get(home: int, away: int):
    try:
        res = predict(home, away)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if res is None:
        raise HTTPException(status_code=404, detail="No prediction available")
    return res


@app.post("/predict")
async def predict_post(match: MatchRequest):
    try:
        res = predict(match.home_team_id, match.away_team_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if res is None:
        raise HTTPException(status_code=404, detail="No prediction available")
    return res
