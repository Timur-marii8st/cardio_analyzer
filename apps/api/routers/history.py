from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from ..deps import get_storage
from packages.ctg_core.processing import resample_uniform, basic_qc, rolling_baseline

router = APIRouter(prefix="/v1", tags=["history"])

@router.get("/sessions/{session_id}/signals")
def get_signals(
    session_id: str,
    since: datetime = Query(...),
    until: datetime = Query(...),
    step: str = Query("1s"),
    storage = Depends(get_storage)
):
    points = storage.load_raw_points(session_id, since, until)
    if not points:
        raise HTTPException(status_code=404, detail="No data")
    
    df = pd.DataFrame(points)
    df_bpm = df[["ts","bpm"]].dropna().rename(columns={"bpm":"value"})
    df_ua  = df[["ts","ua"]].dropna().rename(columns={"ua":"value"})
    
    if df_bpm.empty:
        raise HTTPException(status_code=400, detail="No bpm channel")
    
    if df_ua.empty:
        df_ua = pd.DataFrame({"ts": df_bpm["ts"].values, "value": np.nan})
    
    uni = resample_uniform(df_bpm, df_ua)
    uni = basic_qc(uni)
    
    if step != "250ms":
        r = uni.set_index("ts").resample(step).mean(numeric_only=True).reset_index()
        r["t_sec"] = (r["ts"].astype("datetime64[ns]").astype("int64") / 1e9).astype(float)
        uni = r
    
    bl = rolling_baseline(uni["bpm"].to_numpy(float), 60).tolist()
    
    return {
        "ts": uni["ts"].astype(str).tolist(),
        "bpm": uni["bpm"].astype(float).tolist(),
        "ua":  uni["ua"].astype(float).tolist(),
        "baseline_60s": bl
    }

@router.get("/sessions/{session_id}/events")
def get_events(
    session_id: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    storage = Depends(get_storage)
):
    return storage.load_events(session_id, since, until)

@router.get("/sessions/{session_id}/risk_history")
def get_risk_history(
    session_id: str,
    limit: int = 200,
    storage = Depends(get_storage)
):
    return storage.load_risk(session_id, limit=limit)