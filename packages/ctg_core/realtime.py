from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .config import ml_settings
from .processing import resample_uniform, basic_qc, rolling_baseline
from .features import compute_ctg_features_window, CTG_FEATURES
from .anomalies import detect_anomalies

@dataclass
class SessionState:
    bpm_buf: list[tuple[datetime, float]]
    ua_buf: list[tuple[datetime, float]]
    last_risk: Optional[Tuple[float, str, datetime]] = None

class RealtimeProcessor:
    """
    Буферизует точки по сессии, каждые N сек считает фичи на окне T минут,
    запускает предикт и детектор аномалий.
    """
    def __init__(self, predict_fn):
        # predict_fn: callable(features_dict)->prob
        self.predict_fn = predict_fn
        self.sessions: Dict[str, SessionState] = defaultdict(lambda: SessionState(bpm_buf=[], ua_buf=[]))

    def ingest_samples(self, session_id: str, samples: list[dict]):
        st = self.sessions[session_id]
        for s in samples:
            ts = pd.to_datetime(s["ts"], utc=True)
            ch = s["channel"]; v = float(s["value"])
            if ch == "bpm":
                st.bpm_buf.append((ts, v))
            elif ch == "uterus":
                st.ua_buf.append((ts, v))
        # ограничиваем буферы последними (feature_window + запас) сек
        self._trim_buffers(st)

    def _trim_buffers(self, st: SessionState):
        horizon = timedelta(seconds=ml_settings.feature_window_sec + 60)
        cutoff  = datetime.now(timezone.utc) - horizon
        st.bpm_buf = [(ts, v) for ts, v in st.bpm_buf if ts >= cutoff]
        st.ua_buf  = [(ts, v) for ts, v in st.ua_buf  if ts >= cutoff]

    def step(self, session_id: str) -> Optional[dict]:
        st = self.sessions[session_id]
        if not st.bpm_buf:
            return None
        # Собираем последние T минут
        now = datetime.now(timezone.utc)
        start = now - timedelta(seconds=ml_settings.feature_window_sec)
        df_bpm = pd.DataFrame(st.bpm_buf, columns=["ts","value"])
        df_ua  = pd.DataFrame(st.ua_buf,  columns=["ts","value"]) if st.ua_buf else pd.DataFrame(columns=["ts","value"])
        df_bpm = df_bpm[df_bpm["ts"] >= start]
        df_ua  = df_ua[df_ua["ts"] >= start] if not df_ua.empty else df_ua

        if df_bpm.empty:
            return None

        uni = resample_uniform(df_bpm, df_ua if not df_ua.empty else pd.DataFrame({"ts": df_bpm["ts"], "value": np.nan}))
        uni = basic_qc(uni)

        if len(uni) < ml_settings.min_window_points:
            return None

        feats = compute_ctg_features_window(uni[["t_sec","bpm","ua"]])
        prob = float(self.predict_fn(feats))
        band = "normal" if prob < 0.2 else ("elevated" if prob < 0.5 else "high")
        st.last_risk = (prob, band, now)

        anoms = detect_anomalies(uni[["t_sec","bpm","ua"]])

        # baseline для графика
        bl = rolling_baseline(uni["bpm"].to_numpy(float), ml_settings.baseline_roll_sec).tolist()

        # события с таймстампами
        ts_series = pd.to_datetime(uni["ts"])
        decel_events = []
        for ev in anoms.get("decel_events", []):
            s = int(ev["start_idx"]); e = int(ev["end_idx"])
            if 0 <= s < len(ts_series) and 0 <= e < len(ts_series):
                decel_events.append({
                    "start_ts": ts_series.iloc[s].isoformat(),
                    "end_ts": ts_series.iloc[e].isoformat(),
                    "dur_s": ev.get("dur_s"),
                    "min_bpm": ev.get("min_bpm"),
                    "max_drop": ev.get("max_drop"),
                })

        return {
            "ts": now.isoformat(),
            "features": feats,
            "risk": {"hypoxia_prob": prob, "band": band},
            "anomalies": {k:v for k,v in anoms.items() if k not in ("decel_events",)},
            "decel_events": decel_events,
            "series": {
                "ts": ts_series.astype(str).tolist(),
                "bpm": uni["bpm"].astype(float).tolist(),
                "ua":  uni["ua"].astype(float).tolist(),
                "baseline_60s": bl,
            }
        }