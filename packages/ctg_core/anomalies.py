from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from .config import ml_settings
from .processing import rolling_baseline

def detect_anomalies(df: pd.DataFrame) -> Dict:
    """
    Детектор аномалий на окне/сессии. df: ["t_sec","bpm","ua"]
    Возвращает агрегаты + список эпизодов децелераций.
    """
    out = dict(
        tachy_frac=np.nan,
        brady_frac=np.nan,
        low_var_frac=np.nan,
        decel_count=0,
        decel_events=[],
        crosscorr_max=np.nan,
        crosscorr_lag_s=np.nan,
    )
    if df is None or len(df) == 0:
        return out

    bpm = df["bpm"].to_numpy(float)
    ua  = df["ua"].to_numpy(float)
    F   = ml_settings.target_freq_hz

    w = max(5, int(F * ml_settings.baseline_roll_sec))
    if len(bpm) < w:
        out["tachy_frac"] = float(np.mean(bpm > ml_settings.tachy_bpm))
        out["brady_frac"] = float(np.mean(bpm < ml_settings.brady_bpm))
        return out

    baseline = rolling_baseline(bpm, ml_settings.baseline_roll_sec)
    drop = baseline - bpm
    is_decel = drop >= ml_settings.decel_light_drop_bpm

    # Поиск эпизодов
    m = is_decel.astype(np.int8)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    events = []
    for s, e in zip(starts, ends):
        dur_s = (e - s + 1) / F
        if dur_s < ml_settings.decel_min_sec:
            continue
        segment = slice(s, e+1)
        min_bpm = float(np.min(bpm[segment]))
        max_drop = float(np.max((baseline[segment] - bpm[segment])))
        events.append(dict(start_idx=int(s), end_idx=int(e), dur_s=float(dur_s),
                           min_bpm=min_bpm, max_drop=max_drop))

    out["tachy_frac"] = float(np.mean(bpm > ml_settings.tachy_bpm))
    out["brady_frac"] = float(np.mean(bpm < ml_settings.brady_bpm))
    diffs = np.diff(bpm)
    stv = np.mean(np.abs(diffs)) if diffs.size else np.nan
    out["low_var_frac"] = float(stv < ml_settings.low_var_stv) if np.isfinite(stv) else np.nan
    out["decel_count"] = len(events)
    out["decel_events"] = events

    # Кросс-корреляция UA vs -BPM (ограниченный лаг ±120 c)
    max_lag = int(120 * F)
    if len(ua) > 2 * max_lag + 1:
        x = (ua - np.nanmean(ua))
        y = -(bpm - np.nanmean(bpm))
        x = x / (np.nanstd(x) + 1e-6)
        y = y / (np.nanstd(y) + 1e-6)
        corr = np.correlate(x, y, mode="full") / len(x)
        lags = np.arange(-len(x)+1, len(x))
        mask = (lags >= -max_lag) & (lags <= max_lag)
        corr = corr[mask]; lags = lags[mask]
        best_i = int(np.nanargmax(corr))
        out["crosscorr_max"] = float(corr[best_i])
        out["crosscorr_lag_s"] = float(lags[best_i] / F)
    return out