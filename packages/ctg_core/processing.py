from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import ml_settings

def basic_qc(df: pd.DataFrame) -> pd.DataFrame:
    # df: columns ["ts", "bpm", "ua"]
    out = df.copy()
    lo_bpm, hi_bpm = ml_settings.bpm_range_lo, ml_settings.bpm_range_hi
    lo_ua, hi_ua   = ml_settings.ua_range_lo, ml_settings.ua_range_hi
    mask_bpm = (out["bpm"] >= lo_bpm) & (out["bpm"] <= hi_bpm)
    mask_ua  = (out["ua"]  >= lo_ua) & (out["ua"]  <= hi_ua)
    out.loc[~mask_bpm, "bpm"] = np.nan
    out.loc[~mask_ua,  "ua"]  = np.nan
    # линейная интерполяция, затем ffill/bfill как fallback
    out["bpm"] = out["bpm"].interpolate().ffill().bfill()
    out["ua"]  = out["ua"].interpolate().ffill().bfill()
    return out

def resample_uniform(df_bpm: pd.DataFrame, df_ua: pd.DataFrame) -> pd.DataFrame:
    rb = df_bpm.set_index("ts").sort_index().resample(ml_settings.resample_every).mean().rename(columns={"value":"bpm"})
    ru = df_ua.set_index("ts").sort_index().resample(ml_settings.resample_every).mean().rename(columns={"value":"ua"})
    uni = rb.join(ru, how="outer")
    # DatetimeIndex.asi8 -> int64 наносекунды
    t_sec = uni.index.asi8.astype(np.float64) / 1e9
    uni = uni.reset_index()  # колонка 'ts' сохраняется
    uni["t_sec"] = t_sec
    uni["bpm"] = uni["bpm"].ffill().bfill()
    uni["ua"]  = uni["ua"].ffill().bfill()
    return uni[["ts","t_sec","bpm","ua"]]

def rolling_baseline(arr: np.ndarray, win_sec: int) -> np.ndarray:
    # медианный baseline с fallback на mean (устойчивый)
    if arr.size == 0:
        return np.array([], dtype=float)
    freq = ml_settings.target_freq_hz
    w = max(5, int(freq * win_sec))
    s = pd.Series(arr)
    med = s.rolling(w, center=True, min_periods=w//2).median()
    mean = s.rolling(w, center=True, min_periods=w//2).mean()
    baseline = med.to_numpy()
    bad = np.isnan(baseline)
    if bad.any():
        baseline[bad] = mean.to_numpy()[bad]
    return baseline

def histogram_features(values: np.ndarray, bin_width: float = 1.0) -> Dict[str, float]:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {
            "histogram_width": np.nan, "histogram_min": np.nan, "histogram_max": np.nan,
            "histogram_number_of_peaks": np.nan, "histogram_number_of_zeroes": np.nan,
            "histogram_mode": np.nan, "histogram_mean": np.nan, "histogram_median": np.nan,
            "histogram_variance": np.nan, "histogram_tendency": np.nan
        }
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax <= vmin:
        return {
            "histogram_width": 0.0, "histogram_min": vmin, "histogram_max": vmax,
            "histogram_number_of_peaks": 0, "histogram_number_of_zeroes": 0,
            "histogram_mode": vmin, "histogram_mean": float(np.mean(v)),
            "histogram_median": float(np.median(v)), "histogram_variance": float(np.var(v)),
            "histogram_tendency": 0
        }
    bins = np.arange(np.floor(vmin), np.ceil(vmax) + bin_width, bin_width)
    hist, edges = np.histogram(v, bins=bins)
    # примитивный подсчёт пиков/нулей
    peaks = int(sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])))
    nz = np.where(hist > 0)[0]
    if nz.size == 0:
        zeros = int(hist.size)
    else:
        lo, hi = nz[0], nz[-1]
        zeros = int(np.sum(hist[lo:hi+1] == 0))
    imax = int(np.argmax(hist)) if hist.size else 0
    mode_val = float(0.5 * (edges[imax] + edges[imax+1])) if hist.size else float("nan")
    x = np.arange(len(v), dtype=float)
    try:
        slope = float(np.polyfit(x, v, 1)[0])
    except Exception:
        slope = 0.0
    thr = 0.01  # bpm/сек
    tendency = 1 if slope > thr else (-1 if slope < -thr else 0)
    return {
        "histogram_width": float(vmax - vmin), "histogram_min": vmin, "histogram_max": vmax,
        "histogram_number_of_peaks": peaks, "histogram_number_of_zeroes": zeros,
        "histogram_mode": mode_val, "histogram_mean": float(np.mean(v)),
        "histogram_median": float(np.median(v)), "histogram_variance": float(np.var(v)),
        "histogram_tendency": tendency
    }