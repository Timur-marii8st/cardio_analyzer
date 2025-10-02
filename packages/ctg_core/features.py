from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import ml_settings
from .processing import rolling_baseline, histogram_features

CTG_FEATURES = [
    "baseline value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
]

def _find_events(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    out = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_len:
            out.append((int(s), int(e)))
    return out

def compute_ctg_features_window(df: pd.DataFrame) -> Dict[str, float]:
    """
    Ожидается df с колонками ["t_sec","bpm","ua"] для последнего окна (например, 10 мин).
    Возвращает словарь признаков уровня окна = те же CTG, но нормированные per second.
    """
    out = {k: np.nan for k in CTG_FEATURES}
    if df is None or len(df) == 0:
        return out

    df = df.sort_values("t_sec")
    bpm = df["bpm"].to_numpy(float)
    ua  = df["ua"].to_numpy(float) if "ua" in df.columns else np.full_like(bpm, np.nan)
    n   = len(bpm)
    F   = int(ml_settings.target_freq_hz)

    duration_sec = max(float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0]), 1e-6)

    baseline = rolling_baseline(bpm, ml_settings.baseline_roll_sec)
    out["baseline value"] = float(np.nanmedian(baseline))

    event_bl = rolling_baseline(bpm, ml_settings.baseline_event_roll_sec)

    if n >= F:
        n1 = (n // F) * F
        bpm_1s = bpm[:n1].reshape(-1, F).mean(axis=1)
        bl_1s  = event_bl[:n1].reshape(-1, F).mean(axis=1)
    else:
        bpm_1s, bl_1s = bpm.copy(), event_bl.copy()

    # Детекция событий
    def detect_events(acc_thr, acc_min_s, dec_light_thr, dec_severe_thr, dec_min_s, dec_prol_s,
                      ua_thr, ua_min_s):
        acc_mask = (bpm - event_bl) >= acc_thr
        dec_mask = (event_bl - bpm) >= dec_light_thr
        acc_events = _find_events(acc_mask, int(acc_min_s * ml_settings.target_freq_hz))
        dec_events = _find_events(dec_mask, int(dec_min_s  * ml_settings.target_freq_hz))
        light_dec = severe_dec = prolonged_dec = 0
        for s, e in dec_events:
            dur_s = (e - s + 1) / ml_settings.target_freq_hz
            drop  = (event_bl[s:e+1] - bpm[s:e+1])
            max_drop = float(np.nanmax(drop)) if drop.size else 0.0
            if (dec_light_thr <= max_drop < dec_severe_thr) and (dur_s >= dec_min_s):
                light_dec += 1
            if (max_drop >= dec_severe_thr) and (dur_s >= dec_min_s):
                severe_dec += 1
            if dur_s >= dec_prol_s:
                prolonged_dec += 1

        # Схватки UA
        if np.isfinite(ua).any():
            ua_bl = rolling_baseline(ua, ml_settings.baseline_event_roll_sec)
            ua_mask = (ua - ua_bl) >= ua_thr
            ua_events = _find_events(ua_mask, int(ua_min_s * ml_settings.target_freq_hz))
        else:
            ua_events = []

        # fetal_movement = акцелерации вне ±10с от UA
        fm_count = 0
        if acc_events:
            ua_intervals = [(s / ml_settings.target_freq_hz, e / ml_settings.target_freq_hz) for s, e in ua_events]
            for s, e in acc_events:
                mid_t = 0.5 * (s + e) / ml_settings.target_freq_hz
                near_ua = any((mid_t >= us - 10.0) and (mid_t <= ue + 10.0) for us, ue in ua_intervals)
                if not near_ua:
                    fm_count += 1

        return dict(acc=len(acc_events), fm=fm_count, ua=len(ua_events),
                    light=light_dec, severe=severe_dec, prolong=prolonged_dec)

    ev = detect_events(
        ml_settings.acc_rise_bpm, ml_settings.acc_min_sec,
        ml_settings.decel_light_drop_bpm, ml_settings.decel_severe_drop_bpm,
        ml_settings.decel_min_sec, ml_settings.decel_prolonged_sec,
        ml_settings.ua_contr_min_rise, ml_settings.ua_contr_min_sec
    )
    if (ev["acc"] + ev["ua"] + ev["light"] + ev["severe"] + ev["prolong"]) == 0:
        # fallback: ослабленные пороги
        ev = detect_events(
            max(6.0, ml_settings.acc_rise_bpm * 0.7),
            max(8.0, ml_settings.acc_min_sec  * 0.7),
            max(10.0, ml_settings.decel_light_drop_bpm * 0.8),
            max(18.0, ml_settings.decel_severe_drop_bpm * 0.8),
            max(8.0, ml_settings.decel_min_sec * 0.7),
            ml_settings.decel_prolonged_sec,
            max(5.0, ml_settings.ua_contr_min_rise * 0.5),
            max(8.0, ml_settings.ua_contr_min_sec * 0.7),
        )

    out["accelerations"]            = float(ev["acc"]     / duration_sec)
    out["fetal_movement"]           = float(ev["fm"]      / duration_sec)
    out["uterine_contractions"]     = float(ev["ua"]      / duration_sec)
    out["light_decelerations"]      = float(ev["light"]   / duration_sec)
    out["severe_decelerations"]     = float(ev["severe"]  / duration_sec)
    out["prolongued_decelerations"] = float(ev["prolong"] / duration_sec)

    # STV/LTV
    if len(bpm_1s) >= 2:
        stv_steps = np.abs(np.diff(bpm_1s))
        out["mean_value_of_short_term_variability"] = float(np.nanmean(stv_steps))
        out["abnormal_short_term_variability"] = float(np.mean(stv_steps < ml_settings.low_var_stv) * 100.0)
    if len(bpm_1s) >= 60:
        m = (len(bpm_1s) // 60) * 60
        mins = bpm_1s[:m].reshape(-1, 60)
        ltv = np.nanstd(mins, axis=1)
        out["mean_value_of_long_term_variability"] = float(np.nanmean(ltv))
        out["percentage_of_time_with_abnormal_long_term_variability"] = float(np.mean(ltv < 5.0) * 100.0)

    # Гистограммы по baseline (1-сек)
    hist_feats = histogram_features(bl_1s if len(bpm_1s) else event_bl, bin_width=1.0)
    out.update(hist_feats)
    return out