from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

# МЛ
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import optuna

# =========================
# Конфигурация пайплайна
# =========================

class CFG:
    ROOT = Path('/data/ИТЭЛМА_ЛЦТ/ЛЦТ "НПП "ИТЭЛМА"')     # корень датасета 
    GROUPS = ["hypoxia", "regular"]
    CHANNELS = ["bpm", "uterus"]

    ACC_RISE_BPM = 10           # амплитуда акцелерации
    ACC_MIN_SEC = 10            # длительность акцелерации

    DECEL_LIGHT_DROP_BPM = 15   # лёгкая децелерация
    DECEL_SEVERE_DROP_BPM = 25  # сильная децелерация
    DECEL_MIN_SEC = 10          # минимальная длительность децелерации
    DECEL_PROLONGED_SEC = 120   # продолжительная децелерация

    UA_CONTR_MIN_RISE = 10      # подъем UA над базой (ранее было 20)
    UA_CONTR_MIN_SEC = 20       # длительность схватки (ранее 30)

    BASELINE_ROLL_SEC_EVENT = 30  # окно baseline для детекции событий (короче стандартных 60с)

    # Выравнивание/ресемплинг
    JOIN_TOL_SEC = 0.5                     # допуск ближайшей точки при asof join
    RESAMPLE_EVERY = "250ms"               # сетка 4 Гц
    FREQ_HZ = 4.0                          # частота после ресемплинга

    # QC и пороги аномалий
    BPM_RANGE = (50, 220)
    UA_RANGE  = (0, 120)
    TACHY_BPM = 160
    BRADY_BPM = 110

    # Децелерации (критерий FIGO приближённый)
    DECEL_DROP_BPM = 15        # падение относительно локального baseline
    BASELINE_ROLL_SEC = 60     # окно для baseline, сек
    BASELINE_ROLL_SAMPLES = int(FREQ_HZ * BASELINE_ROLL_SEC)

    # Низкая вариабельность (пример: STV < 1 bpm)
    LOW_VAR_STV = 1.0

    # Разбиение выборки
    SEED = 42
    N_SPLITS = 10

    EXTRA_DATA_PATH = "/fetal-health-classification/fetal_health.csv"

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


# =========================
# Утилиты I/O
# =========================
def detect_delim(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    if ";" in head: return ";"
    if "," in head: return ","
    if "\t" in head: return "\t"
    return r"\s+" 

def parse_file_meta(path: Path) -> dict:
    # .../<group>/<folder_id>/<channel>/<filename>.csv
    channel = path.parent.name
    folder_id = path.parent.parent.name
    group = path.parent.parent.parent.name
    stem = path.stem
    if "_" in stem:
        record_key, chan_idx = stem.rsplit("_", 1)
        try:
            chan_idx = int(chan_idx)
        except ValueError:
            chan_idx = None
    else:
        record_key, chan_idx = stem, None
    return {
        "group": group,
        "folder_id": folder_id,
        "channel": channel,
        "record_key": record_key,
        "chan_idx": chan_idx,
        "file_name": path.name,
        "file_path": str(path),
    }

def scan_channel_files(root: Path, group: str, channel: str) -> List[Path]:
    base = root / group
    if not base.exists():
        return []
    return list(base.glob(f"*/{channel}/*.csv"))

def scan_all_files(root: Path) -> List[Path]:
    files = []
    for g in CFG.GROUPS:
        for ch in CFG.CHANNELS:
            files.extend(scan_channel_files(root, g, ch))
    return files


# =========================
# Загрузка сырых CSV в Polars
# =========================
def load_raw_lazy(root: Path) -> pl.LazyFrame:
    files = scan_all_files(root)
    lazies = []
    for p in files:
        meta = parse_file_meta(p)
        sep = detect_delim(p)
        lf = (
            pl.scan_csv(
                p,
                separator=sep if sep != r"\s+" else None,
                has_header=True,
                ignore_errors=True,
                infer_schema_length=50,
                try_parse_dates=False,
            )
            .rename({"time_sec": "t_sec", "value": "val"})
            .select([
                pl.col("t_sec").cast(pl.Float64),
                pl.col("val").cast(pl.Float64),
            ])
            .with_columns([
                pl.lit(meta["group"]).alias("group"),
                pl.lit(meta["folder_id"]).alias("folder_id"),
                pl.lit(meta["channel"]).alias("channel"),
                pl.lit(meta["record_key"]).alias("record_key"),
                pl.lit(meta["chan_idx"]).alias("chan_idx"),
                pl.lit(meta["file_name"]).alias("file_name"),
                pl.lit(meta["file_path"]).alias("file_path"),
            ])
        )
        lazies.append(lf)

    if not lazies:
        raise FileNotFoundError("CSV файлов не найдено. Проверьте путь ROOT и структуру каталогов.")

    raw = pl.concat(lazies, how="diagonal_relaxed").filter(
        pl.col("t_sec").is_not_null() & pl.col("val").is_not_null()
    ).sort(["group", "folder_id", "record_key", "channel", "t_sec"])

    return raw


# =========================
# Выравнивание и ресемплинг
# =========================
def align_bpm_ua_asof(raw_lf: pl.LazyFrame, tol_sec: float) -> pl.LazyFrame:
    bpm = raw_lf.filter(pl.col("channel") == "bpm") \
        .rename({"val": "bpm"}) \
        .select(["group", "folder_id", "record_key", "t_sec", "bpm"])
    ua = raw_lf.filter(pl.col("channel") == "uterus") \
        .rename({"val": "ua"}) \
        .select(["group", "folder_id", "record_key", "t_sec", "ua"])

    bpm = bpm.sort(["group", "folder_id", "record_key", "t_sec"]).collect().lazy()
    ua = ua.sort(["group", "folder_id", "record_key", "t_sec"]).collect().lazy()

    aligned = bpm.join_asof(
        ua,
        on="t_sec",
        by=["group", "folder_id", "record_key"],
        strategy="nearest",
        tolerance=tol_sec,
        suffix="_ua",
    )
    
    return aligned

def to_datetime_sec(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.from_epoch((pl.col("t_sec") * 1_000_000_000).cast(pl.Int64), time_unit="ns").alias("ts")
    )

def resample_uniform_4hz(aligned_lf: pl.LazyFrame, every: str = CFG.RESAMPLE_EVERY) -> pl.LazyFrame:
    lf = to_datetime_sec(aligned_lf)
    # Ресемплим по ts отдельно bpm и ua, затем outer-join, ffill/bfill
    bpm_res = (lf.select(["group", "folder_id", "record_key", "ts", "bpm"])
                 .group_by_dynamic(index_column="ts", every=every, closed="left", group_by=["group", "folder_id", "record_key"])
                 .agg([pl.col("bpm").mean().alias("bpm")]))
    ua_res = (lf.select(["group", "folder_id", "record_key", "ts", "ua"])
                 .group_by_dynamic(index_column="ts", every=every, closed="left", group_by=["group", "folder_id", "record_key"])
                 .agg([pl.col("ua").mean().alias("ua")]))

    uni = bpm_res.join(ua_res, on=["group", "folder_id", "record_key", "ts"], how="full") \
                 .sort(["group", "folder_id", "record_key", "ts"]) \
                 .with_columns([
                     pl.col("bpm").forward_fill().backward_fill(),
                     pl.col("ua").forward_fill().backward_fill(),
                     (pl.col("ts").cast(pl.Int64) / 1_000_000_000).alias("t_sec"),
                 ])
    return uni


# =========================
# Базовый QC
# =========================
def basic_qc(lf: pl.LazyFrame) -> pl.LazyFrame:
    lo_bpm, hi_bpm = CFG.BPM_RANGE
    lo_ua, hi_ua = CFG.UA_RANGE
    out = (lf
           .with_columns([
               pl.when((pl.col("bpm") < lo_bpm) | (pl.col("bpm") > hi_bpm)).then(None).otherwise(pl.col("bpm")).alias("bpm"),
               pl.when((pl.col("ua") < lo_ua) | (pl.col("ua") > hi_ua)).then(None).otherwise(pl.col("ua")).alias("ua"),
           ])
           .with_columns([
               pl.col("bpm").interpolate(),
               pl.col("ua").interpolate(),
           ]))
    return out

# =========================
# Аномалии (детекция)
# =========================
def detect_anomalies_per_record(df: pd.DataFrame, freq_hz: float = CFG.FREQ_HZ) -> Dict:
    """
    df: pandas DataFrame одной записи/пациента с колонками:
        ['t_sec','bpm','ua'] (и можно иметь index ts)
    Возвращает словарь с метриками и списками событий.
    """
    # baseline по FHR через сглаженную медиану
    w = max(5, int(freq_hz * CFG.BASELINE_ROLL_SEC))
    bpm = df["bpm"].to_numpy()
    ua = df["ua"].to_numpy()
    if len(bpm) < w:
        return dict(
            tachy_frac=float(np.mean(bpm > CFG.TACHY_BPM)),
            brady_frac=float(np.mean(bpm < CFG.BRADY_BPM)),
            low_var_frac=np.nan,
            decel_count=0,
            decel_events=[],
            crosscorr_max=np.nan,
            crosscorr_lag_s=np.nan
        )

    # rolling median/mean (медиана устойчивее, но медленнее; возьмём медиану)
    # Используем pandas для простоты
    baseline = pd.Series(bpm).rolling(w, center=True, min_periods=w//2).median().to_numpy()
    # fallback на mean там, где baseline NaN
    bad = np.isnan(baseline)
    if bad.any():
        baseline[bad] = pd.Series(bpm).rolling(w, center=True, min_periods=w//2).mean().to_numpy()[bad]

    drop = baseline - bpm  # положительное => падение от baseline
    is_decel = drop >= CFG.DECEL_DROP_BPM

    # Детект непрерывных участков True
    # run-length encoding
    idx = np.arange(len(is_decel))
    # границы run-ов
    changes = np.diff(is_decel.astype(int), prepend=is_decel[0])
    run_id = np.cumsum(changes != 0)
    events = []
    for rid in np.unique(run_id[is_decel]):
        mask = (run_id == rid) & is_decel
        if not mask.any():
            continue
        start_i = np.argmax(mask)  # первый True
        end_i = len(mask) - 1 - np.argmax(mask[::-1])  # последний True
        # длительность
        dur_s = (end_i - start_i + 1) / freq_hz
        if dur_s < CFG.DECEL_MIN_SEC:
            continue
        # амплитуда падения
        min_bpm = bpm[start_i:end_i+1].min()
        max_drop = (baseline[start_i:end_i+1] - bpm[start_i:end_i+1]).max()
        events.append(dict(
            start_idx=int(start_i),
            end_idx=int(end_i),
            dur_s=float(dur_s),
            min_bpm=float(min_bpm),
            max_drop=float(max_drop),
        ))

    tachy_frac = float(np.mean(bpm > CFG.TACHY_BPM))
    brady_frac = float(np.mean(bpm < CFG.BRADY_BPM))

    # Вариабельность: STV как среднее |diff|
    diffs = np.diff(bpm)
    stv = np.mean(np.abs(diffs)) if len(diffs) else np.nan
    # "низкая вариабельность" можно считать, если STV<порога на окне; упростим до глобальной оценки:
    low_var_frac = float(stv < CFG.LOW_VAR_STV) if not np.isnan(stv) else np.nan

    # Кросс-корреляция UA vs -BPM (для "поздней" децелерации ожидаем лаг > 0 и отрицательную корреляцию)
    # Рассмотрим лаги +/- 120 с
    max_lag = int(120 * freq_hz)
    if len(ua) > 2 * max_lag + 1:
        x = (ua - np.nanmean(ua))
        y = -(bpm - np.nanmean(bpm))
        # Нормировка:
        sx = np.nanstd(x) + 1e-6
        sy = np.nanstd(y) + 1e-6
        x = x / sx
        y = y / sy
        corr = np.correlate(x, y, mode="full") / len(x)
        lags = np.arange(-len(x)+1, len(x))
        mask = (lags >= -max_lag) & (lags <= max_lag)
        corr = corr[mask]
        lags = lags[mask]
        best_i = int(np.nanargmax(corr))
        crosscorr_max = float(corr[best_i])
        crosscorr_lag_s = float(lags[best_i] / freq_hz)
    else:
        crosscorr_max = np.nan
        crosscorr_lag_s = np.nan

    return dict(
        tachy_frac=tachy_frac,
        brady_frac=brady_frac,
        low_var_frac=low_var_frac,
        decel_count=len(events),
        decel_events=events,
        crosscorr_max=crosscorr_max,
        crosscorr_lag_s=crosscorr_lag_s,
    )


# =========================
# Обновлённая сборка датасета
# =========================
def build_dataset(root: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    raw_lf = load_raw_lazy(root)
    aligned_lf = align_bpm_ua_asof(raw_lf, CFG.JOIN_TOL_SEC)
    uni_lf = resample_uniform_4hz(aligned_lf, CFG.RESAMPLE_EVERY)
    uni_lf = basic_qc(uni_lf)
    uni_lf = uni_lf.sort(["group", "folder_id", "record_key", "ts"])
    uni_df = uni_lf.collect(streaming=True)

    # Вместо окон — CTG-признаки уровня пациента:
    patients_df = compute_patient_features_ctg(uni_df)

    return uni_df, patients_df


def _rolling_baseline(arr: np.ndarray, freq_hz: float, win_sec: float | None = None) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return np.array([], dtype=float)
    base_sec = CFG.BASELINE_ROLL_SEC if win_sec is None else win_sec
    w = max(5, int(freq_hz * base_sec))
    s = pd.Series(arr)
    base_med = s.rolling(w, center=True, min_periods=w//2).median()
    base_mean = s.rolling(w, center=True, min_periods=w//2).mean()
    baseline = base_med.to_numpy()
    bad = np.isnan(baseline)
    if bad.any():
        baseline[bad] = base_mean.to_numpy()[bad]
    return baseline

def _find_events(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    # Возвращает список (start_idx, end_idx), где mask==True непрерывно минимум min_len
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    out = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_len:
            out.append((int(s), int(e)))
    return out

def _count_peaks(hist: np.ndarray) -> int:
    if len(hist) < 3:
        return 0
    peaks = 0
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks += 1
    return peaks

def _count_internal_zeroes(hist: np.ndarray) -> int:
    nz = np.where(hist > 0)[0]
    if len(nz) == 0:
        return int(len(hist))
    lo, hi = nz[0], nz[-1]
    return int(np.sum(hist[lo:hi+1] == 0))

def _histogram_features(values: np.ndarray, bin_width: float = 1.0) -> Dict[str, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return dict(histogram_width=np.nan, histogram_min=np.nan, histogram_max=np.nan,
                    histogram_number_of_peaks=np.nan, histogram_number_of_zeroes=np.nan,
                    histogram_mode=np.nan, histogram_mean=np.nan, histogram_median=np.nan,
                    histogram_variance=np.nan, histogram_tendency=np.nan)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        mode = vmin
        return dict(
            histogram_width=0.0, histogram_min=vmin, histogram_max=vmax,
            histogram_number_of_peaks=0, histogram_number_of_zeroes=0,
            histogram_mode=mode,
            histogram_mean=float(np.nanmean(values)),
            histogram_median=float(np.nanmedian(values)),
            histogram_variance=float(np.nanvar(values)),
            histogram_tendency=0,
        )
    bins = np.arange(np.floor(vmin), np.ceil(vmax) + bin_width, bin_width)
    hist, edges = np.histogram(values, bins=bins)
    width = vmax - vmin
    num_peaks = _count_peaks(hist)
    num_zeroes = _count_internal_zeroes(hist)
    imax = int(np.argmax(hist)) if hist.size else 0
    mode_val = float(0.5 * (edges[imax] + edges[imax+1])) if hist.size else float(np.nan)
    mean_val = float(np.nanmean(values))
    median_val = float(np.nanmedian(values))
    var_val = float(np.nanvar(values))
    x = np.arange(len(values), dtype=float)
    try:
        slope = np.polyfit(x, values, 1)[0]
    except Exception:
        slope = 0.0
    thr = 0.01  # bpm/сек
    tendency = 1 if slope > thr else (-1 if slope < -thr else 0)

    return dict(
        histogram_width=float(width),
        histogram_min=vmin,
        histogram_max=vmax,
        histogram_number_of_peaks=int(num_peaks),
        histogram_number_of_zeroes=int(num_zeroes),
        histogram_mode=mode_val,
        histogram_mean=mean_val,
        histogram_median=median_val,
        histogram_variance=var_val,
        histogram_tendency=int(tendency),
    )

def compute_ctg_features_for_patient(df: pd.DataFrame, freq_hz: float = CFG.FREQ_HZ) -> Dict[str, float]:
    out = {k: np.nan for k in CFG.CTG_FEATURES}
    if df is None or len(df) == 0:
        return out

    df = df.sort_values("t_sec")
    bpm = df["bpm"].to_numpy(dtype=float)
    ua = df["ua"].to_numpy(dtype=float) if "ua" in df.columns else np.full_like(bpm, np.nan)
    n = len(bpm)
    F = int(freq_hz)

    # Длительность записи (для "per second")
    if "t_sec" in df.columns and df["t_sec"].notna().any():
        duration_sec = float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0])
        if not np.isfinite(duration_sec) or duration_sec <= 0:
            duration_sec = n / freq_hz
    else:
        duration_sec = n / freq_hz
    duration_sec = max(duration_sec, 1e-6)

    # Базальная линия (60с) — для baseline value и гистограмм
    baseline = _rolling_baseline(bpm, freq_hz, win_sec=CFG.BASELINE_ROLL_SEC)
    out["baseline value"] = float(np.nanmedian(baseline))

    # Более “быстрая” базовая линия (30с) — для событий
    event_bl = _rolling_baseline(bpm, freq_hz, win_sec=CFG.BASELINE_ROLL_SEC_EVENT)

    # 1-секундные агрегаты (для STV/LTV и гистограмм baseline)
    if n >= F:
        n1 = (n // F) * F
        bpm_1s = bpm[:n1].reshape(-1, F).mean(axis=1)
        bl_1s  = event_bl[:n1].reshape(-1, F).mean(axis=1)
    else:
        bpm_1s = bpm.copy()
        bl_1s  = event_bl.copy()

    # Первичный проход детекции
    def detect_events(acc_thr, acc_min_s, dec_light_thr, dec_severe_thr, dec_min_s, dec_prol_s,
                      ua_thr, ua_min_s):
        acc_mask = (bpm - event_bl) >= acc_thr
        dec_mask = (event_bl - bpm) >= dec_light_thr
        acc_events = _find_events(acc_mask, int(acc_min_s * freq_hz))
        dec_events = _find_events(dec_mask, int(dec_min_s * freq_hz))

        light_dec = severe_dec = prolonged_dec = 0
        for s, e in dec_events:
            dur_s = (e - s + 1) / freq_hz
            drop = (event_bl[s:e+1] - bpm[s:e+1])
            max_drop = float(np.nanmax(drop)) if drop.size else 0.0
            if (dec_light_thr <= max_drop < dec_severe_thr) and (dur_s >= dec_min_s):
                light_dec += 1
            if (max_drop >= dec_severe_thr) and (dur_s >= dec_min_s):
                severe_dec += 1
            if dur_s >= dec_prol_s:
                prolonged_dec += 1

        if not np.all(np.isnan(ua)):
            ua_base = _rolling_baseline(ua, freq_hz, win_sec=CFG.BASELINE_ROLL_SEC_EVENT)
            ua_mask = (ua - ua_base) >= ua_thr
            ua_events = _find_events(ua_mask, int(ua_min_s * freq_hz))
        else:
            ua_events = []

        # fetal_movement — акцелерации, не прилегающие к схваткам ±10с
        fm_count = 0
        if len(acc_events):
            ua_intervals = [(s / freq_hz, e / freq_hz) for s, e in ua_events]
            for s, e in acc_events:
                mid_t = 0.5 * (s + e) / freq_hz
                near_ua = any((mid_t >= us - 10.0) and (mid_t <= ue + 10.0) for us, ue in ua_intervals)
                if not near_ua:
                    fm_count += 1

        return dict(
            acc=len(acc_events),
            fm=fm_count,
            ua=len(ua_events),
            light=light_dec,
            severe=severe_dec,
            prolong=prolonged_dec,
        )

    # Детект с базовыми порогами
    ev = detect_events(
        CFG.ACC_RISE_BPM, CFG.ACC_MIN_SEC,
        CFG.DECEL_LIGHT_DROP_BPM, CFG.DECEL_SEVERE_DROP_BPM, CFG.DECEL_MIN_SEC, CFG.DECEL_PROLONGED_SEC,
        CFG.UA_CONTR_MIN_RISE, CFG.UA_CONTR_MIN_SEC
    )

    # Fallback: если вообще ничего не нашли — слегка ослабим пороги (помогает на «плоских» сериях)
    if (ev["acc"] + ev["ua"] + ev["light"] + ev["severe"] + ev["prolong"]) == 0:
        ev = detect_events(
            acc_thr=max(6.0, CFG.ACC_RISE_BPM * 0.7),
            acc_min_s=max(8.0, CFG.ACC_MIN_SEC * 0.7),
            dec_light_thr=max(10.0, CFG.DECEL_LIGHT_DROP_BPM * 0.8),
            dec_severe_thr=max(18.0, CFG.DECEL_SEVERE_DROP_BPM * 0.8),
            dec_min_s=max(8.0, CFG.DECEL_MIN_SEC * 0.7),
            dec_prol_s=CFG.DECEL_PROLONGED_SEC,  # этот не ослабляем
            ua_thr=max(5.0, CFG.UA_CONTR_MIN_RISE * 0.5),
            ua_min_s=max(8.0, CFG.UA_CONTR_MIN_SEC * 0.7),
        )

    # Преобразуем в “per second”
    out["accelerations"]           = float(ev["acc"]     / duration_sec)
    out["fetal_movement"]          = float(ev["fm"]      / duration_sec)
    out["uterine_contractions"]    = float(ev["ua"]      / duration_sec)
    out["light_decelerations"]     = float(ev["light"]   / duration_sec)
    out["severe_decelerations"]    = float(ev["severe"]  / duration_sec)
    out["prolongued_decelerations"]= float(ev["prolong"] / duration_sec)

    # STV/LTV
    if len(bpm_1s) >= 2:
        stv_steps = np.abs(np.diff(bpm_1s))
        out["mean_value_of_short_term_variability"] = float(np.nanmean(stv_steps))
        out["abnormal_short_term_variability"] = float(np.mean(stv_steps < CFG.LOW_VAR_STV) * 100.0)
    else:
        out["mean_value_of_short_term_variability"] = np.nan
        out["abnormal_short_term_variability"] = np.nan

    if len(bpm_1s) >= 60:
        m = (len(bpm_1s) // 60) * 60
        mins = bpm_1s[:m].reshape(-1, 60)
        ltv = np.nanstd(mins, axis=1)
        out["mean_value_of_long_term_variability"] = float(np.nanmean(ltv))
        out["percentage_of_time_with_abnormal_long_term_variability"] = float(np.mean(ltv < 5.0) * 100.0)
    else:
        out["mean_value_of_long_term_variability"] = np.nan
        out["percentage_of_time_with_abnormal_long_term_variability"] = np.nan

    # Гистограммы (по 1-сек. baseline)
    hist_feats = _histogram_features(bl_1s if len(bl_1s) else event_bl, bin_width=1.0)
    out.update(hist_feats)
    return out


def compute_patient_features_ctg(uni_df: pl.DataFrame) -> pl.DataFrame:
    """
    На входе — униформный ряд (после ресемплинга+QC).
    На выходе — patients_df с CTG-признаками (события в секунду, проценты — [0..100])
    + мета (group, folder_id) и target_hypoxia.
    """
    rows = []
    parts = uni_df.partition_by(["group", "folder_id"], as_dict=True)
    for (g, f), pdf in parts.items():
        d = pdf[["t_sec", "bpm", "ua"]].to_pandas()
        d = d.dropna(subset=["bpm"])  # ua может быть частично NaN
        feats = compute_ctg_features_for_patient(d, CFG.FREQ_HZ)
        rows.append({
            "group": g,
            "folder_id": f,
            **feats,
            "target_hypoxia": 1 if g == "hypoxia" else 0,
        })
    # Сохраняем порядок колонок: meta + CTG_FEATURES + target
    cols_order = ["group", "folder_id"] + CFG.CTG_FEATURES + ["target_hypoxia"]
    df_pl = pl.DataFrame(rows)
    return df_pl.select(cols_order)


# =========================
# Обновлённый train_and_evaluate под CTG-признаки (patient-level)
# =========================
def train_and_evaluate(patients_df: pl.DataFrame):
    zero_cols = [
        col for col in patients_df.columns
        if patients_df.schema[col].is_numeric()  # Фильтруем только числовые типы
        and patients_df.select(pl.col(col).eq(0).all()).item()
    ]
    
    df = patients_df.to_pandas()
    y = df["target_hypoxia"].values
    groups = df["folder_id"].values

    feat_cols = [c for c in CFG.CTG_FEATURES if c not in zero_cols]
    
    X = df[feat_cols]

    sgkf = StratifiedGroupKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)

    print("=== Hyperparameter Tuning for LightGBM with Optuna ===")
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': CFG.SEED,
            'n_jobs': -1,
            'verbose': -1,
        }
        scores = []
        for _, (tr, va) in enumerate(sgkf.split(X, y, groups)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            model = LGBMClassifier(**params)
            model.fit(X_tr_sc, y_tr)
            p = model.predict_proba(X_va_sc)[:, 1]
            scores.append(roc_auc_score(y_va, p))
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', study_name='lgbm_hypoxia_tuning')
    study.optimize(objective, n_trials=100)

    best_lgbm_params = study.best_params
    print(f"\nBest parameters found by Optuna: {best_lgbm_params}")
    print(f"Best CV AUC from Optuna study: {study.best_value:.4f}\n")

    print("=== Patient-Level CV: LightGBM with Tuned Parameters ===")
    lgbm_aucs, lgbm_aps = [], []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        model = LGBMClassifier(**best_lgbm_params, random_state=CFG.SEED,
                               objective='binary', metric='auc', verbose=-1)
        model.fit(X_tr_sc, y_tr)
        p = model.predict_proba(X_va_sc)[:, 1]
        auc = roc_auc_score(y_va, p)
        ap = average_precision_score(y_va, p)
        lgbm_aucs.append(auc); lgbm_aps.append(ap)
        print(f"Fold {fold}: AUC={auc:.4f}, AP={ap:.4f}")
    print(f"\nCV LGBM: ROC-AUC={np.mean(lgbm_aucs):.4f} (± {np.std(lgbm_aucs):.4f}), "
          f"PR-AUC={np.mean(lgbm_aps):.4f} (± {np.std(lgbm_aps):.4f})\n")

    print("=== Training final LightGBM model on all data ===")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = LGBMClassifier(**best_lgbm_params, random_state=CFG.SEED,
                                 objective='binary', metric='auc', verbose=-1)
    final_model.fit(X_scaled, y)
    print("Final model training complete.")

    return final_model, final_scaler, feat_cols

def predict_patient_risk(model, feature_cols: List[str], patients_df: pl.DataFrame, final_scaler: StandardScaler) -> pd.DataFrame:
    df = patients_df.to_pandas()
    X = df[feature_cols]
    X_sc = final_scaler.transform(X)
    proba = model.predict_proba(X_sc)[:, 1]
    out = df[["group", "folder_id"]].copy()
    out["risk_hypoxia"] = proba
    return out.sort_values(["group", "risk_hypoxia"], ascending=[True, False])


# =========================
# Аномалии на уровне пациента
# =========================
def anomalies_for_all_patients(uni_df: pl.DataFrame) -> pd.DataFrame:
    out_records = []
    # разбиваем на подтаблицы по group и folder_id
    for sub_df in uni_df.partition_by(["group", "folder_id"], as_dict=True).items():
        (g, f), pdf = sub_df
        
        d = pdf[["t_sec", "bpm", "ua"]].to_pandas()
        d = d.dropna(subset=["bpm", "ua"])
        if len(d) == 0:
            continue
        res = detect_anomalies_per_record(d, CFG.FREQ_HZ)
        out_records.append({
            "group": g,
            "folder_id": f,
            **{k: v for k, v in res.items() if k != "decel_events"}
        })
    return pd.DataFrame(out_records)


root = CFG.ROOT

# 1) Подготовка данных
uni_df, patients_df = build_dataset(root)
print("Uniform rows:", len(uni_df))
print("Patients:", len(patients_df))
print("Patients CTG features preview:")



patients_df


extra_df = pl.read_csv(CFG.EXTRA_DATA_PATH)
extra_df = extra_df.rename({"fetal_health": "target_hypoxia"})
fl_to_int_list = ["histogram_number_of_peaks", "histogram_number_of_zeroes", "histogram_tendency", "target_hypoxia"]
extra_df = extra_df.with_columns(pl.col(col).cast(pl.Int64) for col in fl_to_int_list)

extra_df_hypoxia = extra_df.filter(pl.col("target_hypoxia") == 3).with_columns(
    (pl.col("target_hypoxia") - 2).alias("target_hypoxia")
)

extra_df_regular = extra_df.filter(pl.col("target_hypoxia") == 1).head(106).with_columns(
    (pl.col("target_hypoxia") - 1).alias("target_hypoxia")
)

patients_df = pl.concat([patients_df, extra_df_hypoxia, extra_df_regular], how="diagonal")

print(patients_df.shape)

patients_df = patients_df.with_columns(
    pl.when(pl.col("group").is_null() & (pl.col("target_hypoxia") == 1))
    .then(pl.lit("hypoxia"))
    .when(pl.col("group").is_null() & (pl.col("target_hypoxia") == 0))
    .then(pl.lit("regular"))
    .otherwise(pl.col("group"))
    .alias("group")
)

max_folder_id = patients_df.select(
    pl.col("folder_id").drop_nulls().cast(pl.Int64).max()
).item()

patients_df = patients_df.with_columns(
    pl.when(pl.col("folder_id").is_null())
    .then(
        (max_folder_id + pl.col("folder_id").is_null().cum_sum()).cast(pl.String)
    )
    .otherwise(pl.col("folder_id"))
    .alias("folder_id")
)

patients_df = patients_df.with_columns(
    pl.when(pl.col("folder_id").is_null())
    .then(
        (max_folder_id + 1 + 
         pl.col("folder_id").is_null().cum_sum().over(pl.lit(1)) - 1
        ).cast(pl.String)
    )
    .otherwise(pl.col("folder_id"))
    .alias("folder_id")
)



# 2) Обучение и предсказание риска гипоксии (patient-level)
model, final_scaler, feat_cols = train_and_evaluate(patients_df)
risk_df = predict_patient_risk(model, feat_cols, patients_df, final_scaler)
print("\nРиск гипоксии по пациентам:")
print(risk_df.head(20))

# 3) Детектор аномалий (для отчётов/мониторинга)
anom_df = anomalies_for_all_patients(uni_df)
print("\nАномалии (агрегировано по пациенту):")
print(anom_df.head(10))

# 4) Сохранение результатов (опционально)
# pl.from_pandas(risk_df).write_csv("patient_risk.csv")
# patients_df.write_parquet("patients_ctg_features.parquet", compression="zstd")
# anom_df.to_json("anomalies.json", orient="records", force_ascii=False, indent=2)

import joblib

joblib.dump({
    "model": model,
    "scaler": final_scaler,
    "features": feat_cols
}, "patient_risk_pipeline.pkl")