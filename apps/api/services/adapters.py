from __future__ import annotations
from io import StringIO
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timezone, timedelta

def _auto_sep(sample: str) -> str:
    if ";" in sample: return ";"
    if "\t" in sample: return "\t"
    return ","

def parse_csv_to_samples(content: bytes, session_id: str, channel: str = None) -> List[Dict]:
    """
    Парсит CSV файл в сэмплы.
    
    Args:
        content: Байты CSV файла
        session_id: ID сессии
        channel: Канал данных ("bpm" или "uterus"). Если None, пытается определить автоматически
    
    Ожидаемые столбцы:
    - ts (ISO8601) или time_sec (float seconds)
    - value (float) - будет переименована в bpm или uterus в зависимости от channel
    """
    text = content.decode("utf-8", errors="ignore")
    sample = text.splitlines()[0] if text else ""
    sep = _auto_sep(sample)
    df = pd.read_csv(StringIO(text), sep=sep)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # Обработка временных меток
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    elif "time_sec" in df.columns:
        # Конвертируем относительное время в абсолютное
        t0 = datetime.now(timezone.utc) - timedelta(seconds=float(df["time_sec"].max()))
        df["ts"] = [t0 + timedelta(seconds=float(s)) for s in df["time_sec"].values]
    else:
        raise ValueError("CSV must contain 'ts' or 'time_sec' column")

    # Определение канала и переименование колонки value
    if channel:
        # Если канал задан явно, используем его
        if "value" not in df.columns:
            raise ValueError("CSV must contain 'value' column")
        
        df["channel"] = channel
        # Переименовываем value в соответствии с каналом
        if channel == "bpm":
            df = df.rename(columns={"value": "bpm"})
            value_col = "bpm"
        elif channel == "uterus":
            df = df.rename(columns={"value": "ua"})
            value_col = "ua"
        else:
            raise ValueError(f"Unknown channel: {channel}. Expected 'bpm' or 'uterus'")
    else:
        # Автоматическое определение канала (legacy behavior)
        if "channel" not in df.columns:
            if {"bpm","value"} <= set(df.columns):
                df = df.rename(columns={"value":"bpm"})
                df["channel"] = "bpm"
                value_col = "bpm"
            elif {"ua","value"} <= set(df.columns):
                df = df.rename(columns={"value":"ua"})
                df["channel"] = "uterus"
                value_col = "ua"
            else:
                raise ValueError("CSV must contain 'channel' column or be a single-channel file with 'value' column")
        else:
            value_col = "value"

    df = df.dropna(subset=["ts", "channel"])
    
    # Формируем список сэмплов
    samples = []
    for r in df.itertuples(index=False):
        ts = pd.to_datetime(r.ts).isoformat()
        ch = str(r.channel)
        
        # Получаем значение в зависимости от канала
        if ch == "bpm":
            val = getattr(r, "bpm", None) or getattr(r, "value", None)
        elif ch == "uterus":
            val = getattr(r, "ua", None) or getattr(r, "value", None)
        else:
            val = getattr(r, "value", None)
        
        if val is not None:
            samples.append({
                "ts": ts,
                "channel": ch,
                "value": float(val)
            })
    
    return samples

def parse_fhir_observations(payload: Dict[str, Any]) -> List[Dict]:
    """
    Поддержка FHIR Observations
    """
    def detect_channel(obs: Dict[str, Any]) -> str | None:
        codings = (obs.get("code", {}).get("coding") or []) + sum([c.get("coding", []) for c in obs.get("component", [])], [])
        tokens = { (c.get("code") or "").upper() for c in codings }
        txt = ((obs.get("code", {}) or {}).get("text") or "").upper()

        bpm_codes = {"FHR","BPM","8867-4","MDC_ECG_HEART_RATE","56074-1"}
        ua_codes  = {"UA","TOCO","8687-2","MDC_CONTRACT_HEART","UAW","UTERUS","UTERINE"}

        if tokens & bpm_codes or "FETAL" in txt and ("HR" in txt or "HEART" in txt):
            return "bpm"
        if tokens & ua_codes or "UTER" in txt or "TOCO" in txt:
            return "uterus"
        return None

    def get_effective_start(obs: Dict[str, Any]) -> datetime:
        eff = obs.get("effectiveDateTime") or obs.get("effectiveInstant") \
              or (obs.get("effectivePeriod") or {}).get("start") \
              or obs.get("issued")
        if eff:
            return pd.to_datetime(eff, utc=True).to_pydatetime()
        return datetime.now(timezone.utc)

    out: List[Dict] = []

    # Bundle?
    if payload.get("resourceType") == "Bundle":
        for entry in payload.get("entry", []):
            res = entry.get("resource", {})
            if res.get("resourceType") == "Observation":
                out.extend(parse_fhir_observations(res))
        return out

    # Single Observation
    obs = payload
    ch = detect_channel(obs)
    if ch is None:
        ch = "bpm"
    start_ts = get_effective_start(obs)

    if "valueSampledData" in obs:
        sd = obs["valueSampledData"]
        data_str = (sd.get("data") or "").strip()
        if not data_str:
            return out
        period_ms = sd.get("period") or 250.0
        factor = sd.get("factor") or 1.0
        origin = (sd.get("origin") or {}).get("value") or 0.0
        vals = []
        for token in data_str.replace(",", " ").split():
            try:
                vals.append(float(token))
            except ValueError:
                continue
        for i, raw in enumerate(vals):
            v = origin + raw * factor
            ts = (start_ts + pd.to_timedelta(i * period_ms, unit="ms")).isoformat()
            out.append({"ts": ts, "channel": ch, "value": float(v)})
        return out

    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        v = float(vq.get("value"))
        out.append({"ts": start_ts.isoformat(), "channel": ch, "value": v})
        return out

    for comp in obs.get("component", []):
        cch = detect_channel({"code": comp.get("code")})
        if "valueQuantity" in comp and cch:
            v = float(comp["valueQuantity"]["value"])
            out.append({"ts": start_ts.isoformat(), "channel": cch, "value": v})

    return out