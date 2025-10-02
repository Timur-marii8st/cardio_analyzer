from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import polars as pl

from packages.ctg_core.features import compute_ctg_features_window, CTG_FEATURES

GROUPS = ["hypoxia", "regular"]
CHANNELS = ["bpm", "uterus"]

def detect_delim(path: Path) -> str:
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    if not head:
        return ","
    h = head[0]
    if ";" in h: return ";"
    if "," in h: return ","
    if "\t" in h: return "\t"
    return r"\s+"

def parse_file_meta(path: Path) -> dict:
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
        "group": group, "folder_id": folder_id, "channel": channel,
        "record_key": record_key, "chan_idx": chan_idx,
        "file_name": path.name, "file_path": str(path),
    }

def scan_channel_files(root: Path, group: str, channel: str) -> List[Path]:
    base = root / group
    if not base.exists(): return []
    return list(base.glob(f"*/{channel}/*.csv"))

def scan_all_files(root: Path) -> List[Path]:
    files = []
    for g in GROUPS:
        for ch in CHANNELS:
            files.extend(scan_channel_files(root, g, ch))
    return files

def load_raw_lazy(root: Path) -> pl.LazyFrame:
    files = scan_all_files(root)
    if not files:
        raise FileNotFoundError(f"No CSV files under {root}")
    lazies = []
    for p in files:
        meta = parse_file_meta(p)
        sep = detect_delim(p)
        lf = (
            pl.scan_csv(
                p, separator=None if sep == r"\s+" else sep,
                has_header=True, ignore_errors=True,
                infer_schema_length=50, try_parse_dates=False,
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
    raw = pl.concat(lazies, how="diagonal_relaxed").filter(
        pl.col("t_sec").is_not_null() & pl.col("val").is_not_null()
    ).sort(["group","folder_id","record_key","channel","t_sec"])
    return raw

def align_bpm_ua_asof(raw_lf: pl.LazyFrame, tol_sec: float = 0.5) -> pl.LazyFrame:
    bpm = raw_lf.filter(pl.col("channel") == "bpm") \
        .rename({"val":"bpm"}).select(["group","folder_id","record_key","t_sec","bpm"])
    ua  = raw_lf.filter(pl.col("channel") == "uterus") \
        .rename({"val":"ua"}).select(["group","folder_id","record_key","t_sec","ua"])
    bpm = bpm.sort(["group","folder_id","record_key","t_sec"]).collect().lazy()
    ua  = ua.sort(["group","folder_id","record_key","t_sec"]).collect().lazy()
    aligned = bpm.join_asof(
        ua, on="t_sec", by=["group","folder_id","record_key"],
        strategy="nearest", tolerance=tol_sec, suffix="_ua"
    )
    return aligned

def to_datetime_sec(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.from_epoch((pl.col("t_sec") * 1_000_000_000).cast(pl.Int64), time_unit="ns").alias("ts")
    )

def resample_uniform_4hz(aligned_lf: pl.LazyFrame, every: str = "250ms") -> pl.LazyFrame:
    lf = to_datetime_sec(aligned_lf)
    bpm_res = (lf.select(["group","folder_id","record_key","ts","bpm"])
                 .group_by_dynamic(index_column="ts", every=every, closed="left",
                                   group_by=["group","folder_id","record_key"])
                 .agg([pl.col("bpm").mean().alias("bpm")]))
    ua_res  = (lf.select(["group","folder_id","record_key","ts","ua"])
                 .group_by_dynamic(index_column="ts", every=every, closed="left",
                                   group_by=["group","folder_id","record_key"])
                 .agg([pl.col("ua").mean().alias("ua")]))
    uni = (bpm_res.join(ua_res, on=["group","folder_id","record_key","ts"], how="full")
                 .sort(["group","folder_id","record_key","ts"])
                 .with_columns([
                     pl.col("bpm").forward_fill().backward_fill(),
                     pl.col("ua").forward_fill().backward_fill(),
                     (pl.col("ts").cast(pl.Int64) / 1_000_000_000).alias("t_sec"),
                 ]))
    return uni

def basic_qc_pl(lf: pl.LazyFrame) -> pl.LazyFrame:
    lo_bpm, hi_bpm = 50, 220
    lo_ua, hi_ua   = 0, 120
    out = (lf
           .with_columns([
               pl.when((pl.col("bpm") < lo_bpm) | (pl.col("bpm") > hi_bpm)).then(None).otherwise(pl.col("bpm")).alias("bpm"),
               pl.when((pl.col("ua")  < lo_ua)  | (pl.col("ua")  > hi_ua)).then(None).otherwise(pl.col("ua")).alias("ua"),
           ])
           .with_columns([
               pl.col("bpm").interpolate(),
               pl.col("ua").interpolate(),
           ]))
    return out

def compute_patient_features_ctg(uni_df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (g, f), part in uni_df.partition_by(["group","folder_id"], as_dict=True).items():
        pdf = part[["t_sec","bpm","ua"]].to_pandas()
        pdf = pdf.dropna(subset=["bpm"])
        feats = compute_ctg_features_window(pdf)
        rows.append({
            "group": g, "folder_id": f,
            **feats,
            "target_hypoxia": 1 if g == "hypoxia" else 0,
        })
    cols = ["group","folder_id"] + CTG_FEATURES + ["target_hypoxia"]
    return pl.DataFrame(rows).select(cols)

def build_patients_df(root: str, extra_csv: Optional[str] = None) -> pl.DataFrame:
    raw = load_raw_lazy(Path(root))
    aligned = align_bpm_ua_asof(raw, tol_sec=0.5)
    uni = resample_uniform_4hz(aligned, every="250ms")
    uni = basic_qc_pl(uni).sort(["group","folder_id","record_key","ts"])
    uni_df = uni.collect(streaming=True)
    patients_df = compute_patient_features_ctg(uni_df)

    if extra_csv:
        extra = pl.read_csv(extra_csv)
        # датасет fetal_health.csv: 1=Normal, 2=Suspect, 3=Pathological
        extra = extra.rename({"fetal_health": "target_hypoxia"})
        # Бинаризуем: 1->0 (regular), 3->1 (hypoxia); 2 (suspect) — опционально выбрасываем
        extra = extra.filter(pl.col("target_hypoxia").is_in([1,3]))
        extra = extra.with_columns(
            (pl.col("target_hypoxia") == 3).cast(pl.Int64).alias("target_hypoxia")
        )
        # Добавим group/folder_id
        extra = extra.with_columns([
            pl.when(pl.col("target_hypoxia") == 1).then(pl.lit("hypoxia")).otherwise(pl.lit("regular")).alias("group"),
            (pl.int_range(0, pl.len()).cast(pl.Int64) + 1_000_000).cast(pl.Utf8).alias("folder_id"),
        ])
        cols = ["group","folder_id"] + CTG_FEATURES + ["target_hypoxia"]
        extra = extra.select(cols)
        patients_df = pl.concat([patients_df, extra], how="diagonal")
    return patients_df