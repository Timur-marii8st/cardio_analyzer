from __future__ import annotations
import typer
from pathlib import Path
import polars as pl
from packages.ctg_ml.data_build import build_patients_df
from packages.ctg_ml.train import train_patient_model

app = typer.Typer(add_completion=False)

@app.command()
def build_dataset(
    data_root: str = typer.Option(..., help="Корень датасета (папки hypoxia/regular/*/{bpm,uterus}/*.csv)"),
    extra_csv: str = typer.Option(None, help="Путь к fetal_health.csv (опционально)"),
    out_path: str = typer.Option("patients_ctg_features.parquet", help="Куда сохранить parquet")
):
    df = build_patients_df(data_root, extra_csv)
    df.write_parquet(out_path)
    typer.echo(f"Saved patients_df to {out_path} with shape {df.shape}")

@app.command()
def train(
    patients_parquet: str = typer.Option(..., help="Файл parquet с patients_df"),
    artifacts_path: str = typer.Option("artifacts/model.joblib", help="Куда сохранить артефакт")
):
    df = pl.read_parquet(patients_parquet).to_pandas()
    artifact = train_patient_model(df, groups_col="folder_id", target_col="target_hypoxia", save_path=artifacts_path)
    typer.echo(f"Saved model to {artifacts_path} with features: {artifact.feature_cols}")

if __name__ == "__main__":
    app()