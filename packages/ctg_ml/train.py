from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
from dataclasses import dataclass
from typing import Tuple, List, Optional
from .artifacts import ModelArtifact, save_artifact

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

def detect_constant_zero_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    # исправление: корректная проверка нулевых столбцов
    zero_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and np.all((s.fillna(0.0) == 0.0).values):
            zero_cols.append(c)
    return zero_cols

def train_patient_model(patients_df: pd.DataFrame, groups_col: str = "folder_id",
                        target_col: str = "target_hypoxia", save_path: str = "artifacts/model.joblib") -> ModelArtifact:
    y = patients_df[target_col].astype(int).values
    groups = patients_df[groups_col].astype(str).values
    feat_cols = [c for c in CTG_FEATURES if c in patients_df.columns]
    feat_cols = [c for c in feat_cols if c not in detect_constant_zero_cols(patients_df, feat_cols)]
    X = patients_df[feat_cols].copy()

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    aucs, aps = [], []

    # Исправление: LightGBM не требует StandardScaler; добавляем class_weight и early_stopping
    params = dict(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric="auc",
                  verbose=False,
                  early_stopping_rounds=100)
        p = model.predict_proba(X_va)[:,1]
        oof[va] = p
        aucs.append(roc_auc_score(y_va, p))
        aps.append(average_precision_score(y_va, p))

    print(f"CV: ROC-AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}, PR-AUC={np.mean(aps):.4f} ± {np.std(aps):.4f}")

    # Калибровка по OOF предсказаниям (исотоник)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    brier = brier_score_loss(y, calibrator.predict(oof))
    print(f"Brier score (OOF): {brier:.4f}")

    # Обучим финальную модель на всех данных
    final_model = LGBMClassifier(**params)
    final_model.fit(X, y)
    artifact = ModelArtifact(model=final_model, feature_cols=feat_cols, calibrator=calibrator, version="0.1.0")
    save_artifact(save_path, artifact)
    print(f"Saved model artifact to {save_path}")
    return artifact