from __future__ import annotations
import joblib
from typing import Any, Dict

class ModelArtifact:
    def __init__(self, model, feature_cols, calibrator=None, version:str="0.1.0"):
        self.model = model
        self.feature_cols = feature_cols
        self.calibrator = calibrator
        self.version = version

    def predict_proba(self, X_df) -> float | list[float]:
        import numpy as np
        p = self.model.predict_proba(X_df[self.feature_cols])[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p.reshape(-1,1)).ravel()
        # клип на [0,1]
        return np.clip(p, 0.0, 1.0)

def save_artifact(path: str, artifact: ModelArtifact):
    joblib.dump(artifact, path)

def load_artifact(path: str) -> ModelArtifact:
    return joblib.load(path)