from __future__ import annotations
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Список ожидаемых фичей (из вашего обучения)
EXPECTED_FEATURES = [
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

class InferenceService:
    def __init__(self, artifacts_path: str):
        """
        Инициализация сервиса инференса.
        
        Args:
            artifacts_path: Путь к файлу модели (.pkl или .joblib)
        """
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        if os.path.exists(artifacts_path):
            try:
                logger.info(f"Loading ML model from {artifacts_path}")
                pipeline = joblib.load(artifacts_path)
                
                # Проверяем формат файла
                if isinstance(pipeline, dict):
                    # Новый формат: {"model": ..., "scaler": ..., "features": ...}
                    self.model = pipeline.get("model")
                    self.scaler = pipeline.get("scaler")
                    self.feature_cols = pipeline.get("features") or EXPECTED_FEATURES
                    
                    if self.model is None:
                        raise ValueError("Model not found in pipeline dict")
                    
                    logger.info(f"✓ Loaded model with {len(self.feature_cols)} features")
                    logger.info(f"✓ Scaler: {'Present' if self.scaler else 'Not present'}")
                    
                else:
                    # Старый формат (ModelArtifact или просто модель)
                    if hasattr(pipeline, 'model'):
                        self.model = pipeline.model
                        self.scaler = getattr(pipeline, 'calibrator', None)
                        self.feature_cols = getattr(pipeline, 'feature_cols', EXPECTED_FEATURES)
                    else:
                        self.model = pipeline
                        self.feature_cols = EXPECTED_FEATURES
                    
                    logger.info(f"✓ Loaded legacy model format")
                
            except Exception as e:
                logger.error(f"Failed to load model from {artifacts_path}: {e}")
                logger.warning("Falling back to rule-based prediction")
                self.model = None
        else:
            logger.warning(f"Model file not found: {artifacts_path}")
            logger.warning("Using rule-based prediction")

    def predict(self, features: Dict[str, float]) -> float:
        """
        Предсказание вероятности гипоксии.
        
        Args:
            features: Словарь с CTG-признаками
            
        Returns:
            Вероятность гипоксии (0.0 - 1.0)
        """
        if self.model is None:
            # Fallback на rule-based предикцию
            return self._rule_based_predict(features)
        
        try:
            # Подготовка данных для модели
            df = pd.DataFrame([features])
            
            # Убедимся что все нужные колонки присутствуют
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Выбираем только нужные колонки в правильном порядке
            X = df[self.feature_cols]
            
            # Применяем scaler если есть
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Предсказание
            prob = self.model.predict_proba(X_scaled)[:, 1][0]
            
            # Клипим значение в [0, 1]
            prob = float(np.clip(prob, 0.0, 1.0))
            
            logger.debug(f"Model prediction: {prob:.4f}")
            return prob
            
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            logger.warning("Falling back to rule-based prediction")
            return self._rule_based_predict(features)
    
    def _rule_based_predict(self, features: Dict[str, float]) -> float:
        """
        Простое rule-based предсказание как fallback.
        """
        import math
        
        # Берем ключевые признаки с дефолтными значениями
        light_dec = features.get("light_decelerations", 0.0)
        severe_dec = features.get("severe_decelerations", 0.0)
        abnormal_stv = features.get("abnormal_short_term_variability", 0.0)
        baseline = features.get("baseline value", 140.0)
        
        # Взвешенная сумма факторов риска
        dec_score = (light_dec + 2.0 * severe_dec) * 300.0
        low_var_score = (abnormal_stv / 100.0) * 0.5
        tachy_score = 0.2 if baseline > 160 else 0.0
        
        # Логистическая функция
        x = dec_score + low_var_score + tachy_score
        prob = 1.0 / (1.0 + math.exp(-(x - 0.5)))
        
        return float(np.clip(prob, 0.0, 1.0))