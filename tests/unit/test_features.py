import numpy as np
import pandas as pd
from packages.ctg_core.features import compute_ctg_features_window, CTG_FEATURES
from packages.ctg_core.config import ml_settings

class TestFeatureComputation:
    
    def test_features_empty_data(self):
        """Test feature computation with empty DataFrame"""
        df = pd.DataFrame()
        features = compute_ctg_features_window(df)
        
        # All features should be present but NaN
        for feat in CTG_FEATURES:
            assert feat in features
            assert np.isnan(features[feat])
    
    def test_baseline_value(self):
        """Test baseline value computation"""
        n_samples = 2400  # 10 minutes at 4Hz
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        bpm = np.full(n_samples, 140.0)
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        features = compute_ctg_features_window(df)
        
        assert abs(features["baseline value"] - 140.0) < 1.0
    
    def test_acceleration_detection(self):
        """Test acceleration detection"""
        n_samples = 2400
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        bpm = np.full(n_samples, 140.0)
        
        # Add acceleration: 15 bpm rise for 15 seconds
        acc_start = 1200  # 5 minutes
        acc_duration = int(15 * ml_settings.target_freq_hz)  # 15 seconds
        bpm[acc_start:acc_start + acc_duration] += 15.0
        
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        features = compute_ctg_features_window(df)
        
        assert features["accelerations"] > 0
    
    def test_histogram_features(self):
        """Test histogram feature computation"""
        n_samples = 2400
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        
        # Create signal with known distribution
        bpm = 140.0 + 10 * np.sin(2 * np.pi * t_sec / 120)  # Sinusoidal
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        features = compute_ctg_features_window(df)
        
        assert "histogram_min" in features
        assert "histogram_max" in features
        assert "histogram_width" in features
        assert features["histogram_width"] > 0
        assert features["histogram_min"] < 140
        assert features["histogram_max"] > 140
    
    def test_variability_features(self):
        """Test short-term and long-term variability"""
        n_samples = 2400
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        
        # High variability signal
        bpm = 140.0 + np.random.normal(0, 5, n_samples)
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        features = compute_ctg_features_window(df)
        
        assert "mean_value_of_short_term_variability" in features
        assert features["mean_value_of_short_term_variability"] > 0
        
        assert "mean_value_of_long_term_variability" in features
        assert features["mean_value_of_long_term_variability"] > 0