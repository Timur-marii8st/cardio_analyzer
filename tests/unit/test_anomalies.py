import pytest
import numpy as np
import pandas as pd
from packages.ctg_core.anomalies import detect_anomalies
from packages.ctg_core.config import ml_settings

class TestAnomalyDetection:
    
    def test_detect_anomalies_empty_data(self):
        """Test anomaly detection with empty DataFrame"""
        df = pd.DataFrame()
        result = detect_anomalies(df)
        
        assert result["tachy_frac"] == np.nan or np.isnan(result["tachy_frac"])
        assert result["brady_frac"] == np.nan or np.isnan(result["brady_frac"])
        assert result["decel_count"] == 0
        assert result["decel_events"] == []
    
    def test_detect_tachycardia(self):
        """Test detection of tachycardia (high heart rate)"""
        # Create data with high BPM
        n_samples = 1000
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        bpm = np.full(n_samples, 170.0)  # Above tachycardia threshold (160)
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        result = detect_anomalies(df)
        
        assert result["tachy_frac"] > 0.9
        assert result["brady_frac"] < 0.1
    
    def test_detect_bradycardia(self):
        """Test detection of bradycardia (low heart rate)"""
        n_samples = 1000
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        bpm = np.full(n_samples, 100.0)  # Below bradycardia threshold (110)
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        result = detect_anomalies(df)
        
        assert result["brady_frac"] > 0.9
        assert result["tachy_frac"] < 0.1
    
    def test_detect_deceleration(self):
        """Test detection of deceleration events"""
        n_samples = 1200  # 5 minutes at 4Hz
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        bpm = np.full(n_samples, 140.0)  # Normal baseline
        
        # Create a deceleration: 30 second drop of 20 bpm
        decel_start = 600  # Start at 2.5 minutes
        decel_end = 720    # End at 3 minutes
        bpm[decel_start:decel_end] -= 20.0
        
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        result = detect_anomalies(df)
        
        assert result["decel_count"] >= 1
        assert len(result["decel_events"]) >= 1
        
        # Check deceleration properties
        decel = result["decel_events"][0]
        assert decel["dur_s"] >= 25  # Should be around 30 seconds
        assert decel["max_drop"] >= 15  # Should be around 20 bpm
        assert decel["min_bpm"] <= 125  # Should be around 120 bpm
    
    def test_low_variability_detection(self):
        """Test detection of low variability"""
        n_samples = 1000
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        
        # Create signal with very low variability
        bpm = 140.0 + np.random.normal(0, 0.1, n_samples)  # Very small variation
        ua = np.zeros(n_samples)
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        result = detect_anomalies(df)
        
        # Should detect low variability
        assert result["low_var_frac"] > 0.5
    
    def test_crosscorrelation(self):
        """Test cross-correlation between UA and BPM"""
        n_samples = 2000
        t_sec = np.arange(n_samples) / ml_settings.target_freq_hz
        
        # Create correlated signals with lag
        bpm = 140.0 + 10 * np.sin(2 * np.pi * t_sec / 60)  # 1 minute period
        lag_samples = 20  # 5 second lag at 4Hz
        ua_base = 20.0 + 15 * np.sin(2 * np.pi * t_sec / 60)
        ua = np.roll(ua_base, lag_samples)  # UA leads BPM
        
        df = pd.DataFrame({"t_sec": t_sec, "bpm": bpm, "ua": ua})
        result = detect_anomalies(df)
        
        assert "crosscorr_max" in result
        assert "crosscorr_lag_s" in result
        assert not np.isnan(result["crosscorr_max"])