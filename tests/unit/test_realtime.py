import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from packages.ctg_core.realtime_redis import RealtimeProcessorRedis

class TestRealtimeProcessor:
    
    @pytest.fixture
    def mock_redis_manager(self):
        """Mock Redis session manager"""
        with patch('packages.ctg_core.realtime_redis.RedisSessionManager') as mock:
            yield mock
    
    @pytest.fixture
    def mock_predict_fn(self):
        """Mock prediction function"""
        return Mock(return_value=0.15)  # Normal risk
    
    def test_ingest_samples(self, mock_redis_manager, mock_predict_fn):
        """Test sample ingestion"""
        processor = RealtimeProcessorRedis(mock_predict_fn)
        
        samples = [
            {"ts": "2024-01-01T00:00:00Z", "channel": "bpm", "value": 140},
            {"ts": "2024-01-01T00:00:01Z", "channel": "bpm", "value": 141},
            {"ts": "2024-01-01T00:00:00Z", "channel": "uterus", "value": 20},
        ]
        
        processor.ingest_samples("test_session", samples)
        
        # Verify Redis manager methods were called
        assert processor.session_manager.add_samples.call_count == 2  # Once for bpm, once for ua
    
    def test_step_insufficient_data(self, mock_redis_manager, mock_predict_fn):
        """Test step with insufficient data"""
        processor = RealtimeProcessorRedis(mock_predict_fn)
        
        # Mock empty buffers
        processor.session_manager.get_session_buffers_as_dataframes.return_value = (
            pd.DataFrame(),
            pd.DataFrame()
        )
        
        result = processor.step("test_session")
        assert result is None
    
    def test_step_with_data(self, mock_redis_manager, mock_predict_fn):
        """Test step with sufficient data"""
        processor = RealtimeProcessorRedis(mock_predict_fn)
        
        # Create mock data
        n_samples = 2400  # 10 minutes at 4Hz
        ts = pd.date_range(start='2024-01-01', periods=n_samples, freq='250ms')
        bpm_df = pd.DataFrame({
            "ts": ts,
            "value": np.random.normal(140, 5, n_samples)
        })
        ua_df = pd.DataFrame({
            "ts": ts,
            "value": np.random.normal(20, 3, n_samples)
        })
        
        processor.session_manager.get_session_buffers_as_dataframes.return_value = (
            bpm_df,
            ua_df
        )
        
        result = processor.step("test_session")
        
        assert result is not None
        assert "risk" in result
        assert "features" in result
        assert "series" in result
        assert result["risk"]["hypoxia_prob"] == 0.15
        assert result["risk"]["band"] == "normal"
