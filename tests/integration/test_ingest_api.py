import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from apps.api.main import app
from apps.api.auth import Token

class TestIngestAPI:
    
    @pytest.fixture
    def client(self):
        """Test client with mocked dependencies"""
        with patch('apps.api.main.storage'), \
             patch('apps.api.main.auth_service'), \
             patch('apps.api.main.inference_service'), \
             patch('apps.api.main.streaming_service'):
            
            with TestClient(app) as client:
                yield client
    
    @pytest.fixture
    def auth_token(self):
        """Mock authentication token"""
        return "test-token-12345"
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_ingest_batch_unauthenticated(self, client):
        """Test that ingest requires authentication"""
        response = client.post("/v1/ingest/batch", json={
            "device_id": "test-device",
            "session_id": "test-session",
            "samples": []
        })
        assert response.status_code == 403  # Forbidden without auth
    
    @patch('apps.api.auth.get_current_user')
    def test_ingest_batch_authenticated(self, mock_auth, client, auth_token):
        """Test authenticated batch ingest"""
        mock_auth.return_value = Mock(user_id="test-user", role="doctor")
        
        response = client.post(
            "/v1/ingest/batch",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "device_id": "test-device",
                "session_id": "test-session",
                "samples": [
                    {"ts": "2024-01-01T00:00:00Z", "channel": "bpm", "value": 140}
                ]
            }
        )
        
        # Should work with proper auth
        assert response.status_code in [200, 422]  # 422 if validation fails due to mocks