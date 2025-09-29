"""
Integration tests for API endpoints.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from main import app
from app.services.stable_diffusion import model_manager, generation_service


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model_ready(self):
        """Mock model as ready for testing."""
        with patch.object(model_manager, 'is_ready', return_value=True):
            yield
    
    @pytest.fixture
    def mock_model_not_ready(self):
        """Mock model as not ready for testing."""
        with patch.object(model_manager, 'is_ready', return_value=False):
            yield
    
    @pytest.fixture
    def mock_generation_success(self):
        """Mock successful image generation."""
        async def mock_generate(**kwargs):
            return {
                "image": Mock(),  # Mock PIL Image
                "metadata": {
                    "prompt": kwargs.get("prompt", "test prompt"),
                    "steps": kwargs.get("steps", 20),
                    "width": kwargs.get("width", 512),
                    "height": kwargs.get("height", 512),
                    "seed": kwargs.get("seed", 12345),
                    "generation_time_seconds": 15.2,
                    "model_version": "runwayml/stable-diffusion-v1-5"
                }
            }
        
        with patch.object(generation_service, 'generate_image', new_callable=AsyncMock, side_effect=mock_generate):
            yield
    
    def test_health_endpoint_healthy(self, client, mock_model_ready):
        """Test health endpoint when service is healthy."""
        with patch.object(model_manager, 'get_model_info') as mock_get_info, \
             patch.object(generation_service, 'get_generation_status') as mock_get_status:
            
            mock_get_info.return_value = {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "cuda",
                "is_initialized": True,
                "cuda_available": True,
                "gpu_memory_gb": 6.0
            }
            
            mock_get_status.return_value = {
                "model_ready": True,
                "timeout_seconds": 60
            }
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["gpu_available"] is True
            assert data["model_loaded"] is True
            assert "memory_usage" in data
    
    def test_health_endpoint_unhealthy(self, client, mock_model_not_ready):
        """Test health endpoint when service is unhealthy."""
        with patch.object(model_manager, 'get_model_info') as mock_get_info, \
             patch.object(generation_service, 'get_generation_status') as mock_get_status:
            
            mock_get_info.return_value = {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "cpu",
                "is_initialized": False,
                "cuda_available": False,
                "gpu_memory_gb": 0
            }
            
            mock_get_status.return_value = {
                "model_ready": False,
                "timeout_seconds": 60
            }
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert data["gpu_available"] is False
            assert data["model_loaded"] is False
    
    def test_generate_endpoint_success(self, client, mock_model_ready, mock_generation_success):
        """Test successful image generation."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            request_data = {
                "prompt": "A beautiful sunset over mountains",
                "steps": 20,
                "width": 512,
                "height": 512,
                "seed": 12345
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "image_base64" in data
            assert "metadata" in data
            assert data["metadata"]["prompt"] == request_data["prompt"]
            assert data["metadata"]["steps"] == request_data["steps"]
            assert data["metadata"]["width"] == request_data["width"]
            assert data["metadata"]["height"] == request_data["height"]
            assert data["metadata"]["seed"] == request_data["seed"]
    
    def test_generate_endpoint_model_not_ready(self, client, mock_model_not_ready):
        """Test image generation when model is not ready."""
        request_data = {
            "prompt": "A beautiful sunset over mountains",
            "steps": 20,
            "width": 512,
            "height": 512
        }
        
        response = client.post("/generate", json=request_data)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        
        assert "detail" in data
        assert data["detail"]["error_code"] == "MODEL_NOT_READY"
    
    def test_generate_endpoint_validation_error(self, client, mock_model_ready):
        """Test image generation with invalid parameters."""
        request_data = {
            "prompt": "",  # Empty prompt should fail validation
            "steps": 20,
            "width": 512,
            "height": 512
        }
        
        response = client.post("/generate", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_generate_endpoint_invalid_dimensions(self, client, mock_model_ready):
        """Test image generation with invalid dimensions."""
        request_data = {
            "prompt": "A beautiful sunset",
            "steps": 20,
            "width": 513,  # Not multiple of 64
            "height": 512
        }
        
        response = client.post("/generate", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_generate_endpoint_invalid_steps(self, client, mock_model_ready):
        """Test image generation with invalid steps."""
        request_data = {
            "prompt": "A beautiful sunset",
            "steps": 100,  # Too high
            "width": 512,
            "height": 512
        }
        
        response = client.post("/generate", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_generate_endpoint_timeout_error(self, client, mock_model_ready):
        """Test image generation timeout handling."""
        with patch.object(generation_service, 'generate_image', new_callable=AsyncMock, side_effect=TimeoutError("Generation timed out")):
            request_data = {
                "prompt": "A beautiful sunset over mountains",
                "steps": 20,
                "width": 512,
                "height": 512
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error_code"] == "TIMEOUT_ERROR"
    
    def test_generate_endpoint_memory_error(self, client, mock_model_ready):
        """Test image generation memory error handling."""
        with patch.object(generation_service, 'generate_image', new_callable=AsyncMock, side_effect=RuntimeError("CUDA out of memory")):
            request_data = {
                "prompt": "A beautiful sunset over mountains",
                "steps": 20,
                "width": 512,
                "height": 512
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error_code"] == "MEMORY_ERROR"
    
    def test_generate_endpoint_runtime_error(self, client, mock_model_ready):
        """Test image generation runtime error handling."""
        with patch.object(generation_service, 'generate_image', new_callable=AsyncMock, side_effect=RuntimeError("Generation failed")):
            request_data = {
                "prompt": "A beautiful sunset over mountains",
                "steps": 20,
                "width": 512,
                "height": 512
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error_code"] == "GENERATION_ERROR"
    
    def test_generate_endpoint_default_values(self, client, mock_model_ready, mock_generation_success):
        """Test image generation with default parameter values."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            request_data = {
                "prompt": "A beautiful sunset over mountains"
                # No steps, width, height, or seed - should use defaults
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["metadata"]["steps"] == 20  # Default steps
            assert data["metadata"]["width"] == 512  # Default width
            assert data["metadata"]["height"] == 512  # Default height
            assert "seed" in data["metadata"]  # Should have generated seed
    
    def test_generate_endpoint_with_seed(self, client, mock_model_ready, mock_generation_success):
        """Test image generation with specific seed."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            request_data = {
                "prompt": "A beautiful sunset over mountains",
                "seed": 42
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["metadata"]["seed"] == 42
    
    def test_health_endpoint_headers(self, client, mock_model_ready):
        """Test that health endpoint returns proper headers."""
        with patch.object(model_manager, 'get_model_info') as mock_get_info, \
             patch.object(generation_service, 'get_generation_status') as mock_get_status:
            
            mock_get_info.return_value = {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "cuda",
                "is_initialized": True,
                "cuda_available": True,
                "gpu_memory_gb": 6.0
            }
            
            mock_get_status.return_value = {
                "model_ready": True,
                "timeout_seconds": 60
            }
            
            response = client.get("/health")
            
            assert response.headers["content-type"] == "application/json"
            assert "cache-control" in response.headers
    
    def test_generate_endpoint_headers(self, client, mock_model_ready, mock_generation_success):
        """Test that generate endpoint returns proper headers."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            request_data = {
                "prompt": "A beautiful sunset over mountains"
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "application/json"
            assert "cache-control" in response.headers


if __name__ == "__main__":
    pytest.main([__file__])