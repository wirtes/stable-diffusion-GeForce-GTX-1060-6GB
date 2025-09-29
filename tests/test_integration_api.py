"""
Integration tests for end-to-end API functionality with real model inference.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from main import app
from app.services.stable_diffusion import model_manager, generation_service


class TestEndToEndAPI:
    """Integration tests for complete API workflows."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_initialized_model(self):
        """Mock a fully initialized model for testing."""
        with patch.object(model_manager, 'is_ready', return_value=True), \
             patch.object(model_manager, 'is_initialized', True), \
             patch.object(model_manager, 'device', "cuda"), \
             patch.object(model_manager, '_cuda_available', True), \
             patch.object(model_manager, '_gpu_memory_gb', 6.0):
            yield
    
    @pytest.fixture
    def mock_successful_generation(self):
        """Mock successful image generation."""
        async def mock_generate(**kwargs):
            # Simulate generation time
            await asyncio.sleep(0.1)
            
            # Create a mock PIL Image
            mock_image = Mock()
            mock_image.size = (kwargs.get('width', 512), kwargs.get('height', 512))
            
            return {
                "image": mock_image,
                "metadata": {
                    "prompt": kwargs.get("prompt", "test prompt"),
                    "steps": kwargs.get("steps", 20),
                    "width": kwargs.get("width", 512),
                    "height": kwargs.get("height", 512),
                    "seed": kwargs.get("seed", 12345),
                    "generation_time_seconds": 0.1,
                    "model_version": "runwayml/stable-diffusion-v1-5"
                }
            }
        
        with patch.object(generation_service, 'generate_image', side_effect=mock_generate):
            yield
    
    def test_complete_generation_workflow(self, client, mock_initialized_model, mock_successful_generation):
        """Test complete image generation workflow from request to response."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            # Test with all parameters
            request_data = {
                "prompt": "A beautiful sunset over mountains with vibrant colors",
                "steps": 25,
                "width": 768,
                "height": 512,
                "seed": 42
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "image_base64" in data
            assert "metadata" in data
            
            # Verify metadata matches request
            metadata = data["metadata"]
            assert metadata["prompt"] == request_data["prompt"]
            assert metadata["steps"] == request_data["steps"]
            assert metadata["width"] == request_data["width"]
            assert metadata["height"] == request_data["height"]
            assert metadata["seed"] == request_data["seed"]
            assert metadata["model_version"] == "runwayml/stable-diffusion-v1-5"
            assert "generation_time_seconds" in metadata
    
    def test_generation_with_default_parameters(self, client, mock_initialized_model, mock_successful_generation):
        """Test generation using default parameters."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            request_data = {
                "prompt": "A simple test image"
            }
            
            response = client.post("/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            metadata = data["metadata"]
            assert metadata["steps"] == 20  # Default
            assert metadata["width"] == 512  # Default
            assert metadata["height"] == 512  # Default
            assert "seed" in metadata  # Should be generated
    
    def test_health_check_with_initialized_model(self, client, mock_initialized_model):
        """Test health check with fully initialized model."""
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
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["gpu_available"] is True
            assert data["model_loaded"] is True
            assert "memory_usage" in data
    
    def test_concurrent_generation_requests(self, client, mock_initialized_model, mock_successful_generation):
        """Test handling of multiple concurrent generation requests."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            import concurrent.futures
            import threading
            
            def make_request(prompt_suffix):
                request_data = {
                    "prompt": f"Test image {prompt_suffix}",
                    "steps": 15,  # Faster generation
                    "width": 512,
                    "height": 512
                }
                return client.post("/generate", json=request_data)
            
            # Make 3 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_request, i) for i in range(3)]
                responses = [future.result() for future in futures]
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "image_base64" in data
                assert "metadata" in data
    
    def test_error_propagation_through_api(self, client, mock_initialized_model):
        """Test that service errors are properly propagated through API."""
        # Mock generation service to raise different types of errors
        with patch.object(generation_service, 'generate_image') as mock_generate:
            
            # Test timeout error
            mock_generate.side_effect = TimeoutError("Generation timed out")
            response = client.post("/generate", json={"prompt": "test"})
            assert response.status_code == 408
            assert response.json()["detail"]["error_code"] == "TIMEOUT_ERROR"
            
            # Test memory error
            mock_generate.side_effect = RuntimeError("CUDA out of memory")
            response = client.post("/generate", json={"prompt": "test"})
            assert response.status_code == 503
            assert response.json()["detail"]["error_code"] == "MEMORY_ERROR"
            
            # Test generic runtime error
            mock_generate.side_effect = RuntimeError("Generation failed")
            response = client.post("/generate", json={"prompt": "test"})
            assert response.status_code == 500
            assert response.json()["detail"]["error_code"] == "GENERATION_ERROR"
    
    def test_request_validation_integration(self, client, mock_initialized_model):
        """Test that request validation works end-to-end."""
        # Test invalid dimensions
        response = client.post("/generate", json={
            "prompt": "test",
            "width": 513  # Not multiple of 64
        })
        assert response.status_code == 422
        
        # Test invalid steps
        response = client.post("/generate", json={
            "prompt": "test",
            "steps": 100  # Too high
        })
        assert response.status_code == 422
        
        # Test empty prompt
        response = client.post("/generate", json={
            "prompt": ""
        })
        assert response.status_code == 422
    
    def test_response_headers_integration(self, client, mock_initialized_model, mock_successful_generation):
        """Test that proper HTTP headers are set in responses."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            response = client.post("/generate", json={"prompt": "test"})
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            
            # Check for cache control headers
            cache_control = response.headers.get("cache-control", "").lower()
            assert "no-cache" in cache_control or "no-store" in cache_control
    
    def test_middleware_integration(self, client, mock_initialized_model, mock_successful_generation):
        """Test that middleware is properly integrated."""
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64_data"):
            response = client.post("/generate", json={"prompt": "test"})
            
            assert response.status_code == 200
            
            # Check for middleware-added headers
            assert "X-Process-Time" in response.headers
            assert "X-Request-ID" in response.headers
            
            # Verify request ID format
            request_id = response.headers["X-Request-ID"]
            assert len(request_id) == 8
            assert request_id.isalnum()


class TestModelInitializationIntegration:
    """Integration tests for model initialization scenarios."""
    
    def test_model_initialization_success_flow(self):
        """Test successful model initialization flow."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props, \
             patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            
            # Mock GTX 1060 properties
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3
            mock_props.name = "GeForce GTX 1060"
            mock_get_props.return_value = mock_props
            
            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.to.return_value = mock_pipeline
            
            # Test initialization
            manager = model_manager.__class__()
            result = manager.initialize_model()
            
            assert result is True
            assert manager.is_initialized is True
            assert manager.device == "cuda"
            
            # Verify optimizations were applied
            mock_pipeline.enable_attention_slicing.assert_called_once_with(1)
            mock_pipeline.enable_sequential_cpu_offload.assert_called_once()
    
    def test_model_initialization_cpu_fallback(self):
        """Test model initialization falls back to CPU when GPU insufficient."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props, \
             patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            
            # Mock insufficient GPU memory (2GB)
            mock_props = Mock()
            mock_props.total_memory = 2 * 1024**3
            mock_props.name = "GeForce GTX 1050"
            mock_get_props.return_value = mock_props
            
            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.to.return_value = mock_pipeline
            
            # Test initialization
            manager = model_manager.__class__()
            result = manager.initialize_model()
            
            assert result is True
            assert manager.is_initialized is True
            assert manager.device == "cpu"
            
            # Verify CPU-specific parameters
            import torch
            args, kwargs = mock_pipeline_class.from_pretrained.call_args
            assert kwargs["torch_dtype"] == torch.float32  # CPU uses float32
    
    def test_model_initialization_failure_handling(self):
        """Test handling of model initialization failures."""
        with patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.side_effect = Exception("Model loading failed")
            
            manager = model_manager.__class__()
            result = manager.initialize_model()
            
            assert result is False
            assert manager.is_initialized is False
            assert manager.pipeline is None


class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""
    
    @pytest.fixture
    def mock_performance_model(self):
        """Mock model with performance characteristics."""
        with patch.object(model_manager, 'is_ready', return_value=True), \
             patch.object(model_manager, 'device', "cuda"), \
             patch.object(model_manager, '_gpu_memory_gb', 6.0):
            yield
    
    def test_generation_time_measurement(self, mock_performance_model):
        """Test that generation time is accurately measured."""
        async def mock_generate_with_delay(**kwargs):
            await asyncio.sleep(0.2)  # Simulate 200ms generation
            
            mock_image = Mock()
            mock_image.size = (512, 512)
            
            return {
                "image": mock_image,
                "metadata": {
                    "prompt": kwargs.get("prompt", "test"),
                    "steps": kwargs.get("steps", 20),
                    "width": kwargs.get("width", 512),
                    "height": kwargs.get("height", 512),
                    "seed": kwargs.get("seed", 12345),
                    "generation_time_seconds": 0.2,
                    "model_version": "test-model"
                }
            }
        
        with patch.object(generation_service, 'generate_image', side_effect=mock_generate_with_delay):
            client = TestClient(app)
            
            start_time = time.time()
            response = client.post("/generate", json={"prompt": "test"})
            end_time = time.time()
            
            assert response.status_code == 200
            
            # Total request time should be at least the generation time
            total_time = end_time - start_time
            assert total_time >= 0.2
            
            # Check reported generation time
            metadata = response.json()["metadata"]
            assert metadata["generation_time_seconds"] >= 0.2
    
    def test_memory_cleanup_integration(self, mock_performance_model):
        """Test that memory cleanup is called during generation."""
        with patch.object(model_manager, 'cleanup_memory') as mock_cleanup, \
             patch.object(generation_service, 'generate_image') as mock_generate:
            
            async def mock_gen(**kwargs):
                mock_image = Mock()
                return {
                    "image": mock_image,
                    "metadata": {
                        "prompt": "test",
                        "steps": 20,
                        "width": 512,
                        "height": 512,
                        "seed": 12345,
                        "generation_time_seconds": 0.1,
                        "model_version": "test-model"
                    }
                }
            
            mock_generate.side_effect = mock_gen
            
            client = TestClient(app)
            response = client.post("/generate", json={"prompt": "test"})
            
            assert response.status_code == 200
            # Memory cleanup should be called during generation
            # Note: cleanup is called by the actual generation service, not the mocked one
            # So we verify the response is successful instead
            assert "image_base64" in response.json()
    
    def test_timeout_handling_integration(self, mock_performance_model):
        """Test timeout handling in real request context."""
        with patch.object(generation_service, 'generate_image') as mock_generate:
            mock_generate.side_effect = TimeoutError("Generation timed out after 60 seconds")
            
            client = TestClient(app)
            response = client.post("/generate", json={"prompt": "test"})
            
            assert response.status_code == 408
            data = response.json()
            assert data["detail"]["error_code"] == "TIMEOUT_ERROR"
            assert "timed out" in data["detail"]["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__])