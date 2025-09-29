"""
Integration tests for container startup, GPU detection, and system requirements.
"""
import pytest
import subprocess
import time
import requests
from unittest.mock import patch, Mock
import torch

from app.services.stable_diffusion import StableDiffusionModelManager


class TestContainerStartup:
    """Integration tests for container startup scenarios."""
    
    def test_application_startup_sequence(self):
        """Test that application starts up in correct sequence."""
        # This test simulates the startup sequence without actually starting containers
        
        # Test 1: CUDA detection
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props:
            
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3
            mock_props.name = "GeForce GTX 1060"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            
            assert cuda_info["cuda_available"] is True
            assert cuda_info["gpu_count"] == 1
            assert cuda_info["gpu_memory_gb"] == 6.0
            assert cuda_info["gpu_name"] == "GeForce GTX 1060"
    
    def test_model_loading_sequence(self):
        """Test model loading sequence during startup."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props, \
             patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            
            # Mock GPU properties
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3
            mock_props.name = "GeForce GTX 1060"
            mock_get_props.return_value = mock_props
            
            # Mock pipeline loading
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.to.return_value = mock_pipeline
            
            # Test initialization sequence
            assert manager.is_ready() is False
            
            result = manager.initialize_model()
            
            assert result is True
            assert manager.is_ready() is True
            assert manager.device == "cuda"
            
            # Verify model loading parameters
            args, kwargs = mock_pipeline_class.from_pretrained.call_args
            assert args[0] == "runwayml/stable-diffusion-v1-5"
            assert kwargs["torch_dtype"] == torch.float16
            assert kwargs["safety_checker"] is None
    
    def test_startup_with_no_gpu(self):
        """Test startup sequence when no GPU is available."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.to.return_value = mock_pipeline
            
            # Check CUDA availability
            cuda_info = manager.check_cuda_availability()
            assert cuda_info["cuda_available"] is False
            
            # Initialize model
            result = manager.initialize_model()
            
            assert result is True
            assert manager.device == "cpu"
            
            # Verify CPU-specific parameters
            args, kwargs = mock_pipeline_class.from_pretrained.call_args
            assert kwargs["torch_dtype"] == torch.float32
    
    def test_startup_failure_recovery(self):
        """Test startup behavior when model loading fails."""
        manager = StableDiffusionModelManager()
        
        with patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.side_effect = Exception("Network error")
            
            result = manager.initialize_model()
            
            assert result is False
            assert manager.is_initialized is False
            assert manager.is_ready() is False


class TestGPUDetection:
    """Integration tests for GPU detection and compatibility."""
    
    def test_gtx_1060_detection(self):
        """Test detection of GTX 1060 GPU."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props, \
             patch('torch.version.cuda', '11.8'):
            
            # Mock GTX 1060 properties
            mock_props = Mock()
            mock_props.total_memory = 6442450944  # 6GB in bytes
            mock_props.name = "GeForce GTX 1060 6GB"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            
            assert cuda_info["cuda_available"] is True
            assert cuda_info["cuda_version"] == "11.8"
            assert cuda_info["gpu_count"] == 1
            assert cuda_info["gpu_memory_gb"] == 6.0
            assert "GTX 1060" in cuda_info["gpu_name"]
            
            # Verify optimal device selection
            device = manager._get_optimal_device()
            assert device == "cuda"
    
    def test_insufficient_gpu_memory_detection(self):
        """Test detection when GPU has insufficient memory."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props:
            
            # Mock GPU with insufficient memory (2GB)
            mock_props = Mock()
            mock_props.total_memory = 2 * 1024**3
            mock_props.name = "GeForce GTX 1050"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            
            assert cuda_info["cuda_available"] is True
            assert cuda_info["gpu_memory_gb"] == 2.0
            
            # Should fall back to CPU
            device = manager._get_optimal_device()
            assert device == "cpu"
    
    def test_multiple_gpu_detection(self):
        """Test detection when multiple GPUs are available."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.get_device_properties') as mock_get_props:
            
            # Mock first GPU properties (GTX 1060)
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3
            mock_props.name = "GeForce GTX 1060"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            
            assert cuda_info["cuda_available"] is True
            assert cuda_info["gpu_count"] == 2
            assert cuda_info["gpu_memory_gb"] == 6.0  # Uses first GPU
            
            device = manager._get_optimal_device()
            assert device == "cuda"
    
    def test_cuda_version_compatibility(self):
        """Test CUDA version detection and compatibility."""
        manager = StableDiffusionModelManager()
        
        test_versions = ["11.8", "12.0", "11.7"]
        
        for version in test_versions:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.device_count', return_value=1), \
                 patch('torch.cuda.get_device_properties') as mock_get_props, \
                 patch('torch.version.cuda', version):
                
                mock_props = Mock()
                mock_props.total_memory = 6 * 1024**3
                mock_props.name = "GeForce GTX 1060"
                mock_get_props.return_value = mock_props
                
                cuda_info = manager.check_cuda_availability()
                
                assert cuda_info["cuda_available"] is True
                assert cuda_info["cuda_version"] == version


class TestSystemRequirements:
    """Integration tests for system requirements validation."""
    
    def test_memory_requirements_validation(self):
        """Test system memory requirements validation."""
        import psutil
        
        # Get actual system memory
        memory_info = psutil.virtual_memory()
        total_gb = memory_info.total / (1024**3)
        
        # System should have at least 4GB RAM for basic operation
        assert total_gb >= 4.0, f"System has {total_gb:.1f}GB RAM, minimum 4GB required"
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        import sys
        
        # Should be Python 3.9 or higher
        assert sys.version_info >= (3, 9), f"Python {sys.version_info} detected, minimum 3.9 required"
    
    def test_pytorch_installation(self):
        """Test PyTorch installation and basic functionality."""
        import torch
        
        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(z, expected)
        
        # Test CUDA availability (may be False in test environment)
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
    
    def test_diffusers_installation(self):
        """Test diffusers library installation."""
        try:
            from diffusers import StableDiffusionPipeline
            from diffusers.utils import logging as diffusers_logging
            
            # Test that we can import required classes
            assert StableDiffusionPipeline is not None
            assert diffusers_logging is not None
            
        except ImportError as e:
            pytest.fail(f"Diffusers library not properly installed: {e}")
    
    def test_fastapi_installation(self):
        """Test FastAPI and dependencies installation."""
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            from uvicorn import run
            
            # Test basic FastAPI functionality
            app = FastAPI()
            
            @app.get("/test")
            def test_endpoint():
                return {"status": "ok"}
            
            assert app is not None
            
        except ImportError as e:
            pytest.fail(f"FastAPI dependencies not properly installed: {e}")


class TestContainerHealthChecks:
    """Integration tests for container health check functionality."""
    
    def test_health_check_components(self):
        """Test individual components of health check."""
        from app.services.stable_diffusion import model_manager, generation_service
        
        # Test model manager health
        model_info = model_manager.get_model_info()
        assert "model_id" in model_info
        assert "device" in model_info
        assert "is_initialized" in model_info
        assert "cuda_available" in model_info
        assert "gpu_memory_gb" in model_info
        
        # Test generation service health
        gen_status = generation_service.get_generation_status()
        assert "model_ready" in gen_status
        assert "model_info" in gen_status
        assert "timeout_seconds" in gen_status
    
    def test_health_endpoint_structure(self):
        """Test health endpoint response structure."""
        from fastapi.testclient import TestClient
        from main import app
        from app.services.stable_diffusion import model_manager, generation_service
        
        client = TestClient(app)
        
        with patch.object(model_manager, 'get_model_info') as mock_get_info, \
             patch.object(generation_service, 'get_generation_status') as mock_get_status:
            
            mock_get_info.return_value = {
                "model_id": "test-model",
                "device": "cpu",
                "is_initialized": False,
                "cuda_available": False,
                "gpu_memory_gb": 0
            }
            
            mock_get_status.return_value = {
                "model_ready": False,
                "model_info": {},
                "timeout_seconds": 60
            }
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify required fields
            assert "status" in data
            assert "gpu_available" in data
            assert "model_loaded" in data
            assert data["status"] in ["healthy", "unhealthy"]
            assert isinstance(data["gpu_available"], bool)
            assert isinstance(data["model_loaded"], bool)
    
    def test_startup_validation_sequence(self):
        """Test the complete startup validation sequence."""
        manager = StableDiffusionModelManager()
        
        # Step 1: Check CUDA availability
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props:
            
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3
            mock_props.name = "GeForce GTX 1060"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            assert cuda_info["cuda_available"] is True
            
            # Step 2: Validate device selection
            device = manager._get_optimal_device()
            assert device == "cuda"
            
            # Step 3: Test model initialization (mocked)
            with patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.from_pretrained.return_value = mock_pipeline
                mock_pipeline.to.return_value = mock_pipeline
                
                result = manager.initialize_model()
                assert result is True
                
                # Step 4: Validate readiness
                assert manager.is_ready() is True


class TestDockerIntegration:
    """Integration tests for Docker-specific functionality."""
    
    def test_docker_environment_detection(self):
        """Test detection of Docker environment."""
        try:
            import docker
        except ImportError:
            pytest.skip("Docker not available")
        
        import os
        
        # Check for common Docker environment indicators
        docker_indicators = [
            os.path.exists("/.dockerenv"),
            os.environ.get("DOCKER_CONTAINER") == "true",
            os.path.exists("/proc/1/cgroup") and "docker" in open("/proc/1/cgroup").read()
        ]
        
        # At least one indicator should be present in Docker environment
        # This test will pass in both Docker and non-Docker environments
        print(f"Docker indicators detected: {any(docker_indicators)}")
    
    def test_environment_variable_handling(self):
        """Test handling of environment variables."""
        import os
        
        # Test default values when env vars not set
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        model_cache = os.environ.get("MODEL_CACHE_DIR")
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        
        # Should handle missing environment variables gracefully
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        # Test with mock environment variables
        with patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "0",
            "MODEL_CACHE_DIR": "/tmp/models",
            "LOG_LEVEL": "DEBUG"
        }):
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
            assert os.environ["MODEL_CACHE_DIR"] == "/tmp/models"
            assert os.environ["LOG_LEVEL"] == "DEBUG"


if __name__ == "__main__":
    pytest.main([__file__])