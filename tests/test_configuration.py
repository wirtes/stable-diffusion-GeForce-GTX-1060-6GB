"""
Tests for configuration management
"""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from config.settings import (
    Settings, APISettings, ModelSettings, GPUSettings, 
    LoggingSettings, ResourceSettings
)
from app.core.config_validator import ConfigurationValidator, validate_configuration


class TestAPISettings:
    """Test API configuration settings"""
    
    def test_default_values(self):
        """Test default API settings"""
        api_settings = APISettings()
        assert api_settings.host == "0.0.0.0"
        assert api_settings.port == 8000
        assert api_settings.allowed_hosts == ["*"]
        assert api_settings.max_concurrent_requests == 1
        assert api_settings.generation_timeout_seconds == 60
    
    def test_environment_variable_override(self):
        """Test environment variable override"""
        with patch.dict(os.environ, {
            'API_HOST': '127.0.0.1',
            'API_PORT': '9000',
            'ALLOWED_HOSTS': 'localhost,127.0.0.1',
            'MAX_CONCURRENT_REQUESTS': '3'
        }):
            api_settings = APISettings()
            assert api_settings.host == "127.0.0.1"
            assert api_settings.port == 9000
            assert api_settings.allowed_hosts == ["localhost", "127.0.0.1"]
            assert api_settings.max_concurrent_requests == 3
    
    def test_port_validation(self):
        """Test port validation"""
        with pytest.raises(ValueError):
            APISettings(port=0)
        
        with pytest.raises(ValueError):
            APISettings(port=70000)
    
    def test_allowed_hosts_parsing(self):
        """Test allowed hosts parsing"""
        api_settings = APISettings(allowed_hosts="host1, host2 , host3")
        assert api_settings.allowed_hosts == ["host1", "host2", "host3"]


class TestModelSettings:
    """Test model configuration settings"""
    
    def test_default_values(self):
        """Test default model settings"""
        model_settings = ModelSettings()
        assert model_settings.name == "runwayml/stable-diffusion-v1-5"
        assert model_settings.torch_dtype == "float16"
        assert model_settings.enable_attention_slicing is True
        assert model_settings.enable_cpu_offload is True
    
    def test_cache_dir_creation(self):
        """Test cache directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            model_settings = ModelSettings(cache_dir=str(cache_dir))
            assert cache_dir.exists()
            assert cache_dir.is_dir()
    
    def test_torch_dtype_validation(self):
        """Test torch dtype validation"""
        with pytest.raises(ValueError):
            ModelSettings(torch_dtype="invalid_dtype")
        
        # Valid dtypes should work
        for dtype in ['float16', 'float32', 'bfloat16']:
            settings = ModelSettings(torch_dtype=dtype)
            assert settings.torch_dtype == dtype
    
    def test_torch_dtype_property(self):
        """Test torch dtype property conversion"""
        settings = ModelSettings(torch_dtype="float16")
        assert settings.torch_dtype_obj == torch.float16
        
        settings = ModelSettings(torch_dtype="float32")
        assert settings.torch_dtype_obj == torch.float32


class TestGPUSettings:
    """Test GPU configuration settings"""
    
    def test_default_values(self):
        """Test default GPU settings"""
        gpu_settings = GPUSettings()
        assert gpu_settings.cuda_visible_devices == "0"
        assert gpu_settings.memory_fraction == 0.9
        assert gpu_settings.allow_growth is True
        assert gpu_settings.force_cpu is False
    
    def test_cuda_devices_validation(self):
        """Test CUDA devices validation"""
        # Valid device IDs
        gpu_settings = GPUSettings(cuda_visible_devices="0,1,2")
        assert gpu_settings.cuda_visible_devices == "0,1,2"
        
        # Invalid device IDs should raise error
        with pytest.raises(ValueError):
            GPUSettings(cuda_visible_devices="invalid")
    
    def test_memory_fraction_validation(self):
        """Test memory fraction validation"""
        with pytest.raises(ValueError):
            GPUSettings(memory_fraction=0.0)
        
        with pytest.raises(ValueError):
            GPUSettings(memory_fraction=1.5)


class TestLoggingSettings:
    """Test logging configuration settings"""
    
    def test_default_values(self):
        """Test default logging settings"""
        logging_settings = LoggingSettings()
        assert logging_settings.level == "INFO"
        assert "%(asctime)s" in logging_settings.format
        assert logging_settings.file_path is None
    
    def test_log_level_validation(self):
        """Test log level validation"""
        with pytest.raises(ValueError):
            LoggingSettings(level="INVALID")
        
        # Valid levels should work
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            settings = LoggingSettings(level=level)
            assert settings.level == level


class TestResourceSettings:
    """Test resource configuration settings"""
    
    def test_default_values(self):
        """Test default resource settings"""
        resource_settings = ResourceSettings()
        assert resource_settings.memory_limit == "8G"
        assert resource_settings.cpu_limit == 4.0
        assert resource_settings.worker_processes == 1
    
    def test_memory_limit_validation(self):
        """Test memory limit validation"""
        # Valid formats
        for limit in ['8G', '4096M', '1024']:
            settings = ResourceSettings(memory_limit=limit)
            assert settings.memory_limit == limit.upper()
        
        # Invalid format should raise error
        with pytest.raises(ValueError):
            ResourceSettings(memory_limit="invalid")
    
    def test_cpu_limit_validation(self):
        """Test CPU limit validation"""
        with pytest.raises(ValueError):
            ResourceSettings(cpu_limit=0.0)
        
        with pytest.raises(ValueError):
            ResourceSettings(cpu_limit=50.0)


class TestSettings:
    """Test main settings class"""
    
    def test_default_initialization(self):
        """Test default settings initialization"""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.gpu, GPUSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.resources, ResourceSettings)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_unavailable_fallback(self, mock_cuda):
        """Test fallback when CUDA is unavailable"""
        settings = Settings()
        # Should automatically set force_cpu when CUDA unavailable
        assert settings.gpu.force_cpu is True
    
    def test_get_model_config(self):
        """Test model configuration generation"""
        settings = Settings()
        config = settings.get_model_config()
        
        assert 'torch_dtype' in config
        assert 'cache_dir' in config
        assert 'safety_checker' in config
        assert 'requires_safety_checker' in config
    
    def test_get_pipeline_config(self):
        """Test pipeline configuration generation"""
        settings = Settings()
        config = settings.get_pipeline_config()
        
        assert 'enable_attention_slicing' in config
        assert 'enable_cpu_offload' in config
        assert 'enable_model_cpu_offload' in config
        assert 'enable_vae_slicing' in config
    
    def test_validate_startup_requirements(self):
        """Test startup requirements validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.model.cache_dir = temp_dir
            
            errors = settings.validate_startup_requirements()
            # Should have minimal errors with valid temp directory
            assert isinstance(errors, list)


class TestConfigurationValidator:
    """Test configuration validator"""
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        settings = Settings()
        validator = ConfigurationValidator(settings)
        
        assert validator.settings == settings
        assert validator.errors == []
        assert validator.warnings == []
    
    @patch('socket.socket')
    def test_api_validation_port_available(self, mock_socket):
        """Test API validation with available port"""
        mock_socket_instance = MagicMock()
        mock_socket_instance.connect_ex.return_value = 1  # Port not in use
        mock_socket.return_value.__enter__.return_value = mock_socket_instance
        
        settings = Settings()
        validator = ConfigurationValidator(settings)
        validator._validate_api_settings()
        
        assert len(validator.errors) == 0
    
    @patch('socket.socket')
    def test_api_validation_port_in_use(self, mock_socket):
        """Test API validation with port in use"""
        mock_socket_instance = MagicMock()
        mock_socket_instance.connect_ex.return_value = 0  # Port in use
        mock_socket.return_value.__enter__.return_value = mock_socket_instance
        
        settings = Settings()
        validator = ConfigurationValidator(settings)
        validator._validate_api_settings()
        
        assert len(validator.errors) > 0
        assert "already in use" in validator.errors[0]
    
    def test_model_validation_invalid_cache_dir(self):
        """Test model validation with invalid cache directory"""
        settings = Settings()
        settings.model.cache_dir = "/invalid/path/that/cannot/be/created"
        
        validator = ConfigurationValidator(settings)
        validator._validate_model_settings()
        
        # Should have error about cache directory
        assert len(validator.errors) > 0
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_gpu_validation_cuda_unavailable(self, mock_cuda):
        """Test GPU validation when CUDA unavailable"""
        settings = Settings()
        settings.gpu.force_cpu = False  # Request GPU mode
        
        validator = ConfigurationValidator(settings)
        validator._validate_gpu_settings()
        
        assert len(validator.errors) > 0
        assert "CUDA is not available" in validator.errors[0]
    
    def test_validation_summary(self):
        """Test validation summary generation"""
        settings = Settings()
        validator = ConfigurationValidator(settings)
        
        # Add some test errors and warnings
        validator.errors.append("Test error")
        validator.warnings.append("Test warning")
        
        summary = validator.get_validation_summary()
        
        assert summary['valid'] is False
        assert summary['error_count'] == 1
        assert summary['warning_count'] == 1
        assert "Test error" in summary['errors']
        assert "Test warning" in summary['warnings']


class TestConfigurationFunctions:
    """Test configuration utility functions"""
    
    @patch('app.core.config_validator.ConfigurationValidator')
    def test_validate_configuration_function(self, mock_validator_class):
        """Test validate_configuration function"""
        mock_validator = MagicMock()
        mock_validator.validate_all.return_value = True
        mock_validator_class.return_value = mock_validator
        
        result = validate_configuration()
        
        assert result is True
        mock_validator_class.assert_called_once()
        mock_validator.validate_all.assert_called_once()
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables"""
        env_vars = {
            'ENVIRONMENT': 'production',
            'DEBUG': 'true',
            'API_PORT': '9000',
            'MODEL_NAME': 'custom/model',
            'LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.environment == 'production'
            assert settings.debug is True
            assert settings.api.port == 9000
            assert settings.model.name == 'custom/model'
            assert settings.logging.level == 'DEBUG'


if __name__ == "__main__":
    pytest.main([__file__])