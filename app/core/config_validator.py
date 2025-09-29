"""
Configuration validation utilities
"""
import os
import socket
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from config.settings import Settings


logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validates application configuration and system requirements"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """
        Validate all configuration aspects
        
        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Run all validation checks
        self._validate_environment()
        self._validate_api_settings()
        self._validate_model_settings()
        self._validate_gpu_settings()
        self._validate_resource_settings()
        self._validate_logging_settings()
        self._validate_system_requirements()
        
        # Log results
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        if self.errors:
            for error in self.errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def _validate_environment(self) -> None:
        """Validate environment settings"""
        valid_environments = ['development', 'staging', 'production']
        if self.settings.environment not in valid_environments:
            self.warnings.append(
                f"Unknown environment '{self.settings.environment}'. "
                f"Expected one of: {valid_environments}"
            )
    
    def _validate_api_settings(self) -> None:
        """Validate API configuration"""
        # Check port availability
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((self.settings.api.host, self.settings.api.port))
                if result == 0:
                    self.errors.append(
                        f"Port {self.settings.api.port} is already in use on {self.settings.api.host}"
                    )
        except Exception as e:
            self.warnings.append(f"Could not check port availability: {e}")
        
        # Validate host format
        if self.settings.api.host not in ['0.0.0.0', 'localhost', '127.0.0.1']:
            try:
                socket.inet_aton(self.settings.api.host)
            except socket.error:
                self.errors.append(f"Invalid host address: {self.settings.api.host}")
        
        # Validate concurrent requests limit
        if self.settings.api.max_concurrent_requests > 5:
            self.warnings.append(
                f"High concurrent request limit ({self.settings.api.max_concurrent_requests}) "
                "may cause memory issues on GTX 1060"
            )
    
    def _validate_model_settings(self) -> None:
        """Validate model configuration"""
        # Check cache directory
        try:
            cache_path = Path(self.settings.model.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            if not cache_path.is_dir():
                self.errors.append(f"Model cache directory is not accessible: {cache_path}")
            elif not os.access(cache_path, os.W_OK):
                self.errors.append(f"Model cache directory is not writable: {cache_path}")
            else:
                # Check available space (at least 10GB recommended)
                try:
                    stat = os.statvfs(cache_path)
                    free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    if free_space_gb < 10:
                        self.warnings.append(
                            f"Low disk space in model cache directory: {free_space_gb:.1f}GB available. "
                            "At least 10GB recommended."
                        )
                except Exception:
                    pass  # statvfs not available on all systems
                    
        except Exception as e:
            self.errors.append(f"Cannot access model cache directory: {e}")
        
        # Validate model name format
        if not self.settings.model.name or '/' not in self.settings.model.name:
            self.errors.append(
                f"Invalid model name format: {self.settings.model.name}. "
                "Expected format: 'organization/model-name'"
            )
        
        # Check torch dtype compatibility
        if not self.settings.gpu.force_cpu and self.settings.model.torch_dtype == 'float32':
            self.warnings.append(
                "Using float32 with GPU may cause memory issues. Consider using float16."
            )
    
    def _validate_gpu_settings(self) -> None:
        """Validate GPU configuration"""
        if not self.settings.gpu.force_cpu:
            # Check CUDA availability
            if not torch.cuda.is_available():
                self.errors.append(
                    "CUDA is not available but GPU mode is requested. "
                    "Set FORCE_CPU=true to use CPU mode."
                )
                return
            
            # Validate GPU devices
            try:
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    self.errors.append("No CUDA devices found")
                    return
                
                # Parse and validate device IDs
                device_ids = []
                if self.settings.gpu.cuda_visible_devices != "-1":
                    for device_str in self.settings.gpu.cuda_visible_devices.split(','):
                        try:
                            device_id = int(device_str.strip())
                            if device_id >= device_count:
                                self.errors.append(
                                    f"GPU device {device_id} not available. "
                                    f"Only {device_count} devices found."
                                )
                            else:
                                device_ids.append(device_id)
                        except ValueError:
                            self.errors.append(f"Invalid GPU device ID: {device_str}")
                
                # Check GPU memory for each device
                for device_id in device_ids:
                    try:
                        props = torch.cuda.get_device_properties(device_id)
                        memory_gb = props.total_memory / (1024**3)
                        
                        if memory_gb < 4:
                            self.warnings.append(
                                f"GPU {device_id} has only {memory_gb:.1f}GB memory. "
                                "May not be sufficient for stable diffusion."
                            )
                        elif memory_gb < 6:
                            self.warnings.append(
                                f"GPU {device_id} has {memory_gb:.1f}GB memory. "
                                "Consider enabling CPU offloading for better performance."
                            )
                        
                        logger.info(f"GPU {device_id}: {props.name} ({memory_gb:.1f}GB)")
                        
                    except Exception as e:
                        self.warnings.append(f"Could not check GPU {device_id} properties: {e}")
                        
            except Exception as e:
                self.errors.append(f"Error validating GPU configuration: {e}")
        
        # Validate memory fraction
        if self.settings.gpu.memory_fraction > 0.95:
            self.warnings.append(
                f"High GPU memory fraction ({self.settings.gpu.memory_fraction}) "
                "may cause out-of-memory errors"
            )
    
    def _validate_resource_settings(self) -> None:
        """Validate resource limits"""
        # Parse memory limit
        memory_limit = self.settings.resources.memory_limit
        try:
            if memory_limit.endswith('G'):
                memory_gb = int(memory_limit[:-1])
            elif memory_limit.endswith('M'):
                memory_gb = int(memory_limit[:-1]) / 1024
            else:
                memory_gb = int(memory_limit) / (1024**3)
            
            if memory_gb < 4:
                self.warnings.append(
                    f"Low memory limit ({memory_gb:.1f}GB). "
                    "At least 8GB recommended for stable diffusion."
                )
        except ValueError:
            self.errors.append(f"Invalid memory limit format: {memory_limit}")
        
        # Validate CPU limit
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            if self.settings.resources.cpu_limit > cpu_count:
                self.warnings.append(
                    f"CPU limit ({self.settings.resources.cpu_limit}) exceeds "
                    f"available CPUs ({cpu_count})"
                )
        except ImportError:
            pass  # psutil not available
    
    def _validate_logging_settings(self) -> None:
        """Validate logging configuration"""
        # Check log file path if specified
        if self.settings.logging.file_path:
            try:
                log_path = Path(self.settings.logging.file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = log_path.parent / '.write_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except Exception:
                    self.errors.append(f"Log directory is not writable: {log_path.parent}")
                    
            except Exception as e:
                self.errors.append(f"Invalid log file path: {e}")
        
        # Validate log file size
        if self.settings.logging.max_file_size < 1024 * 1024:  # 1MB
            self.warnings.append("Very small log file size may cause frequent rotation")
    
    def _validate_system_requirements(self) -> None:
        """Validate system requirements"""
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4:
                self.warnings.append(
                    f"Low available system memory: {available_gb:.1f}GB. "
                    "At least 8GB recommended."
                )
            
            # Check disk space in current directory
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 15:
                self.warnings.append(
                    f"Low disk space: {free_gb:.1f}GB available. "
                    "At least 15GB recommended for models and temporary files."
                )
                
        except ImportError:
            self.warnings.append("psutil not available - cannot check system resources")
        except Exception as e:
            self.warnings.append(f"Error checking system requirements: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


def validate_configuration(settings: Optional[Settings] = None) -> bool:
    """
    Validate application configuration
    
    Args:
        settings: Settings object to validate (uses global settings if None)
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if settings is None:
        from config.settings import settings
    
    validator = ConfigurationValidator(settings)
    return validator.validate_all()


def get_configuration_summary(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Get configuration validation summary
    
    Args:
        settings: Settings object to validate (uses global settings if None)
        
    Returns:
        Dictionary with validation results
    """
    if settings is None:
        from config.settings import settings
    
    validator = ConfigurationValidator(settings)
    validator.validate_all()
    return validator.get_validation_summary()