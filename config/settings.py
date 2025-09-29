"""
Application configuration settings
"""
import os
import logging
from typing import Optional, List
from pathlib import Path
from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class APISettings(BaseSettings):
    """API-specific configuration settings"""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    allowed_hosts: List[str] = Field(default=["*"])
    max_concurrent_requests: int = Field(default=1, ge=1, le=10)
    generation_timeout_seconds: int = Field(default=60, ge=10, le=300)
    
    @field_validator('allowed_hosts', mode='before')
    @classmethod
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
        return v


class ModelSettings(BaseSettings):
    """Model-specific configuration settings"""
    
    name: str = Field(default="runwayml/stable-diffusion-v1-5")
    cache_dir: str = Field(default="./models")
    torch_dtype: str = Field(default="float16")
    enable_attention_slicing: bool = Field(default=True)
    enable_cpu_offload: bool = Field(default=True)
    enable_model_cpu_offload: bool = Field(default=False)
    enable_vae_slicing: bool = Field(default=True)
    safety_checker: bool = Field(default=True)
    
    @field_validator('cache_dir')
    @classmethod
    def validate_cache_dir(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"Model cache directory {v} is not accessible")
        return str(path.absolute())
    
    @field_validator('torch_dtype')
    @classmethod
    def validate_torch_dtype(cls, v):
        valid_types = ['float16', 'float32', 'bfloat16']
        if v not in valid_types:
            raise ValueError(f"torch_dtype must be one of {valid_types}")
        return v
    
    @property
    def torch_dtype_obj(self):
        """Get the actual torch dtype object"""
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16
        }
        return dtype_map[self.torch_dtype]


class GPUSettings(BaseSettings):
    """GPU and hardware configuration settings"""
    
    cuda_visible_devices: str = Field(default="0")
    memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)
    allow_growth: bool = Field(default=True)
    force_cpu: bool = Field(default=False)
    
    @field_validator('cuda_visible_devices')
    @classmethod
    def validate_cuda_devices(cls, v):
        # Validate that device IDs are numeric
        if v and v != "-1":
            devices = v.split(',')
            for device in devices:
                try:
                    int(device.strip())
                except ValueError:
                    raise ValueError(f"Invalid CUDA device ID: {device}")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
    file_path: Optional[str] = Field(default=None)
    max_file_size: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5)
    
    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class ResourceSettings(BaseSettings):
    """Resource limits and performance settings"""
    
    memory_limit: str = Field(default="8G")
    cpu_limit: float = Field(default=4.0, ge=0.1, le=32.0)
    worker_processes: int = Field(default=1, ge=1, le=8)
    
    @field_validator('memory_limit')
    @classmethod
    def validate_memory_limit(cls, v):
        # Validate memory limit format (e.g., "8G", "4096M")
        import re
        if not re.match(r'^\d+[GMK]?$', v.upper()):
            raise ValueError("Memory limit must be in format like '8G', '4096M', or '1024'")
        return v.upper()


class Settings(BaseSettings):
    """Main application settings"""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Sub-configurations
    api: APISettings = APISettings()
    model: ModelSettings = ModelSettings()
    gpu: GPUSettings = GPUSettings()
    logging: LoggingSettings = LoggingSettings()
    resources: ResourceSettings = ResourceSettings()
    
    @model_validator(mode='after')
    def validate_configuration(self):
        """Validate the overall configuration consistency"""
        # If forcing CPU, disable GPU-specific optimizations
        if self.gpu.force_cpu:
            self.model.enable_cpu_offload = False
            self.model.enable_model_cpu_offload = True
            self.model.torch_dtype = "float32"  # CPU works better with float32
        
        # Validate CUDA availability if not forcing CPU
        if not self.gpu.force_cpu and not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU mode")
            self.gpu.force_cpu = True
            self.model.enable_cpu_offload = False
            self.model.enable_model_cpu_offload = True
        
        return self
    
    def validate_startup_requirements(self) -> List[str]:
        """Validate all startup requirements and return list of errors"""
        errors = []
        
        # Check model cache directory
        try:
            cache_path = Path(self.model.cache_dir)
            if not cache_path.exists():
                cache_path.mkdir(parents=True, exist_ok=True)
            if not cache_path.is_dir():
                errors.append(f"Model cache directory {self.model.cache_dir} is not accessible")
        except Exception as e:
            errors.append(f"Cannot create model cache directory: {e}")
        
        # Check GPU availability if required
        if not self.gpu.force_cpu:
            if not torch.cuda.is_available():
                errors.append("CUDA not available but GPU mode requested")
            else:
                # Check specific GPU device
                try:
                    device_count = torch.cuda.device_count()
                    requested_devices = [int(d.strip()) for d in self.gpu.cuda_visible_devices.split(',') if d.strip()]
                    for device_id in requested_devices:
                        if device_id >= device_count:
                            errors.append(f"GPU device {device_id} not available (only {device_count} devices found)")
                except Exception as e:
                    errors.append(f"Error checking GPU devices: {e}")
        
        # Check port availability
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex((self.api.host, self.api.port))
                if result == 0:
                    errors.append(f"Port {self.api.port} is already in use")
        except Exception as e:
            errors.append(f"Cannot check port availability: {e}")
        
        return errors
    
    def get_model_config(self) -> dict:
        """Get model configuration for stable diffusion pipeline"""
        return {
            "torch_dtype": self.model.torch_dtype_obj,
            "cache_dir": self.model.cache_dir,
            "safety_checker": None if not self.model.safety_checker else "default",
            "requires_safety_checker": self.model.safety_checker,
        }
    
    def get_pipeline_config(self) -> dict:
        """Get pipeline optimization configuration"""
        return {
            "enable_attention_slicing": self.model.enable_attention_slicing,
            "enable_cpu_offload": self.model.enable_cpu_offload,
            "enable_model_cpu_offload": self.model.enable_model_cpu_offload,
            "enable_vae_slicing": self.model.enable_vae_slicing,
        }


# Global settings instance
settings = Settings()