"""
Application configuration settings
"""
import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # GPU Configuration
    cuda_visible_devices: str = "0"
    
    # Model Configuration
    model_cache_dir: str = "./models"
    model_name: str = "runwayml/stable-diffusion-v1-5"
    
    # Performance Configuration
    max_concurrent_requests: int = 1
    generation_timeout_seconds: int = 60
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()