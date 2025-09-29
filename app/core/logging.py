"""
Logging configuration for the Stable Diffusion API
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import LoggingSettings


def setup_logging(logging_settings: Optional[LoggingSettings] = None) -> None:
    """
    Configure logging for the application
    
    Args:
        logging_settings: Logging configuration settings
    """
    if logging_settings is None:
        from config.settings import settings
        logging_settings = settings.logging
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=logging_settings.format,
        datefmt=logging_settings.date_format
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_settings.level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, logging_settings.level))
    root_logger.addHandler(console_handler)
    
    # Create file handler if specified
    if logging_settings.file_path:
        try:
            log_file = Path(logging_settings.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=logging_settings.max_file_size,
                backupCount=logging_settings.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, logging_settings.level))
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"File logging enabled: {log_file}")
        except Exception as e:
            root_logger.warning(f"Failed to setup file logging: {e}")
    
    # Configure specific loggers with appropriate levels
    configure_third_party_loggers(logging_settings.level)
    
    root_logger.info(f"Logging configured with level: {logging_settings.level}")


def configure_third_party_loggers(log_level: str) -> None:
    """
    Configure third-party library loggers
    
    Args:
        log_level: Base logging level
    """
    # Map our log levels to appropriate levels for third-party libraries
    level_mapping = {
        'DEBUG': {
            'uvicorn': logging.DEBUG,
            'uvicorn.access': logging.DEBUG,
            'diffusers': logging.INFO,
            'transformers': logging.INFO,
            'torch': logging.INFO,
            'PIL': logging.WARNING,
        },
        'INFO': {
            'uvicorn': logging.INFO,
            'uvicorn.access': logging.INFO,
            'diffusers': logging.WARNING,
            'transformers': logging.WARNING,
            'torch': logging.WARNING,
            'PIL': logging.WARNING,
        },
        'WARNING': {
            'uvicorn': logging.WARNING,
            'uvicorn.access': logging.WARNING,
            'diffusers': logging.WARNING,
            'transformers': logging.WARNING,
            'torch': logging.WARNING,
            'PIL': logging.ERROR,
        },
        'ERROR': {
            'uvicorn': logging.ERROR,
            'uvicorn.access': logging.ERROR,
            'diffusers': logging.ERROR,
            'transformers': logging.ERROR,
            'torch': logging.ERROR,
            'PIL': logging.ERROR,
        },
        'CRITICAL': {
            'uvicorn': logging.CRITICAL,
            'uvicorn.access': logging.CRITICAL,
            'diffusers': logging.CRITICAL,
            'transformers': logging.CRITICAL,
            'torch': logging.CRITICAL,
            'PIL': logging.CRITICAL,
        }
    }
    
    # Apply the configuration
    logger_config = level_mapping.get(log_level, level_mapping['INFO'])
    for logger_name, level in logger_config.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info() -> None:
    """Log system information for debugging"""
    logger = get_logger(__name__)
    
    try:
        import torch
        import platform
        import psutil
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        # GPU Information
        if torch.cuda.is_available():
            logger.info(f"CUDA Available: True")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("CUDA Available: False")
        
        # Memory Information
        memory = psutil.virtual_memory()
        logger.info(f"System Memory: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available")
        
        logger.info("=== End System Information ===")
        
    except ImportError as e:
        logger.warning(f"Could not log full system info: {e}")
    except Exception as e:
        logger.error(f"Error logging system info: {e}")


def log_configuration(settings_obj) -> None:
    """
    Log current configuration (without sensitive data)
    
    Args:
        settings_obj: Settings object to log
    """
    logger = get_logger(__name__)
    
    logger.info("=== Configuration ===")
    logger.info(f"Environment: {settings_obj.environment}")
    logger.info(f"Debug Mode: {settings_obj.debug}")
    
    # API Settings
    logger.info(f"API Host: {settings_obj.api.host}")
    logger.info(f"API Port: {settings_obj.api.port}")
    logger.info(f"Max Concurrent Requests: {settings_obj.api.max_concurrent_requests}")
    logger.info(f"Generation Timeout: {settings_obj.api.generation_timeout_seconds}s")
    
    # Model Settings
    logger.info(f"Model: {settings_obj.model.name}")
    logger.info(f"Model Cache: {settings_obj.model.cache_dir}")
    logger.info(f"Torch Dtype: {settings_obj.model.torch_dtype}")
    logger.info(f"Attention Slicing: {settings_obj.model.enable_attention_slicing}")
    logger.info(f"CPU Offload: {settings_obj.model.enable_cpu_offload}")
    
    # GPU Settings
    logger.info(f"CUDA Devices: {settings_obj.gpu.cuda_visible_devices}")
    logger.info(f"Force CPU: {settings_obj.gpu.force_cpu}")
    logger.info(f"GPU Memory Fraction: {settings_obj.gpu.memory_fraction}")
    
    logger.info("=== End Configuration ===")


class ContextualLogger:
    """Logger with contextual information"""
    
    def __init__(self, name: str, context: Dict[str, Any] = None):
        self.logger = get_logger(name)
        self.context = context or {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context"""
        if self.context:
            context_str = " | ".join([f"{k}={v}" for k, v in self.context.items()])
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message), **kwargs)