"""
Core configuration module
"""
from config.settings import settings
from app.core.config_validator import validate_configuration, get_configuration_summary
from app.core.logging import setup_logging, log_system_info, log_configuration

# Re-export for easy access
__all__ = [
    "settings",
    "validate_configuration", 
    "get_configuration_summary",
    "setup_logging",
    "log_system_info",
    "log_configuration"
]


def initialize_configuration():
    """
    Initialize and validate configuration
    
    Returns:
        bool: True if configuration is valid
    """
    # Setup logging first
    setup_logging(settings.logging)
    
    # Log system information
    log_system_info()
    
    # Log configuration
    log_configuration(settings)
    
    # Validate configuration
    is_valid = validate_configuration(settings)
    
    if not is_valid:
        summary = get_configuration_summary(settings)
        raise RuntimeError(
            f"Configuration validation failed with {summary['error_count']} errors. "
            f"Check logs for details."
        )
    
    return True