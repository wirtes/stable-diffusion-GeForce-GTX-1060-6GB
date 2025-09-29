"""
Application startup and initialization management
"""
import asyncio
import logging
import signal
import sys
import time
from typing import Dict, Any, List, Optional, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from config.settings import settings
from app.core.config import initialize_configuration
from app.core.logging import get_logger, ContextualLogger
from app.core.config_validator import validate_configuration


logger = get_logger(__name__)


class StartupManager:
    """Manages application startup sequence and validation"""
    
    def __init__(self):
        self.startup_tasks: List[Dict[str, Any]] = []
        self.shutdown_tasks: List[Dict[str, Any]] = []
        self.startup_completed = False
        self.startup_errors: List[str] = []
        self.startup_start_time: Optional[float] = None
        self.model_preload_task: Optional[asyncio.Task] = None
        
    def add_startup_task(self, name: str, func: Callable, critical: bool = True, timeout: int = 30):
        """
        Add a startup task
        
        Args:
            name: Task name for logging
            func: Async function to execute
            critical: If True, failure will prevent startup
            timeout: Task timeout in seconds
        """
        self.startup_tasks.append({
            'name': name,
            'func': func,
            'critical': critical,
            'timeout': timeout
        })
    
    def add_shutdown_task(self, name: str, func: Callable, timeout: int = 10):
        """
        Add a shutdown task
        
        Args:
            name: Task name for logging
            func: Async function to execute
            timeout: Task timeout in seconds
        """
        self.shutdown_tasks.append({
            'name': name,
            'func': func,
            'timeout': timeout
        })
    
    async def execute_startup_sequence(self) -> bool:
        """
        Execute the complete startup sequence
        
        Returns:
            True if startup successful, False otherwise
        """
        self.startup_start_time = time.time()
        startup_logger = ContextualLogger(__name__, {'phase': 'startup'})
        
        startup_logger.info("=== Starting Application Startup Sequence ===")
        
        try:
            # Phase 1: Configuration and validation
            startup_logger.info("Phase 1: Configuration initialization and validation")
            if not await self._initialize_configuration():
                return False
            
            # Phase 2: System requirements validation
            startup_logger.info("Phase 2: System requirements validation")
            if not await self._validate_system_requirements():
                return False
            
            # Phase 3: Execute startup tasks
            startup_logger.info("Phase 3: Executing startup tasks")
            if not await self._execute_startup_tasks():
                return False
            
            # Phase 4: Model preloading (non-blocking)
            startup_logger.info("Phase 4: Starting model preloading")
            await self._start_model_preloading()
            
            self.startup_completed = True
            startup_time = time.time() - self.startup_start_time
            startup_logger.info(f"=== Startup Sequence Completed Successfully in {startup_time:.2f}s ===")
            
            return True
            
        except Exception as e:
            startup_logger.error(f"Startup sequence failed: {e}", exc_info=True)
            self.startup_errors.append(f"Startup failed: {e}")
            return False
    
    async def execute_shutdown_sequence(self):
        """Execute the shutdown sequence"""
        shutdown_logger = ContextualLogger(__name__, {'phase': 'shutdown'})
        shutdown_logger.info("=== Starting Application Shutdown Sequence ===")
        
        # Cancel model preloading if still running
        if self.model_preload_task and not self.model_preload_task.done():
            shutdown_logger.info("Cancelling model preloading task")
            self.model_preload_task.cancel()
            try:
                await self.model_preload_task
            except asyncio.CancelledError:
                pass
        
        # Execute shutdown tasks
        for task in reversed(self.shutdown_tasks):  # Reverse order
            try:
                shutdown_logger.info(f"Executing shutdown task: {task['name']}")
                await asyncio.wait_for(task['func'](), timeout=task['timeout'])
                shutdown_logger.info(f"Shutdown task completed: {task['name']}")
            except asyncio.TimeoutError:
                shutdown_logger.warning(f"Shutdown task timed out: {task['name']}")
            except Exception as e:
                shutdown_logger.error(f"Shutdown task failed: {task['name']} - {e}")
        
        shutdown_logger.info("=== Shutdown Sequence Completed ===")
    
    async def _initialize_configuration(self) -> bool:
        """Initialize and validate configuration"""
        try:
            initialize_configuration()
            return True
        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            self.startup_errors.append(f"Configuration error: {e}")
            return False
    
    async def _validate_system_requirements(self) -> bool:
        """Validate system requirements"""
        try:
            # Run validation
            is_valid = validate_configuration(settings)
            
            if not is_valid:
                logger.error("System requirements validation failed")
                self.startup_errors.append("System requirements not met")
                return False
            
            # Additional startup-specific validations
            errors = settings.validate_startup_requirements()
            if errors:
                for error in errors:
                    logger.error(f"Startup requirement error: {error}")
                    self.startup_errors.append(error)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            self.startup_errors.append(f"System validation error: {e}")
            return False
    
    async def _execute_startup_tasks(self) -> bool:
        """Execute all registered startup tasks"""
        for task in self.startup_tasks:
            task_logger = ContextualLogger(__name__, {'task': task['name']})
            
            try:
                task_logger.info(f"Starting task: {task['name']}")
                start_time = time.time()
                
                await asyncio.wait_for(task['func'](), timeout=task['timeout'])
                
                execution_time = time.time() - start_time
                task_logger.info(f"Task completed in {execution_time:.2f}s: {task['name']}")
                
            except asyncio.TimeoutError:
                error_msg = f"Task timed out after {task['timeout']}s: {task['name']}"
                task_logger.error(error_msg)
                
                if task['critical']:
                    self.startup_errors.append(error_msg)
                    return False
                else:
                    task_logger.warning(f"Non-critical task failed, continuing: {task['name']}")
                    
            except Exception as e:
                error_msg = f"Task failed: {task['name']} - {e}"
                task_logger.error(error_msg, exc_info=True)
                
                if task['critical']:
                    self.startup_errors.append(error_msg)
                    return False
                else:
                    task_logger.warning(f"Non-critical task failed, continuing: {task['name']}")
        
        return True
    
    async def _start_model_preloading(self):
        """Start model preloading in background"""
        try:
            from app.services.stable_diffusion import model_manager
            
            async def preload_model():
                preload_logger = ContextualLogger(__name__, {'task': 'model_preload'})
                try:
                    preload_logger.info("Starting model preloading")
                    success = model_manager.initialize_model()
                    
                    if success:
                        preload_logger.info("Model preloading completed successfully")
                    else:
                        preload_logger.error("Model preloading failed")
                        
                except Exception as e:
                    preload_logger.error(f"Model preloading error: {e}", exc_info=True)
            
            # Start preloading task
            self.model_preload_task = asyncio.create_task(preload_model())
            
        except Exception as e:
            logger.error(f"Failed to start model preloading: {e}")
    
    def get_startup_status(self) -> Dict[str, Any]:
        """Get current startup status"""
        status = {
            'completed': self.startup_completed,
            'errors': self.startup_errors.copy(),
            'startup_time': None,
            'model_preload_status': 'not_started'
        }
        
        if self.startup_start_time:
            if self.startup_completed:
                status['startup_time'] = time.time() - self.startup_start_time
            else:
                status['startup_time'] = time.time() - self.startup_start_time
        
        # Check model preload status
        if self.model_preload_task:
            if self.model_preload_task.done():
                if self.model_preload_task.exception():
                    status['model_preload_status'] = 'failed'
                else:
                    status['model_preload_status'] = 'completed'
            else:
                status['model_preload_status'] = 'in_progress'
        
        return status


# Global startup manager instance
startup_manager = StartupManager()


async def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(startup_manager.execute_shutdown_sequence())
        sys.exit(0)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def initialize_directories():
    """Initialize required directories"""
    directories = [
        Path(settings.model.cache_dir),
        Path("logs") if settings.logging.file_path else None,
        Path("temp")
    ]
    
    for directory in directories:
        if directory:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise


async def validate_gpu_setup():
    """Validate GPU setup and configuration"""
    if settings.gpu.force_cpu:
        logger.info("Running in CPU-only mode")
        return
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU mode")
        settings.gpu.force_cpu = True
        return
    
    # Check GPU devices
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    # Set CUDA visible devices
    if settings.gpu.cuda_visible_devices:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu.cuda_visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={settings.gpu.cuda_visible_devices}")


async def cleanup_resources():
    """Cleanup application resources"""
    try:
        # Cleanup model resources
        from app.services.stable_diffusion import model_manager
        model_manager.cleanup_memory()
        logger.info("Model resources cleaned up")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
            
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")


async def initialize_monitoring_services():
    """Initialize performance monitoring and memory management services"""
    try:
        from app.services.memory_manager import memory_manager
        from app.services.performance_monitor import performance_monitor
        
        # Start memory monitoring
        memory_manager.start_monitoring()
        logger.info("Memory manager monitoring started")
        
        # Start performance monitoring
        performance_monitor.start_monitoring()
        logger.info("Performance monitor started")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring services: {e}")
        raise


async def cleanup_monitoring_services():
    """Cleanup monitoring services"""
    try:
        from app.services.memory_manager import memory_manager
        from app.services.performance_monitor import performance_monitor
        
        # Stop monitoring services
        memory_manager.stop_monitoring()
        performance_monitor.stop_monitoring()
        
        logger.info("Monitoring services stopped")
        
    except Exception as e:
        logger.error(f"Error stopping monitoring services: {e}")


def register_default_startup_tasks():
    """Register default startup tasks"""
    startup_manager.add_startup_task(
        "setup_signal_handlers",
        setup_signal_handlers,
        critical=False,
        timeout=5
    )
    
    startup_manager.add_startup_task(
        "initialize_directories",
        initialize_directories,
        critical=True,
        timeout=10
    )
    
    startup_manager.add_startup_task(
        "validate_gpu_setup",
        validate_gpu_setup,
        critical=True,
        timeout=15
    )
    
    startup_manager.add_startup_task(
        "initialize_monitoring_services",
        initialize_monitoring_services,
        critical=False,
        timeout=10
    )
    
    startup_manager.add_shutdown_task(
        "cleanup_monitoring_services",
        cleanup_monitoring_services,
        timeout=10
    )
    
    startup_manager.add_shutdown_task(
        "cleanup_resources",
        cleanup_resources,
        timeout=15
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    register_default_startup_tasks()
    
    startup_success = await startup_manager.execute_startup_sequence()
    if not startup_success:
        logger.error("Application startup failed")
        raise RuntimeError("Application startup failed")
    
    yield
    
    # Shutdown
    await startup_manager.execute_shutdown_sequence()


# Convenience function for backward compatibility
async def startup_event():
    """Legacy startup event handler"""
    register_default_startup_tasks()
    return await startup_manager.execute_startup_sequence()


async def shutdown_event():
    """Legacy shutdown event handler"""
    await startup_manager.execute_shutdown_sequence()