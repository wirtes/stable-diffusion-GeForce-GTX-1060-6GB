"""
Health monitoring and status checking
"""
import asyncio
import logging
import time
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import torch

from config.settings import settings
from app.core.logging import get_logger
from app.core.startup import startup_manager


logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a component"""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    last_check: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        if self.last_check:
            result['last_check'] = self.last_check.isoformat()
        return result


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.component_health: Dict[str, ComponentHealth] = {}
        self.last_full_check: Optional[datetime] = None
        self.check_interval = 30  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Health monitoring already running")
            return
        
        logger.info("Starting health monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self.monitoring_task:
            logger.info("Stopping health monitoring")
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        # Check all components
        await self._check_startup_status()
        await self._check_model_status()
        await self._check_gpu_status()
        await self._check_system_resources()
        await self._check_api_status()
        
        # Collect metrics
        metrics = await self._collect_system_metrics()
        
        # Update history
        self._update_metrics_history(metrics)
        
        self.last_full_check = datetime.now()
        check_duration = time.time() - start_time
        
        # Determine overall health
        overall_status = self._determine_overall_health()
        
        health_report = {
            'overall_status': overall_status.value,
            'components': {name: comp.to_dict() for name, comp in self.component_health.items()},
            'metrics': metrics.to_dict() if metrics else None,
            'last_check': self.last_full_check.isoformat(),
            'check_duration_ms': round(check_duration * 1000, 2)
        }
        
        logger.debug(f"Health check completed in {check_duration:.3f}s, status: {overall_status.value}")
        
        return health_report
    
    async def _check_startup_status(self):
        """Check application startup status"""
        try:
            startup_status = startup_manager.get_startup_status()
            
            if startup_status['completed']:
                if startup_status['errors']:
                    status = HealthStatus.DEGRADED
                    message = f"Startup completed with {len(startup_status['errors'])} errors"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Startup completed successfully in {startup_status.get('startup_time', 0):.2f}s"
            else:
                status = HealthStatus.INITIALIZING
                message = "Application startup in progress"
            
            self.component_health['startup'] = ComponentHealth(
                name='startup',
                status=status,
                message=message,
                details=startup_status,
                last_check=datetime.now()
            )
            
        except Exception as e:
            self.component_health['startup'] = ComponentHealth(
                name='startup',
                status=HealthStatus.UNKNOWN,
                message=f"Error checking startup status: {e}",
                last_check=datetime.now()
            )
    
    async def _check_model_status(self):
        """Check model loading and readiness status"""
        try:
            from app.services.stable_diffusion import model_manager
            
            model_info = model_manager.get_model_info()
            
            if model_manager.is_ready():
                status = HealthStatus.HEALTHY
                message = "Model loaded and ready for inference"
            elif model_info.get('is_initialized', False):
                status = HealthStatus.INITIALIZING
                message = "Model initialized but not ready"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Model not initialized"
            
            self.component_health['model'] = ComponentHealth(
                name='model',
                status=status,
                message=message,
                details=model_info,
                last_check=datetime.now()
            )
            
        except Exception as e:
            self.component_health['model'] = ComponentHealth(
                name='model',
                status=HealthStatus.UNKNOWN,
                message=f"Error checking model status: {e}",
                last_check=datetime.now()
            )
    
    async def _check_gpu_status(self):
        """Check GPU availability and status"""
        try:
            if settings.gpu.force_cpu:
                status = HealthStatus.HEALTHY
                message = "Running in CPU-only mode"
                details = {'mode': 'cpu', 'cuda_available': False}
            elif not torch.cuda.is_available():
                status = HealthStatus.DEGRADED
                message = "CUDA not available, using CPU fallback"
                details = {'mode': 'cpu_fallback', 'cuda_available': False}
            else:
                # Check GPU health
                device_count = torch.cuda.device_count()
                gpu_details = {
                    'cuda_available': True,
                    'device_count': device_count,
                    'devices': []
                }
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    memory_total = props.total_memory / (1024**2)  # MB
                    memory_percent = (memory_allocated / memory_total) * 100
                    
                    device_info = {
                        'id': i,
                        'name': props.name,
                        'memory_total_mb': round(memory_total, 1),
                        'memory_used_mb': round(memory_allocated, 1),
                        'memory_percent': round(memory_percent, 1)
                    }
                    gpu_details['devices'].append(device_info)
                
                # Determine status based on memory usage
                max_memory_percent = max([d['memory_percent'] for d in gpu_details['devices']], default=0)
                
                if max_memory_percent > 90:
                    status = HealthStatus.DEGRADED
                    message = f"High GPU memory usage: {max_memory_percent:.1f}%"
                elif max_memory_percent > 95:
                    status = HealthStatus.UNHEALTHY
                    message = f"Critical GPU memory usage: {max_memory_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"GPU healthy, memory usage: {max_memory_percent:.1f}%"
                
                details = gpu_details
            
            self.component_health['gpu'] = ComponentHealth(
                name='gpu',
                status=status,
                message=message,
                details=details,
                last_check=datetime.now()
            )
            
        except Exception as e:
            self.component_health['gpu'] = ComponentHealth(
                name='gpu',
                status=HealthStatus.UNKNOWN,
                message=f"Error checking GPU status: {e}",
                last_check=datetime.now()
            )
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Determine status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical resource usage"
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 85:
                status = HealthStatus.DEGRADED
                message = "High resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            details = {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory_percent, 1),
                'memory_used_gb': round(memory.used / (1024**3), 1),
                'memory_total_gb': round(memory.total / (1024**3), 1),
                'disk_percent': round(disk_percent, 1),
                'disk_free_gb': round(disk_free_gb, 1)
            }
            
            self.component_health['system'] = ComponentHealth(
                name='system',
                status=status,
                message=message,
                details=details,
                last_check=datetime.now()
            )
            
        except Exception as e:
            self.component_health['system'] = ComponentHealth(
                name='system',
                status=HealthStatus.UNKNOWN,
                message=f"Error checking system resources: {e}",
                last_check=datetime.now()
            )
    
    async def _check_api_status(self):
        """Check API service status"""
        try:
            # Check if we can import and access key services
            from app.services.stable_diffusion import generation_service
            
            generation_status = generation_service.get_generation_status()
            
            if generation_status.get('active_requests', 0) > settings.api.max_concurrent_requests:
                status = HealthStatus.DEGRADED
                message = "High request load"
            else:
                status = HealthStatus.HEALTHY
                message = "API service healthy"
            
            self.component_health['api'] = ComponentHealth(
                name='api',
                status=status,
                message=message,
                details=generation_status,
                last_check=datetime.now()
            )
            
        except Exception as e:
            self.component_health['api'] = ComponentHealth(
                name='api',
                status=HealthStatus.UNKNOWN,
                message=f"Error checking API status: {e}",
                last_check=datetime.now()
            )
    
    async def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect detailed system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            metrics = SystemMetrics(
                cpu_percent=round(cpu_percent, 1),
                memory_percent=round(memory.percent, 1),
                memory_used_gb=round(memory.used / (1024**3), 2),
                memory_total_gb=round(memory.total / (1024**3), 2),
                disk_percent=round((disk.used / disk.total) * 100, 1),
                disk_free_gb=round(disk.free / (1024**3), 2)
            )
            
            # GPU metrics if available
            if torch.cuda.is_available() and not settings.gpu.force_cpu:
                try:
                    memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                    
                    metrics.gpu_memory_used_mb = round(memory_allocated, 1)
                    metrics.gpu_memory_total_mb = round(memory_total, 1)
                    metrics.gpu_memory_percent = round((memory_allocated / memory_total) * 100, 1)
                    
                    # Try to get GPU temperature (may not be available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics.gpu_temperature = temp
                    except:
                        pass  # Temperature monitoring not available
                        
                except Exception:
                    pass  # GPU metrics not available
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _update_metrics_history(self, metrics: Optional[SystemMetrics]):
        """Update metrics history"""
        if metrics:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics.to_dict()
            }
            
            self.metrics_history.append(history_entry)
            
            # Keep only recent history
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _determine_overall_health(self) -> HealthStatus:
        """Determine overall health status from components"""
        if not self.component_health:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in self.component_health.values()]
        
        # If any component is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any component is initializing, overall is initializing
        if HealthStatus.INITIALIZING in statuses:
            return HealthStatus.INITIALIZING
        
        # If any component is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If all components are healthy, overall is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status"""
        overall_status = self._determine_overall_health()
        
        return {
            'overall_status': overall_status.value,
            'component_count': len(self.component_health),
            'healthy_components': len([c for c in self.component_health.values() if c.status == HealthStatus.HEALTHY]),
            'degraded_components': len([c for c in self.component_health.values() if c.status == HealthStatus.DEGRADED]),
            'unhealthy_components': len([c for c in self.component_health.values() if c.status == HealthStatus.UNHEALTHY]),
            'last_check': self.last_full_check.isoformat() if self.last_full_check else None
        }
    
    def get_metrics_history(self, minutes: int = 30) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            entry for entry in self.metrics_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]


# Global health monitor instance
health_monitor = HealthMonitor()


async def get_health_status() -> Dict[str, Any]:
    """Get current health status (convenience function)"""
    return await health_monitor.perform_health_check()


async def get_health_summary() -> Dict[str, Any]:
    """Get health summary (convenience function)"""
    return health_monitor.get_health_summary()