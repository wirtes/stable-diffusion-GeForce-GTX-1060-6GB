"""
Performance monitoring service for tracking generation metrics and system performance.
"""
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for a single image generation."""
    request_id: str
    timestamp: datetime
    prompt_length: int
    width: int
    height: int
    steps: int
    seed: int
    generation_time_seconds: float
    queue_wait_time_seconds: float
    memory_used_gb: float
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_utilization_percent: float
    temperature_celsius: Optional[float] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_generation_time: float
    median_generation_time: float
    p95_generation_time: float
    average_queue_wait_time: float
    average_memory_usage_gb: float
    peak_memory_usage_gb: float
    requests_per_minute: float
    error_rate_percent: float
    most_common_errors: List[Dict[str, Any]]


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for the stable diffusion API.
    Tracks generation metrics, system performance, and provides analytics.
    """
    
    def __init__(self, max_history_size: int = 1000, metrics_file: Optional[str] = None):
        self.max_history_size = max_history_size
        self.metrics_file = metrics_file
        
        # Storage for metrics
        self.generation_metrics: deque = deque(maxlen=max_history_size)
        self.system_metrics: deque = deque(maxlen=max_history_size)
        self.error_counts: defaultdict = defaultdict(int)
        
        # Active request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            "max_generation_time": 120.0,  # seconds
            "max_queue_wait_time": 60.0,   # seconds
            "max_memory_usage": 5.5,       # GB
            "max_error_rate": 10.0,        # percent
        }
        
        logger.info(f"PerformanceMonitor initialized - History size: {max_history_size}")
        
    def start_monitoring(self):
        """Start background system monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._system_monitor_loop())
            logger.info("Performance monitoring started")
            
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("Performance monitoring stopped")
            
    async def _system_monitor_loop(self):
        """Background task to collect system metrics."""
        try:
            import psutil
            import torch
            
            while True:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # GPU metrics
                    gpu_memory_used = 0.0
                    gpu_memory_total = 0.0
                    gpu_utilization = 0.0
                    temperature = None
                    
                    if torch.cuda.is_available():
                        try:
                            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                            gpu_props = torch.cuda.get_device_properties(0)
                            gpu_memory_total = gpu_props.total_memory / (1024**3)
                            
                            # Try to get GPU utilization (requires nvidia-ml-py)
                            try:
                                import pynvml
                                pynvml.nvmlInit()
                                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_utilization = util.gpu
                                
                                # Get temperature
                                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                                temperature = temp
                                
                            except ImportError:
                                logger.debug("pynvml not available, GPU utilization not monitored")
                            except Exception as e:
                                logger.debug(f"Could not get GPU utilization: {e}")
                                
                        except Exception as e:
                            logger.debug(f"Could not get GPU memory stats: {e}")
                    
                    # Store system metrics
                    system_metric = SystemMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        gpu_memory_used_gb=gpu_memory_used,
                        gpu_memory_total_gb=gpu_memory_total,
                        gpu_utilization_percent=gpu_utilization,
                        temperature_celsius=temperature
                    )
                    
                    with self._lock:
                        self.system_metrics.append(system_metric)
                    
                    # Check for performance issues
                    self._check_performance_thresholds(system_metric)
                    
                except Exception as e:
                    logger.warning(f"Error collecting system metrics: {e}")
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
        except asyncio.CancelledError:
            logger.info("System monitoring cancelled")
        except Exception as e:
            logger.error(f"System monitoring error: {e}")
            
    def _check_performance_thresholds(self, system_metric: SystemMetrics):
        """Check if system metrics exceed performance thresholds."""
        if system_metric.memory_percent > 90:
            logger.warning(f"High system memory usage: {system_metric.memory_percent}%")
            
        if system_metric.gpu_memory_used_gb > self.thresholds["max_memory_usage"]:
            logger.warning(f"High GPU memory usage: {system_metric.gpu_memory_used_gb:.1f}GB")
            
        if system_metric.temperature_celsius and system_metric.temperature_celsius > 80:
            logger.warning(f"High GPU temperature: {system_metric.temperature_celsius}°C")
            
        if system_metric.cpu_percent > 95:
            logger.warning(f"High CPU usage: {system_metric.cpu_percent}%")
            
    def start_request_tracking(self, request_id: str, prompt: str, width: int, height: int, steps: int, seed: int):
        """Start tracking a new generation request."""
        with self._lock:
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "queue_start_time": time.time(),
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
                "memory_start": self._get_current_memory_usage()
            }
        
        logger.debug(f"Started tracking request {request_id}")
        
    def mark_request_processing(self, request_id: str):
        """Mark that a request has started processing (left the queue)."""
        with self._lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]["processing_start_time"] = time.time()
                
    def complete_request_tracking(self, request_id: str, success: bool = True, error_type: str = None, error_message: str = None):
        """Complete tracking for a generation request."""
        with self._lock:
            if request_id not in self.active_requests:
                logger.warning(f"Request {request_id} not found in active tracking")
                return
            
            request_data = self.active_requests.pop(request_id)
            end_time = time.time()
            
            # Calculate timings
            total_time = end_time - request_data["start_time"]
            queue_wait_time = request_data.get("processing_start_time", end_time) - request_data["queue_start_time"]
            
            # Calculate memory usage
            memory_used = max(
                self._get_current_memory_usage() - request_data["memory_start"],
                0.0
            )
            
            # Create metrics record
            metrics = GenerationMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                prompt_length=len(request_data["prompt"]),
                width=request_data["width"],
                height=request_data["height"],
                steps=request_data["steps"],
                seed=request_data["seed"],
                generation_time_seconds=total_time,
                queue_wait_time_seconds=queue_wait_time,
                memory_used_gb=memory_used,
                success=success,
                error_type=error_type,
                error_message=error_message
            )
            
            # Store metrics
            self.generation_metrics.append(metrics)
            
            # Track errors
            if not success and error_type:
                self.error_counts[error_type] += 1
            
            # Save to file if configured
            if self.metrics_file:
                self._save_metrics_to_file(metrics)
            
            logger.info(f"Completed tracking request {request_id} - Success: {success}, Time: {total_time:.2f}s")
            
    def _get_current_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            pass
        return 0.0
        
    def _save_metrics_to_file(self, metrics: GenerationMetrics):
        """Save metrics to file for persistence."""
        try:
            metrics_dict = asdict(metrics)
            metrics_dict["timestamp"] = metrics.timestamp.isoformat()
            
            # Append to file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics_dict) + "\n")
                
        except Exception as e:
            logger.warning(f"Could not save metrics to file: {e}")
            
    def get_performance_stats(self, time_window_minutes: int = 60) -> PerformanceStats:
        """
        Get aggregated performance statistics.
        
        Args:
            time_window_minutes: Time window for statistics calculation
            
        Returns:
            PerformanceStats object with aggregated metrics
        """
        with self._lock:
            # Filter metrics by time window
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_metrics = [
                m for m in self.generation_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return PerformanceStats(
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    average_generation_time=0.0,
                    median_generation_time=0.0,
                    p95_generation_time=0.0,
                    average_queue_wait_time=0.0,
                    average_memory_usage_gb=0.0,
                    peak_memory_usage_gb=0.0,
                    requests_per_minute=0.0,
                    error_rate_percent=0.0,
                    most_common_errors=[]
                )
            
            # Calculate statistics
            total_requests = len(recent_metrics)
            successful_requests = sum(1 for m in recent_metrics if m.success)
            failed_requests = total_requests - successful_requests
            
            generation_times = [m.generation_time_seconds for m in recent_metrics]
            queue_wait_times = [m.queue_wait_time_seconds for m in recent_metrics]
            memory_usage = [m.memory_used_gb for m in recent_metrics]
            
            # Sort for percentile calculations
            generation_times.sort()
            
            # Calculate percentiles
            def percentile(data: List[float], p: float) -> float:
                if not data:
                    return 0.0
                k = (len(data) - 1) * p
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return data[f] * (1 - c) + data[f + 1] * c
                return data[f]
            
            # Error analysis
            recent_errors = defaultdict(int)
            for m in recent_metrics:
                if not m.success and m.error_type:
                    recent_errors[m.error_type] += 1
            
            most_common_errors = [
                {"error_type": error_type, "count": count, "percentage": (count / failed_requests) * 100}
                for error_type, count in sorted(recent_errors.items(), key=lambda x: x[1], reverse=True)
            ][:5]  # Top 5 errors
            
            return PerformanceStats(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_generation_time=sum(generation_times) / len(generation_times) if generation_times else 0.0,
                median_generation_time=percentile(generation_times, 0.5),
                p95_generation_time=percentile(generation_times, 0.95),
                average_queue_wait_time=sum(queue_wait_times) / len(queue_wait_times) if queue_wait_times else 0.0,
                average_memory_usage_gb=sum(memory_usage) / len(memory_usage) if memory_usage else 0.0,
                peak_memory_usage_gb=max(memory_usage) if memory_usage else 0.0,
                requests_per_minute=(total_requests / time_window_minutes) if time_window_minutes > 0 else 0.0,
                error_rate_percent=(failed_requests / total_requests) * 100 if total_requests > 0 else 0.0,
                most_common_errors=most_common_errors
            )
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        with self._lock:
            if not self.system_metrics:
                return {"status": "no_data", "message": "No system metrics available"}
            
            latest_metric = self.system_metrics[-1]
            
            # Determine health status
            issues = []
            
            if latest_metric.memory_percent > 90:
                issues.append(f"High system memory usage: {latest_metric.memory_percent}%")
                
            if latest_metric.gpu_memory_used_gb > self.thresholds["max_memory_usage"]:
                issues.append(f"High GPU memory usage: {latest_metric.gpu_memory_used_gb:.1f}GB")
                
            if latest_metric.temperature_celsius and latest_metric.temperature_celsius > 80:
                issues.append(f"High GPU temperature: {latest_metric.temperature_celsius}°C")
                
            if latest_metric.cpu_percent > 95:
                issues.append(f"High CPU usage: {latest_metric.cpu_percent}%")
            
            # Determine overall status
            if not issues:
                status = "healthy"
            elif len(issues) <= 2:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "status": status,
                "timestamp": latest_metric.timestamp.isoformat(),
                "issues": issues,
                "metrics": {
                    "cpu_percent": latest_metric.cpu_percent,
                    "memory_percent": latest_metric.memory_percent,
                    "gpu_memory_used_gb": latest_metric.gpu_memory_used_gb,
                    "gpu_memory_total_gb": latest_metric.gpu_memory_total_gb,
                    "gpu_utilization_percent": latest_metric.gpu_utilization_percent,
                    "temperature_celsius": latest_metric.temperature_celsius
                },
                "active_requests": len(self.active_requests)
            }
            
    def export_metrics(self, filepath: str, format: str = "json"):
        """
        Export all metrics to a file.
        
        Args:
            filepath: Path to save the metrics
            format: Export format ("json" or "csv")
        """
        try:
            with self._lock:
                if format.lower() == "json":
                    self._export_json(filepath)
                elif format.lower() == "csv":
                    self._export_csv(filepath)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                    
            logger.info(f"Exported metrics to {filepath} in {format} format")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
            
    def _export_json(self, filepath: str):
        """Export metrics in JSON format."""
        data = {
            "generation_metrics": [asdict(m) for m in self.generation_metrics],
            "system_metrics": [asdict(m) for m in self.system_metrics],
            "error_counts": dict(self.error_counts),
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        for metrics_list in [data["generation_metrics"], data["system_metrics"]]:
            for metric in metrics_list:
                if "timestamp" in metric:
                    metric["timestamp"] = metric["timestamp"].isoformat() if hasattr(metric["timestamp"], "isoformat") else str(metric["timestamp"])
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
            
    def _export_csv(self, filepath: str):
        """Export generation metrics in CSV format."""
        import csv
        
        if not self.generation_metrics:
            return
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.generation_metrics[0]).keys())
            writer.writeheader()
            
            for metric in self.generation_metrics:
                row = asdict(metric)
                row["timestamp"] = metric.timestamp.isoformat()
                writer.writerow(row)


# Global performance monitor instance
performance_monitor = PerformanceMonitor(
    max_history_size=1000,
    metrics_file="performance_metrics.jsonl"
)