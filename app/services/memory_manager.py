"""
Advanced memory management service optimized for GTX 1060 6GB VRAM.
"""
import logging
import asyncio
import torch
import gc
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    timestamp: datetime
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_free_gb: float
    system_ram_percent: float
    system_ram_used_gb: float
    system_ram_total_gb: float


@dataclass
class RequestQueueItem:
    """Item in the request queue."""
    request_id: str
    priority: int
    estimated_memory_gb: float
    created_at: datetime
    future: asyncio.Future


class GTX1060MemoryManager:
    """
    Advanced memory manager optimized for GTX 1060 6GB VRAM constraints.
    Implements request queuing and memory monitoring to prevent OOM errors.
    """
    
    def __init__(self, max_vram_usage_gb: float = 5.0, max_queue_size: int = 10):
        self.max_vram_usage_gb = max_vram_usage_gb
        self.max_queue_size = max_queue_size
        self.request_queue: List[RequestQueueItem] = []
        self.active_requests: Dict[str, RequestQueueItem] = {}
        self.memory_history: List[MemoryStats] = []
        self.queue_lock = asyncio.Lock()
        self.is_processing = False
        self._cuda_available = torch.cuda.is_available()
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Memory thresholds for different image sizes
        self.memory_estimates = {
            (512, 512): 2.5,    # Base memory usage
            (768, 768): 4.0,    # Larger images need more memory
            (1024, 1024): 5.5,  # Maximum for GTX 1060
        }
        
        logger.info(f"GTX1060MemoryManager initialized - Max VRAM: {max_vram_usage_gb}GB, Queue size: {max_queue_size}")
        
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._memory_monitor_loop())
            logger.info("Memory monitoring started")
            
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("Memory monitoring stopped")
            
    async def _memory_monitor_loop(self):
        """Background task to monitor memory usage."""
        try:
            while True:
                stats = self.get_memory_stats()
                self.memory_history.append(stats)
                
                # Keep only last 100 entries
                if len(self.memory_history) > 100:
                    self.memory_history.pop(0)
                
                # Log warnings for high memory usage
                if stats.gpu_allocated_gb > self.max_vram_usage_gb * 0.8:
                    logger.warning(f"High GPU memory usage: {stats.gpu_allocated_gb:.1f}GB / {self.max_vram_usage_gb}GB")
                
                if stats.system_ram_percent > 85:
                    logger.warning(f"High system RAM usage: {stats.system_ram_percent}%")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
        except asyncio.CancelledError:
            logger.info("Memory monitoring cancelled")
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
            
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # System RAM
        ram = psutil.virtual_memory()
        
        # GPU memory
        gpu_allocated = 0.0
        gpu_reserved = 0.0
        gpu_free = 0.0
        
        if self._cuda_available:
            try:
                gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_properties = torch.cuda.get_device_properties(0)
                gpu_total = gpu_properties.total_memory / (1024**3)
                gpu_free = gpu_total - gpu_reserved
            except Exception as e:
                logger.warning(f"Could not get GPU memory stats: {e}")
        
        return MemoryStats(
            timestamp=datetime.now(),
            gpu_allocated_gb=gpu_allocated,
            gpu_reserved_gb=gpu_reserved,
            gpu_free_gb=gpu_free,
            system_ram_percent=ram.percent,
            system_ram_used_gb=ram.used / (1024**3),
            system_ram_total_gb=ram.total / (1024**3)
        )
        
    def estimate_memory_usage(self, width: int, height: int, steps: int) -> float:
        """
        Estimate memory usage for a generation request.
        
        Args:
            width: Image width
            height: Image height
            steps: Number of diffusion steps
            
        Returns:
            Estimated memory usage in GB
        """
        # Find closest size match
        base_memory = 2.5  # Default base memory
        
        for (w, h), memory in self.memory_estimates.items():
            if width <= w and height <= h:
                base_memory = memory
                break
        
        # Adjust for steps (more steps = slightly more memory)
        step_multiplier = 1.0 + (steps - 20) * 0.01  # 1% per step above 20
        
        # Adjust for resolution beyond base estimates
        resolution_factor = (width * height) / (512 * 512)
        if resolution_factor > 1:
            base_memory *= (1 + (resolution_factor - 1) * 0.5)
        
        estimated = base_memory * step_multiplier
        
        logger.debug(f"Memory estimate for {width}x{height}, {steps} steps: {estimated:.1f}GB")
        return estimated
        
    def can_process_request(self, estimated_memory_gb: float) -> bool:
        """
        Check if a request can be processed immediately.
        
        Args:
            estimated_memory_gb: Estimated memory usage
            
        Returns:
            True if request can be processed now
        """
        if not self._cuda_available:
            return True  # CPU mode has different constraints
            
        current_stats = self.get_memory_stats()
        available_memory = self.max_vram_usage_gb - current_stats.gpu_allocated_gb
        
        # Add safety margin
        safety_margin = 0.5  # 500MB safety margin
        can_process = available_memory >= (estimated_memory_gb + safety_margin)
        
        logger.debug(f"Memory check - Available: {available_memory:.1f}GB, Required: {estimated_memory_gb:.1f}GB, Can process: {can_process}")
        return can_process
        
    async def queue_request(self, request_id: str, width: int, height: int, steps: int, priority: int = 0) -> asyncio.Future:
        """
        Queue a request for processing when memory is available.
        
        Args:
            request_id: Unique request identifier
            width: Image width
            height: Image height
            steps: Number of diffusion steps
            priority: Request priority (higher = more important)
            
        Returns:
            Future that will be resolved when request can be processed
            
        Raises:
            RuntimeError: If queue is full
        """
        async with self.queue_lock:
            if len(self.request_queue) >= self.max_queue_size:
                raise RuntimeError(f"Request queue is full ({self.max_queue_size} items)")
            
            estimated_memory = self.estimate_memory_usage(width, height, steps)
            future = asyncio.Future()
            
            queue_item = RequestQueueItem(
                request_id=request_id,
                priority=priority,
                estimated_memory_gb=estimated_memory,
                created_at=datetime.now(),
                future=future
            )
            
            self.request_queue.append(queue_item)
            # Sort by priority (higher first), then by creation time
            self.request_queue.sort(key=lambda x: (-x.priority, x.created_at))
            
            logger.info(f"Queued request {request_id} (priority: {priority}, estimated memory: {estimated_memory:.1f}GB)")
            
            # Start processing if not already running
            if not self.is_processing:
                asyncio.create_task(self._process_queue())
            
            return future
            
    async def _process_queue(self):
        """Process queued requests when memory is available."""
        if self.is_processing:
            return
            
        self.is_processing = True
        logger.info("Started processing request queue")
        
        try:
            while True:
                async with self.queue_lock:
                    if not self.request_queue:
                        break
                    
                    # Find next processable request
                    processable_item = None
                    for i, item in enumerate(self.request_queue):
                        if self.can_process_request(item.estimated_memory_gb):
                            processable_item = self.request_queue.pop(i)
                            break
                    
                    if processable_item is None:
                        # No requests can be processed right now
                        await asyncio.sleep(1)
                        continue
                
                # Process the request
                logger.info(f"Processing queued request {processable_item.request_id}")
                self.active_requests[processable_item.request_id] = processable_item
                processable_item.future.set_result(True)
                
                # Small delay to allow request to start
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
        finally:
            self.is_processing = False
            logger.info("Stopped processing request queue")
            
    def mark_request_complete(self, request_id: str):
        """Mark a request as complete and clean up resources."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            logger.debug(f"Marked request {request_id} as complete")
            
            # Trigger memory cleanup
            self.cleanup_memory()
            
            # Resume queue processing if there are waiting requests
            if self.request_queue and not self.is_processing:
                asyncio.create_task(self._process_queue())
                
    def cleanup_memory(self):
        """Perform aggressive memory cleanup."""
        if self._cuda_available:
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Additional CUDA cleanup
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                
                logger.debug("Memory cleanup completed")
                
            except Exception as e:
                logger.warning(f"Memory cleanup error: {e}")
        else:
            # CPU-only cleanup
            gc.collect()
            
    @asynccontextmanager
    async def request_context(self, request_id: str, width: int, height: int, steps: int, priority: int = 0):
        """
        Context manager for handling requests with automatic queuing and cleanup.
        
        Args:
            request_id: Unique request identifier
            width: Image width
            height: Image height
            steps: Number of diffusion steps
            priority: Request priority
        """
        # Check if we can process immediately
        estimated_memory = self.estimate_memory_usage(width, height, steps)
        
        if self.can_process_request(estimated_memory):
            # Process immediately
            logger.info(f"Processing request {request_id} immediately")
            self.active_requests[request_id] = RequestQueueItem(
                request_id=request_id,
                priority=priority,
                estimated_memory_gb=estimated_memory,
                created_at=datetime.now(),
                future=None
            )
        else:
            # Queue the request
            logger.info(f"Queuing request {request_id} due to memory constraints")
            future = await self.queue_request(request_id, width, height, steps, priority)
            await future  # Wait for our turn
        
        try:
            yield
        finally:
            # Always clean up
            self.mark_request_complete(request_id)
            
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_length": len(self.request_queue),
            "active_requests": len(self.active_requests),
            "max_queue_size": self.max_queue_size,
            "is_processing": self.is_processing,
            "memory_stats": self.get_memory_stats().__dict__
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        if not self.memory_history:
            return {"error": "No memory history available"}
        
        recent_stats = self.memory_history[-10:]  # Last 10 entries
        
        avg_gpu_usage = sum(s.gpu_allocated_gb for s in recent_stats) / len(recent_stats)
        max_gpu_usage = max(s.gpu_allocated_gb for s in recent_stats)
        avg_ram_usage = sum(s.system_ram_percent for s in recent_stats) / len(recent_stats)
        
        return {
            "average_gpu_usage_gb": round(avg_gpu_usage, 2),
            "max_gpu_usage_gb": round(max_gpu_usage, 2),
            "average_ram_usage_percent": round(avg_ram_usage, 1),
            "memory_limit_gb": self.max_vram_usage_gb,
            "queue_utilization": len(self.request_queue) / self.max_queue_size,
            "active_requests": len(self.active_requests),
            "total_memory_samples": len(self.memory_history)
        }


# Global memory manager instance
memory_manager = GTX1060MemoryManager()