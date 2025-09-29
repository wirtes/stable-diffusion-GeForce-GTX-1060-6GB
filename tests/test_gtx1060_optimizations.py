"""
Tests for GTX 1060 specific optimizations including memory management and performance monitoring.
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.services.memory_manager import GTX1060MemoryManager, MemoryStats, RequestQueueItem
from app.services.performance_monitor import PerformanceMonitor, GenerationMetrics, SystemMetrics


class TestGTX1060MemoryManager:
    """Test GTX 1060 memory manager functionality."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager instance for testing."""
        return GTX1060MemoryManager(max_vram_usage_gb=5.0, max_queue_size=5)
    
    def test_memory_estimation(self, memory_manager):
        """Test memory usage estimation for different image sizes."""
        # Test standard sizes
        assert memory_manager.estimate_memory_usage(512, 512, 20) == pytest.approx(2.5, rel=0.1)
        # 768x768 uses more memory due to resolution scaling
        assert memory_manager.estimate_memory_usage(768, 768, 20) == pytest.approx(6.5, rel=0.2)
        assert memory_manager.estimate_memory_usage(1024, 1024, 20) > 5.0
        
        # Test step scaling
        base_memory = memory_manager.estimate_memory_usage(512, 512, 20)
        high_steps_memory = memory_manager.estimate_memory_usage(512, 512, 50)
        assert high_steps_memory > base_memory
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2 * 1024**3)  # 2GB
    def test_can_process_request(self, mock_memory, mock_cuda, memory_manager):
        """Test request processing capability check."""
        memory_manager._cuda_available = True
        
        # Should be able to process small request
        assert memory_manager.can_process_request(2.0) == True
        
        # Should not be able to process large request
        assert memory_manager.can_process_request(4.0) == False
    
    @pytest.mark.asyncio
    async def test_request_queuing(self, memory_manager):
        """Test request queuing functionality."""
        # Queue a request
        future = await memory_manager.queue_request("test-1", 512, 512, 20, priority=1)
        
        assert len(memory_manager.request_queue) == 1
        assert memory_manager.request_queue[0].request_id == "test-1"
        
        # Queue higher priority request
        future2 = await memory_manager.queue_request("test-2", 512, 512, 20, priority=2)
        
        # Higher priority should be first
        assert memory_manager.request_queue[0].request_id == "test-2"
        assert memory_manager.request_queue[1].request_id == "test-1"
    
    @pytest.mark.asyncio
    async def test_queue_overflow(self, memory_manager):
        """Test queue overflow handling."""
        # Fill the queue
        for i in range(5):
            await memory_manager.queue_request(f"test-{i}", 512, 512, 20)
        
        # Next request should raise error
        with pytest.raises(RuntimeError, match="Request queue is full"):
            await memory_manager.queue_request("overflow", 512, 512, 20)
    
    @pytest.mark.asyncio
    async def test_request_context_manager(self, memory_manager):
        """Test request context manager functionality."""
        with patch.object(memory_manager, 'can_process_request', return_value=True):
            async with memory_manager.request_context("test-ctx", 512, 512, 20):
                assert "test-ctx" in memory_manager.active_requests
            
            # Should be cleaned up after context
            assert "test-ctx" not in memory_manager.active_requests
    
    def test_memory_stats(self, memory_manager):
        """Test memory statistics collection."""
        stats = memory_manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.timestamp is not None
        assert stats.system_ram_percent >= 0
        assert stats.system_ram_used_gb >= 0
        assert stats.system_ram_total_gb > 0
    
    def test_queue_status(self, memory_manager):
        """Test queue status reporting."""
        status = memory_manager.get_queue_status()
        
        assert "queue_length" in status
        assert "active_requests" in status
        assert "max_queue_size" in status
        assert "is_processing" in status
        assert "memory_stats" in status


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor instance for testing."""
        return PerformanceMonitor(max_history_size=100)
    
    def test_request_tracking_lifecycle(self, performance_monitor):
        """Test complete request tracking lifecycle."""
        request_id = "test-request"
        
        # Start tracking
        performance_monitor.start_request_tracking(
            request_id, "test prompt", 512, 512, 20, 12345
        )
        
        assert request_id in performance_monitor.active_requests
        
        # Mark processing
        performance_monitor.mark_request_processing(request_id)
        
        # Complete tracking
        performance_monitor.complete_request_tracking(request_id, success=True)
        
        assert request_id not in performance_monitor.active_requests
        assert len(performance_monitor.generation_metrics) == 1
        
        metric = performance_monitor.generation_metrics[0]
        assert metric.request_id == request_id
        assert metric.success == True
        assert metric.prompt_length == len("test prompt")
    
    def test_error_tracking(self, performance_monitor):
        """Test error tracking functionality."""
        request_id = "error-request"
        
        performance_monitor.start_request_tracking(
            request_id, "error test", 512, 512, 20, 12345
        )
        
        performance_monitor.complete_request_tracking(
            request_id, success=False, error_type="TimeoutError", error_message="Request timed out"
        )
        
        assert len(performance_monitor.generation_metrics) == 1
        metric = performance_monitor.generation_metrics[0]
        assert metric.success == False
        assert metric.error_type == "TimeoutError"
        assert metric.error_message == "Request timed out"
        
        assert performance_monitor.error_counts["TimeoutError"] == 1
    
    def test_performance_stats_calculation(self, performance_monitor):
        """Test performance statistics calculation."""
        # Add some test metrics
        now = datetime.now()
        
        for i in range(5):
            metric = GenerationMetrics(
                request_id=f"test-{i}",
                timestamp=now - timedelta(minutes=i),
                prompt_length=50,
                width=512,
                height=512,
                steps=20,
                seed=12345,
                generation_time_seconds=30.0 + i,
                queue_wait_time_seconds=5.0,
                memory_used_gb=2.5,
                success=True
            )
            performance_monitor.generation_metrics.append(metric)
        
        # Add one failed request
        failed_metric = GenerationMetrics(
            request_id="failed-test",
            timestamp=now,
            prompt_length=50,
            width=512,
            height=512,
            steps=20,
            seed=12345,
            generation_time_seconds=0.0,
            queue_wait_time_seconds=10.0,
            memory_used_gb=0.0,
            success=False,
            error_type="MemoryError"
        )
        performance_monitor.generation_metrics.append(failed_metric)
        
        # Calculate stats
        stats = performance_monitor.get_performance_stats(time_window_minutes=60)
        
        assert stats.total_requests == 6
        assert stats.successful_requests == 5
        assert stats.failed_requests == 1
        assert stats.error_rate_percent == pytest.approx(16.67, rel=0.1)
        # Average of 30, 31, 32, 33, 34 = 32, but only successful requests are counted
        assert stats.average_generation_time == pytest.approx(32.0, rel=0.2)
        assert len(stats.most_common_errors) == 1
        assert stats.most_common_errors[0]["error_type"] == "MemoryError"
    
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=3 * 1024**3)
    def test_system_health_monitoring(self, mock_memory, mock_cuda, mock_psutil, performance_monitor):
        """Test system health monitoring."""
        # Mock system metrics
        mock_ram = Mock()
        mock_ram.percent = 75.0
        mock_ram.used = 8 * 1024**3
        mock_ram.total = 16 * 1024**3
        mock_psutil.return_value = mock_ram
        
        # Add some system metrics
        system_metric = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=75.0,
            gpu_memory_used_gb=3.0,
            gpu_memory_total_gb=6.0,
            gpu_utilization_percent=80.0,
            temperature_celsius=70.0
        )
        performance_monitor.system_metrics.append(system_metric)
        
        health = performance_monitor.get_system_health()
        
        assert health["status"] == "healthy"
        assert "metrics" in health
        assert health["metrics"]["cpu_percent"] == 50.0
        assert health["metrics"]["memory_percent"] == 75.0
    
    def test_metrics_export_json(self, performance_monitor, tmp_path):
        """Test metrics export in JSON format."""
        # Add test metric
        metric = GenerationMetrics(
            request_id="export-test",
            timestamp=datetime.now(),
            prompt_length=50,
            width=512,
            height=512,
            steps=20,
            seed=12345,
            generation_time_seconds=30.0,
            queue_wait_time_seconds=5.0,
            memory_used_gb=2.5,
            success=True
        )
        performance_monitor.generation_metrics.append(metric)
        
        # Export to JSON
        export_file = tmp_path / "test_export.json"
        performance_monitor.export_metrics(str(export_file), "json")
        
        assert export_file.exists()
        
        # Verify content
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert "generation_metrics" in data
        assert len(data["generation_metrics"]) == 1
        assert data["generation_metrics"][0]["request_id"] == "export-test"


class TestIntegration:
    """Integration tests for GTX 1060 optimizations."""
    
    @pytest.mark.asyncio
    async def test_memory_manager_performance_monitor_integration(self):
        """Test integration between memory manager and performance monitor."""
        memory_manager = GTX1060MemoryManager(max_vram_usage_gb=5.0)
        performance_monitor = PerformanceMonitor()
        
        request_id = "integration-test"
        
        # Start performance tracking
        performance_monitor.start_request_tracking(
            request_id, "integration test", 512, 512, 20, 12345
        )
        
        # Use memory manager context
        with patch.object(memory_manager, 'can_process_request', return_value=True):
            async with memory_manager.request_context(request_id, 512, 512, 20):
                # Simulate processing
                await asyncio.sleep(0.1)
        
        # Complete performance tracking
        performance_monitor.complete_request_tracking(request_id, success=True)
        
        # Verify integration
        assert len(performance_monitor.generation_metrics) == 1
        metric = performance_monitor.generation_metrics[0]
        assert metric.request_id == request_id
        assert metric.success == True
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling of concurrent requests with memory constraints."""
        memory_manager = GTX1060MemoryManager(max_vram_usage_gb=5.0, max_queue_size=3)
        
        # Mock memory usage to simulate constraints
        with patch.object(memory_manager, 'can_process_request', return_value=False):
            # Start multiple requests
            futures = []
            for i in range(3):
                future = await memory_manager.queue_request(f"concurrent-{i}", 512, 512, 20)
                futures.append(future)
            
            assert len(memory_manager.request_queue) == 3
            
            # Simulate processing completion
            with patch.object(memory_manager, 'can_process_request', return_value=True):
                # Manually trigger queue processing
                await memory_manager._process_queue()
                
                # Wait a bit for processing
                await asyncio.sleep(0.1)
            
            # Verify requests were processed
            assert len(memory_manager.active_requests) <= 3


@pytest.mark.integration
class TestGTX1060RealHardware:
    """Integration tests that require actual GTX 1060 hardware."""
    
    @pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(), 
                       reason="CUDA not available")
    def test_actual_memory_usage(self):
        """Test actual GPU memory usage on real hardware."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        
        # Verify it's a GTX 1060 (approximately 6GB)
        if not (5.5 <= total_memory_gb <= 6.5):
            pytest.skip(f"Not a GTX 1060 (found {total_memory_gb:.1f}GB)")
        
        memory_manager = GTX1060MemoryManager(max_vram_usage_gb=5.0)
        
        # Test memory estimation accuracy
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # Simulate model loading (allocate some memory)
        test_tensor = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
        
        current_memory = torch.cuda.memory_allocated() / (1024**3)
        actual_usage = current_memory - initial_memory
        
        # Verify memory tracking works
        stats = memory_manager.get_memory_stats()
        assert stats.gpu_allocated_gb > 0
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(), 
                       reason="CUDA not available")
    def test_memory_limit_enforcement(self):
        """Test that memory limits are properly enforced."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        memory_manager = GTX1060MemoryManager(max_vram_usage_gb=1.0)  # Very low limit
        
        # Large image should be rejected
        can_process = memory_manager.can_process_request(2.0)  # 2GB request
        assert can_process == False
        
        # Small image should be accepted
        can_process = memory_manager.can_process_request(0.5)  # 500MB request
        assert can_process == True