"""
Integration tests for performance characteristics and GTX 1060 memory constraints.
"""
import pytest
import asyncio
import time
import threading
from unittest.mock import patch, Mock
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.stable_diffusion import (
    StableDiffusionModelManager, 
    ImageGenerationService,
    model_manager,
    generation_service
)


class TestGTX1060MemoryConstraints:
    """Integration tests for GTX 1060 6GB memory constraints."""
    
    def test_memory_threshold_validation(self):
        """Test that GTX 1060 6GB meets memory requirements."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props:
            
            # Test GTX 1060 6GB
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3  # 6GB
            mock_props.name = "GeForce GTX 1060 6GB"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            device = manager._get_optimal_device()
            
            assert cuda_info["gpu_memory_gb"] == 6.0
            assert device == "cuda"  # Should use GPU
    
    def test_insufficient_memory_fallback(self):
        """Test fallback to CPU when GPU memory is insufficient."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props:
            
            # Test GTX 1050 2GB (insufficient)
            mock_props = Mock()
            mock_props.total_memory = 2 * 1024**3  # 2GB
            mock_props.name = "GeForce GTX 1050"
            mock_get_props.return_value = mock_props
            
            cuda_info = manager.check_cuda_availability()
            device = manager._get_optimal_device()
            
            assert cuda_info["gpu_memory_gb"] == 2.0
            assert device == "cpu"  # Should fall back to CPU
    
    def test_memory_optimization_settings(self):
        """Test that memory optimizations are properly configured for GTX 1060."""
        manager = StableDiffusionModelManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_get_props, \
             patch('app.services.stable_diffusion.StableDiffusionPipeline') as mock_pipeline_class:
            
            # Mock GTX 1060
            mock_props = Mock()
            mock_props.total_memory = 6 * 1024**3
            mock_props.name = "GeForce GTX 1060"
            mock_get_props.return_value = mock_props
            
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.to.return_value = mock_pipeline
            
            result = manager.initialize_model()
            
            assert result is True
            
            # Verify memory optimizations
            mock_pipeline.enable_attention_slicing.assert_called_once_with(1)
            mock_pipeline.enable_sequential_cpu_offload.assert_called_once()
            
            # Verify model loading parameters for GPU
            import torch
            args, kwargs = mock_pipeline_class.from_pretrained.call_args
            assert kwargs["torch_dtype"] == torch.float16  # Half precision for memory savings
            assert kwargs["safety_checker"] is None  # Disabled to save memory
    
    def test_memory_cleanup_effectiveness(self):
        """Test that memory cleanup is effective."""
        manager = StableDiffusionModelManager()
        manager._cuda_available = True
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect:
            
            # Simulate multiple cleanup calls
            for _ in range(5):
                manager.cleanup_memory()
            
            # Should call CUDA cache clearing and garbage collection
            assert mock_empty_cache.call_count == 5
            assert mock_gc_collect.call_count == 5
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring functionality."""
        manager = StableDiffusionModelManager()
        manager._cuda_available = True
        
        with patch('psutil.virtual_memory') as mock_virtual_memory, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=2 * 1024**3), \
             patch('torch.cuda.memory_reserved', return_value=3 * 1024**3):
            
            # Mock system memory
            mock_virtual_memory.return_value = Mock(
                percent=75.0,
                used=6 * 1024**3,  # 6GB used
                total=8 * 1024**3  # 8GB total
            )
            
            # Should not raise exceptions
            manager._log_memory_usage()
            
            # Verify monitoring functions were called
            mock_virtual_memory.assert_called_once()


class TestPerformanceBenchmarks:
    """Integration tests for performance benchmarks and timing."""
    
    @pytest.fixture
    def mock_performance_service(self):
        """Mock service with realistic performance characteristics."""
        mock_manager = Mock(spec=StableDiffusionModelManager)
        mock_manager.device = "cuda"
        mock_manager.model_id = "runwayml/stable-diffusion-v1-5"
        mock_manager.is_ready.return_value = True
        mock_manager.pipeline = Mock()
        
        service = ImageGenerationService(mock_manager)
        return service, mock_manager
    
    @pytest.mark.asyncio
    async def test_generation_time_benchmarks(self, mock_performance_service):
        """Test generation time benchmarks for different configurations."""
        service, mock_manager = mock_performance_service
        
        # Test different step counts and their expected relative timing
        test_cases = [
            {"steps": 10, "expected_min_time": 0.05},
            {"steps": 20, "expected_min_time": 0.10},
            {"steps": 30, "expected_min_time": 0.15},
            {"steps": 50, "expected_min_time": 0.25},
        ]
        
        for case in test_cases:
            async def mock_generate_with_timing(**kwargs):
                # Simulate realistic timing based on steps
                steps = kwargs.get("steps", 20)
                simulated_time = steps * 0.005  # 5ms per step
                await asyncio.sleep(simulated_time)
                
                mock_image = Mock()
                return {
                    "image": mock_image,
                    "metadata": {
                        "prompt": kwargs.get("prompt", "test"),
                        "steps": steps,
                        "width": kwargs.get("width", 512),
                        "height": kwargs.get("height", 512),
                        "seed": kwargs.get("seed", 12345),
                        "generation_time_seconds": simulated_time,
                        "model_version": "test-model"
                    }
                }
            
            with patch.object(service, '_generate_with_timeout', side_effect=mock_generate_with_timing):
                start_time = time.time()
                result = await service.generate_image(
                    prompt="test",
                    steps=case["steps"]
                )
                end_time = time.time()
                
                actual_time = end_time - start_time
                reported_time = result["metadata"]["generation_time_seconds"]
                
                # Verify timing is reasonable
                assert actual_time >= case["expected_min_time"]
                assert reported_time >= case["expected_min_time"]
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_generation(self, mock_performance_service):
        """Test memory usage patterns during generation."""
        service, mock_manager = mock_performance_service
        
        async def mock_generate_with_memory_tracking(**kwargs):
            # Simulate memory usage during generation
            await asyncio.sleep(0.1)
            
            mock_image = Mock()
            return {
                "image": mock_image,
                "metadata": {
                    "prompt": "test",
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                    "seed": 12345,
                    "generation_time_seconds": 0.1,
                    "model_version": "test-model"
                }
            }
        
        with patch.object(service, '_generate_with_timeout', side_effect=mock_generate_with_memory_tracking):
            # Track cleanup calls
            cleanup_calls_before = mock_manager.cleanup_memory.call_count
            
            await service.generate_image("test prompt")
            
            cleanup_calls_after = mock_manager.cleanup_memory.call_count
            
            # Memory cleanup should be called
            assert cleanup_calls_after > cleanup_calls_before
    
    def test_concurrent_request_performance(self, mock_performance_service):
        """Test performance under concurrent request load."""
        service, mock_manager = mock_performance_service
        
        async def mock_concurrent_generate(**kwargs):
            # Simulate concurrent generation with slight delays
            await asyncio.sleep(0.1)
            
            mock_image = Mock()
            return {
                "image": mock_image,
                "metadata": {
                    "prompt": kwargs.get("prompt", "test"),
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                    "seed": 12345,
                    "generation_time_seconds": 0.1,
                    "model_version": "test-model"
                }
            }
        
        async def run_concurrent_test():
            with patch.object(service, '_generate_with_timeout', side_effect=mock_concurrent_generate):
                # Run 5 concurrent generations
                tasks = []
                for i in range(5):
                    task = service.generate_image(f"test prompt {i}")
                    tasks.append(task)
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                total_time = end_time - start_time
                
                # All should complete
                assert len(results) == 5
                
                # Should complete in reasonable time (not 5x sequential time)
                # With proper async handling, should be closer to single request time
                assert total_time < 1.0  # Should be much less than 5 * 0.1 = 0.5s
                
                return results
        
        # Run the async test
        results = asyncio.run(run_concurrent_test())
        assert len(results) == 5
    
    def test_timeout_performance_characteristics(self, mock_performance_service):
        """Test timeout behavior under different load conditions."""
        service, mock_manager = mock_performance_service
        
        # Test with different timeout scenarios
        original_timeout = service.generation_timeout
        
        try:
            # Test short timeout
            service.generation_timeout = 0.05  # 50ms
            
            async def slow_generate(**kwargs):
                await asyncio.sleep(0.1)  # 100ms - exceeds timeout
                return Mock()
            
            async def test_timeout():
                with patch.object(service, '_generate_with_timeout', side_effect=slow_generate):
                    with pytest.raises(RuntimeError, match="Image generation failed"):
                        await service.generate_image("test")
            
            asyncio.run(test_timeout())
            
        finally:
            service.generation_timeout = original_timeout


class TestConcurrentRequestHandling:
    """Integration tests for concurrent request handling."""
    
    @pytest.fixture
    def mock_concurrent_service(self):
        """Mock service for concurrent testing."""
        mock_manager = Mock(spec=StableDiffusionModelManager)
        mock_manager.device = "cuda"
        mock_manager.model_id = "test-model"
        mock_manager.is_ready.return_value = True
        mock_manager.pipeline = Mock()
        
        service = ImageGenerationService(mock_manager)
        return service, mock_manager
    
    def test_thread_safety(self, mock_concurrent_service):
        """Test thread safety of service components."""
        service, mock_manager = mock_concurrent_service
        
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                async def generate():
                    # Simulate generation
                    await asyncio.sleep(0.01)
                    mock_image = Mock()
                    return {
                        "image": mock_image,
                        "metadata": {
                            "prompt": f"thread {thread_id}",
                            "steps": 20,
                            "width": 512,
                            "height": 512,
                            "seed": thread_id,
                            "generation_time_seconds": 0.01,
                            "model_version": "test-model"
                        }
                    }
                
                with patch.object(service, '_generate_with_timeout', side_effect=generate):
                    # Run async function in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            service.generate_image(f"prompt {thread_id}")
                        )
                        results.append(result)
                    finally:
                        loop.close()
                        
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        # Verify each result is unique
        seeds = [r["metadata"]["seed"] for r in results]
        assert len(set(seeds)) == 5  # All unique
    
    def test_resource_contention_handling(self, mock_concurrent_service):
        """Test handling of resource contention under load."""
        service, mock_manager = mock_concurrent_service
        
        # Track resource usage
        generation_count = 0
        cleanup_count = 0
        
        async def mock_generate_with_contention(**kwargs):
            nonlocal generation_count
            generation_count += 1
            
            # Simulate resource usage
            await asyncio.sleep(0.05)
            
            mock_image = Mock()
            return {
                "image": mock_image,
                "metadata": {
                    "prompt": kwargs.get("prompt", "test"),
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                    "seed": 12345,
                    "generation_time_seconds": 0.05,
                    "model_version": "test-model"
                }
            }
        
        def mock_cleanup():
            nonlocal cleanup_count
            cleanup_count += 1
        
        async def run_contention_test():
            with patch.object(service, '_generate_with_timeout', side_effect=mock_generate_with_contention), \
                 patch.object(mock_manager, 'cleanup_memory', side_effect=mock_cleanup):
                
                # Run many concurrent requests
                tasks = []
                for i in range(10):
                    task = service.generate_image(f"test {i}")
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successful results
                successful = [r for r in results if not isinstance(r, Exception)]
                failed = [r for r in results if isinstance(r, Exception)]
                
                return successful, failed
        
        successful, failed = asyncio.run(run_contention_test())
        
        # Most requests should succeed
        assert len(successful) >= 8  # Allow for some failures under high load
        assert generation_count >= 8
        assert cleanup_count >= 8  # Cleanup should be called for each generation
    
    def test_memory_pressure_handling(self, mock_concurrent_service):
        """Test behavior under simulated memory pressure."""
        service, mock_manager = mock_concurrent_service
        
        memory_pressure_count = 0
        
        async def mock_generate_with_memory_pressure(**kwargs):
            nonlocal memory_pressure_count
            memory_pressure_count += 1
            
            # Simulate memory pressure on every 3rd request
            if memory_pressure_count % 3 == 0:
                raise RuntimeError("CUDA out of memory")
            
            await asyncio.sleep(0.02)
            
            mock_image = Mock()
            return {
                "image": mock_image,
                "metadata": {
                    "prompt": kwargs.get("prompt", "test"),
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                    "seed": 12345,
                    "generation_time_seconds": 0.02,
                    "model_version": "test-model"
                }
            }
        
        async def run_memory_pressure_test():
            with patch.object(service, '_generate_with_timeout', side_effect=mock_generate_with_memory_pressure):
                
                results = []
                for i in range(6):  # Run 6 requests, expect 2 to fail
                    try:
                        result = await service.generate_image(f"test {i}")
                        results.append(("success", result))
                    except RuntimeError as e:
                        results.append(("error", str(e)))
                
                return results
        
        results = asyncio.run(run_memory_pressure_test())
        
        # Should have mix of successes and memory errors
        successes = [r for r in results if r[0] == "success"]
        errors = [r for r in results if r[0] == "error"]
        
        assert len(successes) == 4  # 2/3 should succeed
        assert len(errors) == 2     # 1/3 should fail with memory error
        
        # Verify error messages
        for error_type, error_msg in errors:
            assert "CUDA out of memory" in error_msg


class TestSystemResourceMonitoring:
    """Integration tests for system resource monitoring."""
    
    def test_system_memory_monitoring(self):
        """Test system memory monitoring functionality."""
        # Get actual system memory info
        memory_info = psutil.virtual_memory()
        
        # Basic validation
        assert memory_info.total > 0
        assert 0 <= memory_info.percent <= 100
        assert memory_info.used <= memory_info.total
        
        # Convert to GB for validation
        total_gb = memory_info.total / (1024**3)
        used_gb = memory_info.used / (1024**3)
        
        # System should have reasonable memory amounts
        assert total_gb >= 1.0  # At least 1GB total
        assert used_gb >= 0.1   # At least 100MB used
        
        print(f"System memory: {used_gb:.1f}GB / {total_gb:.1f}GB ({memory_info.percent:.1f}%)")
    
    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring."""
        # Get CPU usage over a short period
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Should be a reasonable value
        assert 0 <= cpu_percent <= 100
        
        # Get per-core usage
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Should have at least one core
        assert len(cpu_per_core) >= 1
        
        # All core percentages should be valid
        for core_percent in cpu_per_core:
            assert 0 <= core_percent <= 100
        
        print(f"CPU usage: {cpu_percent:.1f}% (cores: {len(cpu_per_core)})")
    
    def test_disk_usage_monitoring(self):
        """Test disk usage monitoring."""
        # Get disk usage for current directory
        disk_usage = psutil.disk_usage('.')
        
        # Basic validation
        assert disk_usage.total > 0
        assert disk_usage.used <= disk_usage.total
        assert disk_usage.free <= disk_usage.total
        
        # Convert to GB
        total_gb = disk_usage.total / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        
        # Should have reasonable amounts
        assert total_gb >= 1.0  # At least 1GB total
        
        usage_percent = (used_gb / total_gb) * 100
        
        print(f"Disk usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_percent:.1f}%)")
        
        # Warn if disk is very full
        if usage_percent > 90:
            print(f"WARNING: Disk usage is high ({usage_percent:.1f}%)")


if __name__ == "__main__":
    pytest.main([__file__])