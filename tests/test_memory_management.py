"""
Unit tests for memory management utilities and GPU optimization.
"""
import pytest
import torch
import gc
from unittest.mock import Mock, patch, MagicMock
from app.services.stable_diffusion import StableDiffusionModelManager


class TestMemoryManagement:
    """Test cases for memory management utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StableDiffusionModelManager()
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_cleanup_memory_with_cuda(self, mock_gc_collect, mock_empty_cache, mock_cuda_available):
        """Test memory cleanup when CUDA is available."""
        mock_cuda_available.return_value = True
        self.manager._cuda_available = True
        
        self.manager.cleanup_memory()
        
        mock_empty_cache.assert_called_once()
        mock_gc_collect.assert_called_once()
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_cleanup_memory_without_cuda(self, mock_gc_collect, mock_empty_cache, mock_cuda_available):
        """Test memory cleanup when CUDA is not available."""
        mock_cuda_available.return_value = False
        self.manager._cuda_available = False
        
        self.manager.cleanup_memory()
        
        # Should not call CUDA functions or gc.collect when CUDA not available
        mock_empty_cache.assert_not_called()
        mock_gc_collect.assert_not_called()
        
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.is_available')
    def test_log_memory_usage_with_cuda(self, mock_cuda_available, mock_memory_reserved, mock_memory_allocated, mock_virtual_memory):
        """Test memory usage logging with CUDA available."""
        mock_cuda_available.return_value = True
        self.manager._cuda_available = True
        
        # Mock memory values
        mock_virtual_memory.return_value = Mock(
            percent=75.5,
            used=8 * 1024**3,  # 8GB
            total=16 * 1024**3  # 16GB
        )
        mock_memory_allocated.return_value = 2 * 1024**3  # 2GB
        mock_memory_reserved.return_value = 3 * 1024**3   # 3GB
        
        # Should not raise any exceptions
        self.manager._log_memory_usage()
        
        mock_virtual_memory.assert_called_once()
        mock_memory_allocated.assert_called_once()
        mock_memory_reserved.assert_called_once()
        
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_log_memory_usage_without_cuda(self, mock_cuda_available, mock_virtual_memory):
        """Test memory usage logging without CUDA."""
        mock_cuda_available.return_value = False
        self.manager._cuda_available = False
        
        mock_virtual_memory.return_value = Mock(
            percent=50.0,
            used=4 * 1024**3,  # 4GB
            total=8 * 1024**3   # 8GB
        )
        
        # Should not raise any exceptions
        self.manager._log_memory_usage()
        
        mock_virtual_memory.assert_called_once()
        
    def test_apply_gpu_optimizations_attention_slicing(self):
        """Test that attention slicing is properly configured."""
        mock_pipeline = Mock()
        self.manager.pipeline = mock_pipeline
        
        self.manager._apply_gpu_optimizations()
        
        mock_pipeline.enable_attention_slicing.assert_called_once_with(1)
        
    def test_apply_gpu_optimizations_cpu_offloading(self):
        """Test that CPU offloading is properly configured."""
        mock_pipeline = Mock()
        self.manager.pipeline = mock_pipeline
        
        self.manager._apply_gpu_optimizations()
        
        mock_pipeline.enable_sequential_cpu_offload.assert_called_once()
        
    def test_apply_gpu_optimizations_xformers_success(self):
        """Test successful xformers memory efficient attention setup."""
        mock_pipeline = Mock()
        self.manager.pipeline = mock_pipeline
        
        self.manager._apply_gpu_optimizations()
        
        mock_pipeline.enable_xformers_memory_efficient_attention.assert_called_once()
        
    def test_apply_gpu_optimizations_xformers_failure(self):
        """Test graceful handling of xformers setup failure."""
        mock_pipeline = Mock()
        mock_pipeline.enable_xformers_memory_efficient_attention.side_effect = Exception("xformers not available")
        self.manager.pipeline = mock_pipeline
        
        # Should not raise exception
        self.manager._apply_gpu_optimizations()
        
        mock_pipeline.enable_xformers_memory_efficient_attention.assert_called_once()
        
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_sufficient_memory(self, mock_cuda_available, mock_device_count, mock_get_props):
        """Test device selection with sufficient GPU memory."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock GTX 1060 with 6GB
        mock_props = Mock()
        mock_props.total_memory = 6 * 1024**3
        mock_props.name = "GeForce GTX 1060"
        mock_get_props.return_value = mock_props
        
        # Initialize CUDA info first
        self.manager.check_cuda_availability()
        
        device = self.manager._get_optimal_device()
        assert device == "cuda"
        
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_insufficient_memory(self, mock_cuda_available, mock_device_count, mock_get_props):
        """Test device selection with insufficient GPU memory."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock GPU with only 2GB
        mock_props = Mock()
        mock_props.total_memory = 2 * 1024**3
        mock_props.name = "GeForce GTX 1050"
        mock_get_props.return_value = mock_props
        
        # Initialize CUDA info first
        self.manager.check_cuda_availability()
        
        device = self.manager._get_optimal_device()
        assert device == "cpu"
        
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_no_cuda(self, mock_cuda_available):
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        # Initialize CUDA info first
        self.manager.check_cuda_availability()
        
        device = self.manager._get_optimal_device()
        assert device == "cpu"


class TestGPUMemoryConstraints:
    """Test GPU memory constraint handling for GTX 1060."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StableDiffusionModelManager()
        
    def test_gtx_1060_memory_threshold(self):
        """Test that GTX 1060 6GB meets minimum memory requirements."""
        self.manager._cuda_available = True
        self.manager._gpu_memory_gb = 6.0
        
        device = self.manager._get_optimal_device()
        assert device == "cuda"
        
    def test_gtx_1050_memory_threshold(self):
        """Test that GTX 1050 2GB falls back to CPU."""
        self.manager._cuda_available = True
        self.manager._gpu_memory_gb = 2.0
        
        device = self.manager._get_optimal_device()
        assert device == "cpu"
        
    def test_memory_threshold_boundary(self):
        """Test memory threshold boundary conditions."""
        self.manager._cuda_available = True
        
        # Exactly 4GB should use GPU
        self.manager._gpu_memory_gb = 4.0
        device = self.manager._get_optimal_device()
        assert device == "cuda"
        
        # Just under 4GB should use CPU
        self.manager._gpu_memory_gb = 3.9
        device = self.manager._get_optimal_device()
        assert device == "cpu"
        
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_memory_calculation_accuracy(self, mock_cuda_available, mock_device_count, mock_get_props):
        """Test accurate memory calculation from bytes to GB."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Test various memory sizes
        test_cases = [
            (6 * 1024**3, 6.0),      # 6GB exactly
            (6442450944, 6.0),       # 6GB in bytes (rounded)
            (8589934592, 8.0),       # 8GB exactly
            (4294967296, 4.0),       # 4GB exactly
        ]
        
        for memory_bytes, expected_gb in test_cases:
            mock_props = Mock()
            mock_props.total_memory = memory_bytes
            mock_props.name = "Test GPU"
            mock_get_props.return_value = mock_props
            
            cuda_info = self.manager.check_cuda_availability()
            assert cuda_info["gpu_memory_gb"] == expected_gb


class TestMemoryOptimizationSettings:
    """Test memory optimization settings for different scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StableDiffusionModelManager()
        
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_gpu_model_loading_parameters(self, mock_cuda_available, mock_device_count, mock_get_props, mock_pipeline_class):
        """Test that GPU model loading uses correct parameters."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock GTX 1060
        mock_props = Mock()
        mock_props.total_memory = 6 * 1024**3
        mock_props.name = "GeForce GTX 1060"
        mock_get_props.return_value = mock_props
        
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        result = self.manager.initialize_model()
        
        assert result is True
        
        # Verify model loading parameters
        args, kwargs = mock_pipeline_class.from_pretrained.call_args
        assert kwargs["torch_dtype"] == torch.float16
        assert kwargs["safety_checker"] is None
        assert kwargs["requires_safety_checker"] is False
        
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    @patch('torch.cuda.is_available')
    def test_cpu_model_loading_parameters(self, mock_cuda_available, mock_pipeline_class):
        """Test that CPU model loading uses correct parameters."""
        mock_cuda_available.return_value = False
        
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        result = self.manager.initialize_model()
        
        assert result is True
        
        # Verify model loading parameters for CPU
        args, kwargs = mock_pipeline_class.from_pretrained.call_args
        assert kwargs["torch_dtype"] == torch.float32
        assert kwargs["safety_checker"] is None
        
    def test_attention_slicing_configuration(self):
        """Test attention slicing configuration for memory efficiency."""
        mock_pipeline = Mock()
        self.manager.pipeline = mock_pipeline
        
        self.manager._apply_gpu_optimizations()
        
        # Should use slice size of 1 for maximum memory savings
        mock_pipeline.enable_attention_slicing.assert_called_once_with(1)


if __name__ == "__main__":
    pytest.main([__file__])