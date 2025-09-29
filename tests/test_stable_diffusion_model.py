"""
Unit tests for Stable Diffusion model initialization and management.
"""
import pytest
import torch
import asyncio
from unittest.mock import Mock, patch, MagicMock
from app.services.stable_diffusion import StableDiffusionModelManager, ImageGenerationService, ResponseFormattingService


class TestStableDiffusionModelManager:
    """Test cases for StableDiffusionModelManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StableDiffusionModelManager()
        
    def test_init_default_values(self):
        """Test manager initialization with default values."""
        assert self.manager.model_id == "runwayml/stable-diffusion-v1-5"
        assert self.manager.pipeline is None
        assert self.manager.device is None
        assert self.manager.is_initialized is False
        assert self.manager._cuda_available is False
        assert self.manager._gpu_memory_gb == 0
        
    def test_init_custom_model_id(self):
        """Test manager initialization with custom model ID."""
        custom_manager = StableDiffusionModelManager("custom/model-id")
        assert custom_manager.model_id == "custom/model-id"
        
    @patch('torch.cuda.is_available')
    def test_check_cuda_availability_no_cuda(self, mock_cuda_available):
        """Test CUDA availability check when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        result = self.manager.check_cuda_availability()
        
        expected = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "gpu_name": None
        }
        assert result == expected
        assert self.manager._cuda_available is False
        assert self.manager._gpu_memory_gb == 0
        
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count')
    @patch('torch.version.cuda', '11.8')
    @patch('torch.cuda.is_available')
    def test_check_cuda_availability_with_cuda(self, mock_cuda_available, mock_device_count, mock_get_props):
        """Test CUDA availability check when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock GPU properties for GTX 1060 (6GB)
        mock_props = Mock()
        mock_props.total_memory = 6 * 1024**3  # 6GB in bytes
        mock_props.name = "GeForce GTX 1060"
        mock_get_props.return_value = mock_props
        
        result = self.manager.check_cuda_availability()
        
        expected = {
            "cuda_available": True,
            "cuda_version": "11.8",
            "gpu_count": 1,
            "gpu_memory_gb": 6.0,
            "gpu_name": "GeForce GTX 1060"
        }
        assert result == expected
        assert self.manager._cuda_available is True
        assert self.manager._gpu_memory_gb == 6.0
        
    def test_get_optimal_device_no_cuda(self):
        """Test device selection when CUDA is not available."""
        self.manager._cuda_available = False
        self.manager._gpu_memory_gb = 0
        
        device = self.manager._get_optimal_device()
        assert device == "cpu"
        
    def test_get_optimal_device_insufficient_memory(self):
        """Test device selection when GPU has insufficient memory."""
        self.manager._cuda_available = True
        self.manager._gpu_memory_gb = 2.0  # Less than 4GB minimum
        
        device = self.manager._get_optimal_device()
        assert device == "cpu"
        
    def test_get_optimal_device_sufficient_memory(self):
        """Test device selection when GPU has sufficient memory."""
        self.manager._cuda_available = True
        self.manager._gpu_memory_gb = 6.0  # GTX 1060 memory
        
        device = self.manager._get_optimal_device()
        assert device == "cuda"
        
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    @patch('torch.cuda.is_available')
    def test_initialize_model_success_cpu(self, mock_cuda_available, mock_pipeline_class):
        """Test successful model initialization on CPU."""
        mock_cuda_available.return_value = False
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        result = self.manager.initialize_model()
        
        assert result is True
        assert self.manager.is_initialized is True
        assert self.manager.device == "cpu"
        assert self.manager.pipeline == mock_pipeline
        
        # Verify model was loaded with correct parameters
        mock_pipeline_class.from_pretrained.assert_called_once()
        args, kwargs = mock_pipeline_class.from_pretrained.call_args
        assert args[0] == "runwayml/stable-diffusion-v1-5"
        assert kwargs["torch_dtype"] == torch.float32
        assert kwargs["safety_checker"] is None
        
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count')
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    @patch('torch.cuda.is_available')
    def test_initialize_model_success_gpu(self, mock_cuda_available, mock_pipeline_class, mock_device_count, mock_get_props):
        """Test successful model initialization on GPU."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock GPU properties for GTX 1060 (6GB)
        mock_props = Mock()
        mock_props.total_memory = 6 * 1024**3  # 6GB in bytes
        mock_props.name = "GeForce GTX 1060"
        mock_get_props.return_value = mock_props
        
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        result = self.manager.initialize_model()
        
        assert result is True
        assert self.manager.is_initialized is True
        assert self.manager.device == "cuda"
        
        # Verify GPU optimizations were applied
        mock_pipeline.enable_attention_slicing.assert_called_once_with(1)
        mock_pipeline.enable_sequential_cpu_offload.assert_called_once()
        
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    def test_initialize_model_failure(self, mock_pipeline_class):
        """Test model initialization failure handling."""
        mock_pipeline_class.from_pretrained.side_effect = Exception("Model loading failed")
        
        result = self.manager.initialize_model()
        
        assert result is False
        assert self.manager.is_initialized is False
        assert self.manager.pipeline is None
        
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.is_available')
    def test_cleanup_memory_with_cuda(self, mock_cuda_available, mock_empty_cache):
        """Test memory cleanup when CUDA is available."""
        mock_cuda_available.return_value = True
        self.manager._cuda_available = True
        
        self.manager.cleanup_memory()
        
        mock_empty_cache.assert_called_once()
        
    @patch('torch.cuda.is_available')
    def test_cleanup_memory_without_cuda(self, mock_cuda_available):
        """Test memory cleanup when CUDA is not available."""
        mock_cuda_available.return_value = False
        self.manager._cuda_available = False
        
        # Should not raise any exceptions
        self.manager.cleanup_memory()
        
    def test_get_model_info(self):
        """Test getting model information."""
        self.manager.device = "cuda"
        self.manager.is_initialized = True
        self.manager._cuda_available = True
        self.manager._gpu_memory_gb = 6.0
        
        info = self.manager.get_model_info()
        
        expected = {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "device": "cuda",
            "is_initialized": True,
            "cuda_available": True,
            "gpu_memory_gb": 6.0,
        }
        assert info == expected
        
    def test_is_ready_not_initialized(self):
        """Test is_ready when model is not initialized."""
        assert self.manager.is_ready() is False
        
    def test_is_ready_no_pipeline(self):
        """Test is_ready when pipeline is None."""
        self.manager.is_initialized = True
        self.manager.pipeline = None
        
        assert self.manager.is_ready() is False
        
    def test_is_ready_success(self):
        """Test is_ready when model is properly initialized."""
        self.manager.is_initialized = True
        self.manager.pipeline = Mock()
        
        assert self.manager.is_ready() is True


class TestGlobalModelManager:
    """Test the global model manager instance."""
    
    def test_global_instance_exists(self):
        """Test that global model manager instance is created."""
        from app.services.stable_diffusion import model_manager
        
        assert model_manager is not None
        assert isinstance(model_manager, StableDiffusionModelManager)
        assert model_manager.model_id == "runwayml/stable-diffusion-v1-5"


class TestImageGenerationService:
    """Test cases for ImageGenerationService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_manager = Mock(spec=StableDiffusionModelManager)
        # Configure mock attributes that are accessed by the service
        self.model_manager.device = "cuda"
        self.model_manager.model_id = "test-model"
        self.model_manager.pipeline = Mock()
        self.service = ImageGenerationService(self.model_manager)
        
    def test_init(self):
        """Test service initialization."""
        assert self.service.model_manager == self.model_manager
        assert self.service.generation_timeout == 60
        
    @pytest.mark.asyncio
    async def test_generate_image_model_not_ready(self):
        """Test image generation when model is not ready."""
        self.model_manager.is_ready.return_value = False
        
        with pytest.raises(RuntimeError, match="Model is not initialized or ready"):
            await self.service.generate_image("test prompt")
            
    @pytest.mark.asyncio
    @patch('torch.randint')
    @patch('torch.Generator')
    @patch('time.time')
    async def test_generate_image_success(self, mock_time, mock_generator_class, mock_randint):
        """Test successful image generation."""
        # Setup mocks
        self.model_manager.is_ready.return_value = True
        self.model_manager.device = "cuda"
        self.model_manager.model_id = "test-model"
        
        mock_randint.return_value.item.return_value = 12345
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Mock time progression
        mock_time.side_effect = [0.0, 2.5]  # Start and end times
        
        # Mock the pipeline execution
        mock_image = Mock()
        mock_image.size = (512, 512)
        
        with patch.object(self.service, '_generate_with_timeout', return_value=mock_image) as mock_generate:
            result = await self.service.generate_image(
                prompt="test prompt",
                steps=25,
                width=512,
                height=512
            )
            
        # Verify result structure
        assert "image" in result
        assert "metadata" in result
        assert result["image"] == mock_image
        
        metadata = result["metadata"]
        assert metadata["prompt"] == "test prompt"
        assert metadata["steps"] == 25
        assert metadata["width"] == 512
        assert metadata["height"] == 512
        assert metadata["seed"] == 12345
        assert metadata["generation_time_seconds"] == 2.5
        assert metadata["model_version"] == "test-model"
        
        # Verify pipeline was called correctly
        mock_generate.assert_called_once_with(
            prompt="test prompt",
            num_inference_steps=25,
            width=512,
            height=512,
            generator=mock_generator
        )
        
        # Verify cleanup was called
        self.model_manager.cleanup_memory.assert_called_once()
        
    @pytest.mark.asyncio
    @patch('time.time')
    @patch('torch.randint')
    async def test_generate_image_with_custom_seed(self, mock_randint, mock_time):
        """Test image generation with custom seed."""
        self.model_manager.is_ready.return_value = True
        self.model_manager.device = "cpu"
        
        # Mock time progression
        mock_time.side_effect = [0.0, 1.0]
        
        mock_image = Mock()
        
        with patch.object(self.service, '_generate_with_timeout', return_value=mock_image):
            with patch('torch.Generator') as mock_generator_class:
                result = await self.service.generate_image(
                    prompt="test",
                    seed=98765
                )
                    
        # Should not call randint when seed is provided
        mock_randint.assert_not_called()
        assert result["metadata"]["seed"] == 98765
        
    @pytest.mark.asyncio
    @patch('time.time')
    async def test_generate_image_pipeline_failure(self, mock_time):
        """Test handling of pipeline failures."""
        self.model_manager.is_ready.return_value = True
        
        # Mock time for the start_time call
        mock_time.return_value = 0.0
        
        with patch.object(self.service, '_generate_with_timeout', side_effect=Exception("Pipeline error")):
            with patch('torch.Generator'):
                with pytest.raises(RuntimeError, match="Image generation failed: Pipeline error"):
                    await self.service.generate_image("test prompt")
                
        # Verify cleanup was called even on error
        self.model_manager.cleanup_memory.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_generate_with_timeout_success(self):
        """Test successful generation with timeout."""
        mock_image = Mock()
        
        with patch.object(self.service, '_run_pipeline', return_value=mock_image):
            result = await self.service._generate_with_timeout(prompt="test")
            
        assert result == mock_image
        
    @pytest.mark.asyncio
    async def test_generate_with_timeout_timeout_error(self):
        """Test timeout handling in generation."""
        # Mock a slow pipeline that exceeds timeout
        def slow_pipeline(kwargs):
            import time
            time.sleep(0.1)  # Simulate slow generation
            return Mock()
            
        self.service.generation_timeout = 0.05  # Very short timeout for testing
        
        with patch.object(self.service, '_run_pipeline', side_effect=slow_pipeline):
            with pytest.raises(TimeoutError, match="Image generation timed out"):
                await self.service._generate_with_timeout(prompt="test")
                
    def test_run_pipeline_success(self):
        """Test successful pipeline execution."""
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        
        self.model_manager.pipeline.return_value = mock_result
        
        result = self.service._run_pipeline({"prompt": "test"})
        
        assert result == mock_image
        self.model_manager.pipeline.assert_called_once_with(prompt="test")
        
    def test_run_pipeline_failure(self):
        """Test pipeline execution failure."""
        self.model_manager.pipeline.side_effect = Exception("Pipeline failed")
        
        with pytest.raises(Exception, match="Pipeline failed"):
            self.service._run_pipeline({"prompt": "test"})
            
    def test_get_generation_status(self):
        """Test getting generation status."""
        self.model_manager.is_ready.return_value = True
        self.model_manager.get_model_info.return_value = {"model": "info"}
        
        status = self.service.get_generation_status()
        
        expected = {
            "model_ready": True,
            "model_info": {"model": "info"},
            "timeout_seconds": 60
        }
        assert status == expected


class TestGlobalInstances:
    """Test the global service instances."""
    
    def test_global_instances_exist(self):
        """Test that global instances are created."""
        from app.services.stable_diffusion import model_manager, generation_service
        
        assert model_manager is not None
        assert generation_service is not None
        assert isinstance(model_manager, StableDiffusionModelManager)
        assert isinstance(generation_service, ImageGenerationService)
        assert generation_service.model_manager == model_manager


class TestResponseFormattingService:
    """Test cases for ResponseFormattingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from app.services.stable_diffusion import ResponseFormattingService
        self.formatter = ResponseFormattingService()
        
    @patch('base64.b64encode')
    @patch('io.BytesIO')
    def test_image_to_base64_success(self, mock_bytesio, mock_b64encode):
        """Test successful image to base64 conversion."""
        # Setup mocks
        mock_image = Mock()
        mock_buffer = Mock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.getvalue.return_value = b'fake_image_bytes'
        mock_b64encode.return_value = b'fake_base64_string'
        
        result = self.formatter.image_to_base64(mock_image)
        
        assert result == "fake_base64_string"
        mock_image.save.assert_called_once_with(mock_buffer, format='PNG')
        mock_buffer.seek.assert_called_once_with(0)
        mock_b64encode.assert_called_once_with(b'fake_image_bytes')
        
    @patch('io.BytesIO')
    def test_image_to_base64_failure(self, mock_bytesio):
        """Test image to base64 conversion failure."""
        mock_image = Mock()
        mock_image.save.side_effect = Exception("Save failed")
        
        with pytest.raises(RuntimeError, match="Image conversion failed: Save failed"):
            self.formatter.image_to_base64(mock_image)
            
    def test_format_generation_response_success(self):
        """Test successful generation response formatting."""
        mock_image = Mock()
        
        generation_result = {
            "image": mock_image,
            "metadata": {
                "prompt": "test prompt",
                "steps": 25,
                "width": 512,
                "height": 512,
                "seed": 12345,
                "generation_time_seconds": 2.5,
                "model_version": "test-model"
            }
        }
        
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', return_value="fake_base64") as mock_convert:
            result = self.formatter.format_generation_response(generation_result)
            
        expected = {
            "image_base64": "fake_base64",
            "metadata": {
                "prompt": "test prompt",
                "steps": 25,
                "width": 512,
                "height": 512,
                "seed": 12345,
                "generation_time_seconds": 2.5,
                "model_version": "test-model"
            }
        }
        
        assert result == expected
        mock_convert.assert_called_once_with(mock_image)
        
    def test_format_generation_response_failure(self):
        """Test generation response formatting failure."""
        generation_result = {
            "image": Mock(),
            "metadata": {"prompt": "test"}
        }
        
        with patch('app.services.stable_diffusion.ResponseFormattingService.image_to_base64', side_effect=Exception("Conversion failed")):
            with pytest.raises(RuntimeError, match="Response formatting failed"):
                self.formatter.format_generation_response(generation_result)
                
    def test_format_error_response_timeout_error(self):
        """Test formatting timeout error response."""
        error = TimeoutError("Generation timed out")
        
        result = self.formatter.format_error_response(error)
        
        expected = {
            "error": "Generation timed out",
            "error_code": "TIMEOUT_ERROR",
            "details": "Image generation exceeded the maximum allowed time"
        }
        
        assert result == expected
        
    def test_format_error_response_model_not_ready(self):
        """Test formatting model not ready error response."""
        error = RuntimeError("Model is not initialized or ready")
        
        result = self.formatter.format_error_response(error)
        
        expected = {
            "error": "Model is not initialized or ready",
            "error_code": "MODEL_NOT_READY",
            "details": "The AI model is not initialized or ready for inference"
        }
        
        assert result == expected
        
    def test_format_error_response_memory_error(self):
        """Test formatting memory error response."""
        error = RuntimeError("Insufficient GPU memory available")
        
        result = self.formatter.format_error_response(error)
        
        expected = {
            "error": "Insufficient GPU memory available",
            "error_code": "MEMORY_ERROR",
            "details": "Insufficient GPU memory for image generation"
        }
        
        assert result == expected
        
    def test_format_error_response_validation_error(self):
        """Test formatting validation error response."""
        error = ValueError("Invalid parameter value")
        
        result = self.formatter.format_error_response(error)
        
        expected = {
            "error": "Invalid parameter value",
            "error_code": "VALIDATION_ERROR",
            "details": "Invalid parameters provided for image generation"
        }
        
        assert result == expected
        
    def test_format_error_response_generic_error(self):
        """Test formatting generic error response."""
        error = Exception("Something went wrong")
        
        result = self.formatter.format_error_response(error, "CUSTOM_ERROR")
        
        expected = {
            "error": "Something went wrong",
            "error_code": "CUSTOM_ERROR"
        }
        
        assert result == expected
        
    def test_get_response_headers_default(self):
        """Test getting default response headers."""
        headers = self.formatter.get_response_headers()
        
        expected = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        assert headers == expected
        
    def test_get_response_headers_custom_content_type(self):
        """Test getting response headers with custom content type."""
        headers = self.formatter.get_response_headers("image/png")
        
        assert headers["Content-Type"] == "image/png"
        assert "Cache-Control" in headers


class TestGlobalFormatterInstance:
    """Test the global response formatter instance."""
    
    def test_global_formatter_exists(self):
        """Test that global response formatter instance is created."""
        from app.services.stable_diffusion import response_formatter, ResponseFormattingService
        
        assert response_formatter is not None
        assert isinstance(response_formatter, ResponseFormattingService)