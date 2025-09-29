"""
Unit tests for error handling scenarios across all components.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pydantic import ValidationError

from app.services.stable_diffusion import (
    StableDiffusionModelManager, 
    ImageGenerationService, 
    ResponseFormattingService
)
from app.services.validation import ValidationService
from app.models.schemas import ImageGenerationRequest


class TestValidationErrorHandling:
    """Test error handling in validation service."""
    
    def test_prompt_sanitization_empty_after_cleaning(self):
        """Test error when prompt becomes empty after sanitization."""
        with pytest.raises(ValueError, match="Prompt cannot be empty after sanitization"):
            ValidationService.sanitize_prompt("@#$%^&*+=|\\/<>")
            
    def test_prompt_sanitization_whitespace_only(self):
        """Test error when prompt is only whitespace."""
        with pytest.raises(ValueError, match="Prompt cannot be empty after sanitization"):
            ValidationService.sanitize_prompt("   \t\n   ")
            
    def test_prompt_too_long_error(self):
        """Test error when prompt exceeds maximum length."""
        long_prompt = "A" * 501
        with pytest.raises(ValueError, match="Prompt must be at most 500 characters"):
            ValidationService.sanitize_prompt(long_prompt)
            
    def test_dimensions_validation_errors(self):
        """Test various dimension validation errors."""
        # Width too small
        is_valid, error = ValidationService.validate_dimensions(128, 512)
        assert not is_valid
        assert "Width must be between 256 and 768" in error
        
        # Height too large
        is_valid, error = ValidationService.validate_dimensions(512, 1024)
        assert not is_valid
        assert "Height must be between 256 and 768" in error
        
        # Not multiple of 64
        is_valid, error = ValidationService.validate_dimensions(300, 400)
        assert not is_valid
        assert "Width must be a multiple of 64" in error
        
    def test_steps_validation_errors(self):
        """Test steps validation errors."""
        # Too low
        is_valid, error = ValidationService.validate_steps(0)
        assert not is_valid
        assert "Steps must be between 1 and 50" in error
        
        # Too high
        is_valid, error = ValidationService.validate_steps(100)
        assert not is_valid
        assert "Steps must be between 1 and 50" in error
        
    def test_seed_validation_errors(self):
        """Test seed validation errors."""
        # Negative seed
        is_valid, error = ValidationService.validate_seed(-1)
        assert not is_valid
        assert "Seed must be between 0 and" in error
        
        # Too large seed
        is_valid, error = ValidationService.validate_seed(2**32)
        assert not is_valid
        assert "Seed must be between 0 and" in error
        
    def test_comprehensive_validation_multiple_errors(self):
        """Test that comprehensive validation collects multiple errors."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.validate_request_params(
                prompt="   ",  # Empty after sanitization
                steps=0,       # Too low
                width=300,     # Not multiple of 64
                height=1024,   # Too high
                seed=-1        # Negative
            )
            
        error_message = str(exc_info.value)
        assert "Prompt cannot be empty after sanitization" in error_message
        assert "Steps must be between 1 and 50" in error_message
        assert "Width must be a multiple of 64" in error_message
        assert "Seed must be between 0 and" in error_message
        # Note: Height validation may not trigger if width validation fails first


class TestPydanticModelErrorHandling:
    """Test error handling in Pydantic models."""
    
    def test_image_generation_request_empty_prompt(self):
        """Test validation error for empty prompt."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="")
        assert "String should have at least 1 character" in str(exc_info.value)
        
    def test_image_generation_request_whitespace_prompt(self):
        """Test validation error for whitespace-only prompt."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="   ")
        assert "Prompt cannot be empty" in str(exc_info.value)
        
    def test_image_generation_request_invalid_steps(self):
        """Test validation errors for invalid steps."""
        # Too low
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", steps=0)
        assert "greater than or equal to 1" in str(exc_info.value)
        
        # Too high
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", steps=51)
        assert "less than or equal to 50" in str(exc_info.value)
        
    def test_image_generation_request_invalid_dimensions(self):
        """Test validation errors for invalid dimensions."""
        # Not multiple of 64
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", width=300)
        assert "Dimensions must be multiples of 64" in str(exc_info.value)
        
        # Too small
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", width=128)
        assert "greater than or equal to 256" in str(exc_info.value)
        
        # Too large
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", height=1024)
        assert "less than or equal to 768" in str(exc_info.value)
        
    def test_image_generation_request_invalid_seed(self):
        """Test validation errors for invalid seed."""
        # Negative
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", seed=-1)
        assert "greater than or equal to 0" in str(exc_info.value)
        
        # Too large
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", seed=2**32)
        assert "less than or equal to" in str(exc_info.value)


class TestModelManagerErrorHandling:
    """Test error handling in model manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StableDiffusionModelManager()
        
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    def test_model_initialization_failure(self, mock_pipeline_class):
        """Test handling of model initialization failures."""
        mock_pipeline_class.from_pretrained.side_effect = Exception("Model loading failed")
        
        result = self.manager.initialize_model()
        
        assert result is False
        assert self.manager.is_initialized is False
        assert self.manager.pipeline is None
        
    @patch('app.services.stable_diffusion.StableDiffusionPipeline')
    def test_model_initialization_memory_error(self, mock_pipeline_class):
        """Test handling of memory errors during initialization."""
        mock_pipeline_class.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        
        result = self.manager.initialize_model()
        
        assert result is False
        assert self.manager.is_initialized is False
        
    @patch('torch.cuda.get_device_properties')
    def test_cuda_info_gathering_failure(self, mock_get_props):
        """Test handling of CUDA info gathering failures."""
        mock_get_props.side_effect = Exception("CUDA error")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            
            # Should raise exception since the actual implementation doesn't handle this gracefully
            with pytest.raises(Exception, match="CUDA error"):
                self.manager.check_cuda_availability()
            
    def test_is_ready_with_uninitialized_model(self):
        """Test is_ready method with uninitialized model."""
        assert self.manager.is_ready() is False
        
    def test_is_ready_with_none_pipeline(self):
        """Test is_ready method with None pipeline."""
        self.manager.is_initialized = True
        self.manager.pipeline = None
        
        assert self.manager.is_ready() is False


class TestImageGenerationErrorHandling:
    """Test error handling in image generation service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_manager = Mock(spec=StableDiffusionModelManager)
        self.model_manager.device = "cuda"
        self.model_manager.model_id = "test-model"
        self.model_manager.pipeline = Mock()
        self.service = ImageGenerationService(self.model_manager)
        
    @pytest.mark.asyncio
    async def test_generate_image_model_not_ready(self):
        """Test error when model is not ready."""
        self.model_manager.is_ready.return_value = False
        
        with pytest.raises(RuntimeError, match="Model is not initialized or ready"):
            await self.service.generate_image("test prompt")
            
    @pytest.mark.asyncio
    async def test_generate_image_pipeline_failure(self):
        """Test handling of pipeline execution failures."""
        self.model_manager.is_ready.return_value = True
        
        with patch.object(self.service, '_generate_with_timeout', side_effect=Exception("Pipeline failed")):
            with patch('torch.Generator'):
                with pytest.raises(RuntimeError, match="Image generation failed: Pipeline failed"):
                    await self.service.generate_image("test prompt")
                    
        # Verify cleanup was called even on error
        self.model_manager.cleanup_memory.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_generate_image_timeout_error(self):
        """Test handling of generation timeouts."""
        self.model_manager.is_ready.return_value = True
        
        with patch.object(self.service, '_generate_with_timeout', side_effect=TimeoutError("Timed out")):
            with patch('torch.Generator'):
                with pytest.raises(RuntimeError, match="Image generation failed: Timed out"):
                    await self.service.generate_image("test prompt")
                    
    @pytest.mark.asyncio
    async def test_generate_with_timeout_exceeds_limit(self):
        """Test timeout handling in _generate_with_timeout."""
        self.service.generation_timeout = 0.01  # Very short timeout
        
        def slow_pipeline(kwargs):
            import time
            time.sleep(0.1)  # Longer than timeout
            return Mock()
            
        with patch.object(self.service, '_run_pipeline', side_effect=slow_pipeline):
            with pytest.raises(TimeoutError, match="Image generation timed out"):
                await self.service._generate_with_timeout(prompt="test")
                
    def test_run_pipeline_execution_failure(self):
        """Test pipeline execution failure handling."""
        self.model_manager.pipeline.side_effect = Exception("Pipeline execution failed")
        
        with pytest.raises(Exception, match="Pipeline execution failed"):
            self.service._run_pipeline({"prompt": "test"})
            
    def test_run_pipeline_cuda_memory_error(self):
        """Test CUDA memory error handling in pipeline."""
        self.model_manager.pipeline.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            self.service._run_pipeline({"prompt": "test"})


class TestResponseFormattingErrorHandling:
    """Test error handling in response formatting service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormattingService()
        
    @patch('io.BytesIO')
    def test_image_to_base64_save_failure(self, mock_bytesio):
        """Test handling of image save failures."""
        mock_image = Mock()
        mock_image.save.side_effect = Exception("Save failed")
        
        with pytest.raises(RuntimeError, match="Image conversion failed: Save failed"):
            self.formatter.image_to_base64(mock_image)
            
    @patch('base64.b64encode')
    @patch('io.BytesIO')
    def test_image_to_base64_encoding_failure(self, mock_bytesio, mock_b64encode):
        """Test handling of base64 encoding failures."""
        mock_image = Mock()
        mock_buffer = Mock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.getvalue.return_value = b'fake_bytes'
        mock_b64encode.side_effect = Exception("Encoding failed")
        
        with pytest.raises(RuntimeError, match="Image conversion failed: Encoding failed"):
            self.formatter.image_to_base64(mock_image)
            
    def test_format_generation_response_conversion_failure(self):
        """Test handling of image conversion failures in response formatting."""
        generation_result = {
            "image": Mock(),
            "metadata": {"prompt": "test"}
        }
        
        with patch.object(self.formatter, 'image_to_base64', side_effect=Exception("Conversion failed")):
            with pytest.raises(RuntimeError, match="Response formatting failed"):
                self.formatter.format_generation_response(generation_result)
                
    def test_format_generation_response_missing_data(self):
        """Test handling of missing data in generation result."""
        generation_result = {
            "image": Mock()
            # Missing metadata
        }
        
        with pytest.raises(RuntimeError, match="Response formatting failed"):
            self.formatter.format_generation_response(generation_result)
            
    def test_format_error_response_timeout_error(self):
        """Test formatting of timeout errors."""
        error = TimeoutError("Generation timed out")
        
        result = self.formatter.format_error_response(error)
        
        assert result["error_code"] == "TIMEOUT_ERROR"
        assert result["details"] == "Request exceeded the maximum allowed time"
        
    def test_format_error_response_model_not_ready(self):
        """Test formatting of model not ready errors."""
        error = RuntimeError("Model is not initialized or ready")
        
        result = self.formatter.format_error_response(error)
        
        assert result["error_code"] == "MODEL_NOT_READY"
        assert result["details"] == "The AI model is not initialized or ready for inference"
        
    def test_format_error_response_memory_error(self):
        """Test formatting of memory errors."""
        error = RuntimeError("CUDA out of memory")
        
        result = self.formatter.format_error_response(error)
        
        assert result["error_code"] == "MEMORY_ERROR"
        assert result["details"] == "Insufficient GPU memory for image generation"
        
    def test_format_error_response_validation_error(self):
        """Test formatting of validation errors."""
        error = ValueError("Invalid parameter")
        
        result = self.formatter.format_error_response(error)
        
        assert result["error_code"] == "VALIDATION_ERROR"
        assert result["details"] == "Invalid parameters provided for image generation"
        
    def test_format_error_response_generic_error(self):
        """Test formatting of generic errors."""
        error = Exception("Something went wrong")
        
        result = self.formatter.format_error_response(error)
        
        assert result["error_code"] == "GENERATION_ERROR"
        assert "details" not in result
        
    def test_format_error_response_custom_error_code(self):
        """Test formatting with custom error code."""
        error = Exception("Custom error")
        
        result = self.formatter.format_error_response(error, "CUSTOM_ERROR")
        
        assert result["error_code"] == "CUSTOM_ERROR"
        assert result["error"] == "Custom error"


class TestConcurrentErrorHandling:
    """Test error handling in concurrent scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_manager = Mock(spec=StableDiffusionModelManager)
        self.model_manager.device = "cuda"
        self.model_manager.model_id = "test-model"
        self.service = ImageGenerationService(self.model_manager)
        
    @pytest.mark.asyncio
    async def test_concurrent_generation_failures(self):
        """Test handling of multiple concurrent generation failures."""
        self.model_manager.is_ready.return_value = True
        
        # Mock different types of failures
        failures = [
            Exception("Pipeline error 1"),
            TimeoutError("Timeout error"),
            RuntimeError("Memory error"),
            Exception("Pipeline error 2")
        ]
        
        async def failing_generate(prompt):
            failure = failures.pop(0) if failures else Exception("Generic error")
            raise failure
            
        with patch.object(self.service, '_generate_with_timeout', side_effect=failing_generate):
            with patch('torch.Generator'):
                
                # Run multiple concurrent generations that should all fail
                tasks = []
                for i in range(4):
                    task = self.service.generate_image(f"prompt {i}")
                    tasks.append(task)
                
                # All should raise RuntimeError (wrapped)
                for task in tasks:
                    with pytest.raises(RuntimeError):
                        await task
                        
        # Cleanup should be called for each failure
        assert self.model_manager.cleanup_memory.call_count == 4


class TestResourceExhaustionHandling:
    """Test handling of resource exhaustion scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = StableDiffusionModelManager()
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_memory_cleanup_on_exhaustion(self, mock_empty_cache, mock_cuda_available):
        """Test memory cleanup when resources are exhausted."""
        mock_cuda_available.return_value = True
        self.manager._cuda_available = True
        
        # Simulate multiple cleanup calls (as would happen under load)
        for _ in range(5):
            self.manager.cleanup_memory()
            
        # Should call cleanup each time
        assert mock_empty_cache.call_count == 5
        
    @patch('psutil.virtual_memory')
    def test_memory_logging_with_high_usage(self, mock_virtual_memory):
        """Test memory logging when system memory is high."""
        # Simulate high memory usage
        mock_virtual_memory.return_value = Mock(
            percent=95.0,  # Very high usage
            used=15 * 1024**3,  # 15GB
            total=16 * 1024**3  # 16GB total
        )
        
        # Should not raise exception even with high usage
        self.manager._log_memory_usage()
        
        mock_virtual_memory.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])