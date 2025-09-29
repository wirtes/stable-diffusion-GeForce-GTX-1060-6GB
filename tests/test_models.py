"""
Unit tests for Pydantic models and validation logic.
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    GenerationMetadata,
    ErrorResponse,
    HealthResponse
)


class TestImageGenerationRequest:
    """Test cases for ImageGenerationRequest model."""
    
    def test_valid_request_with_defaults(self):
        """Test valid request with default values."""
        request = ImageGenerationRequest(prompt="A beautiful sunset")
        assert request.prompt == "A beautiful sunset"
        assert request.steps == 20
        assert request.width == 512
        assert request.height == 512
        assert request.seed is None
    
    def test_valid_request_with_all_params(self):
        """Test valid request with all parameters specified."""
        request = ImageGenerationRequest(
            prompt="A cat in a hat",
            steps=30,
            width=768,
            height=512,
            seed=12345
        )
        assert request.prompt == "A cat in a hat"
        assert request.steps == 30
        assert request.width == 768
        assert request.height == 512
        assert request.seed == 12345
    
    def test_prompt_validation_empty(self):
        """Test that empty prompt raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="")
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_prompt_validation_whitespace_only(self):
        """Test that whitespace-only prompt raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="   ")
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_prompt_validation_too_long(self):
        """Test that overly long prompt raises validation error."""
        long_prompt = "A" * 501
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt=long_prompt)
        assert "at most 500 characters" in str(exc_info.value)
    
    def test_prompt_sanitization(self):
        """Test that prompt is properly sanitized."""
        request = ImageGenerationRequest(prompt="  A   beautiful    sunset  ")
        assert request.prompt == "A beautiful sunset"
    
    def test_steps_validation_too_low(self):
        """Test that steps below 1 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", steps=0)
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_steps_validation_too_high(self):
        """Test that steps above 50 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", steps=51)
        assert "less than or equal to 50" in str(exc_info.value)
    
    def test_dimensions_validation_too_small(self):
        """Test that dimensions below 256 raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", width=128)
        assert "greater than or equal to 256" in str(exc_info.value)
    
    def test_dimensions_validation_too_large(self):
        """Test that dimensions above 768 raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", height=1024)
        assert "less than or equal to 768" in str(exc_info.value)
    
    def test_dimensions_validation_not_multiple_of_64(self):
        """Test that dimensions not multiples of 64 raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", width=300)
        assert "Dimensions must be multiples of 64" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", height=300)
        assert "Dimensions must be multiples of 64" in str(exc_info.value)
    
    def test_dimensions_validation_valid_multiples(self):
        """Test that valid multiples of 64 are accepted."""
        valid_dimensions = [256, 320, 384, 448, 512, 576, 640, 704, 768]
        for dim in valid_dimensions:
            request = ImageGenerationRequest(prompt="test", width=dim, height=dim)
            assert request.width == dim
            assert request.height == dim
    
    def test_seed_validation_negative(self):
        """Test that negative seed raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", seed=-1)
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_seed_validation_too_large(self):
        """Test that seed above 2^32-1 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerationRequest(prompt="test", seed=2**32)
        assert "less than or equal to" in str(exc_info.value)


class TestGenerationMetadata:
    """Test cases for GenerationMetadata model."""
    
    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = GenerationMetadata(
            prompt="A beautiful sunset",
            steps=20,
            width=512,
            height=512,
            seed=12345,
            generation_time_seconds=2.5,
            model_version="stable-diffusion-v1-5"
        )
        assert metadata.prompt == "A beautiful sunset"
        assert metadata.steps == 20
        assert metadata.width == 512
        assert metadata.height == 512
        assert metadata.seed == 12345
        assert metadata.generation_time_seconds == 2.5
        assert metadata.model_version == "stable-diffusion-v1-5"


class TestImageGenerationResponse:
    """Test cases for ImageGenerationResponse model."""
    
    def test_valid_response(self):
        """Test valid response creation."""
        metadata = GenerationMetadata(
            prompt="A cat",
            steps=20,
            width=512,
            height=512,
            seed=12345,
            generation_time_seconds=2.5,
            model_version="stable-diffusion-v1-5"
        )
        response = ImageGenerationResponse(
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            metadata=metadata
        )
        assert response.image_base64.startswith("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ")
        assert response.metadata.prompt == "A cat"


class TestErrorResponse:
    """Test cases for ErrorResponse model."""
    
    def test_error_response_with_details(self):
        """Test error response with details."""
        error = ErrorResponse(
            error="Validation failed",
            details="Prompt cannot be empty",
            error_code="VALIDATION_ERROR"
        )
        assert error.error == "Validation failed"
        assert error.details == "Prompt cannot be empty"
        assert error.error_code == "VALIDATION_ERROR"
    
    def test_error_response_without_details(self):
        """Test error response without details."""
        error = ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR"
        )
        assert error.error == "Internal server error"
        assert error.details is None
        assert error.error_code == "INTERNAL_ERROR"


class TestHealthResponse:
    """Test cases for HealthResponse model."""
    
    def test_health_response_with_memory(self):
        """Test health response with memory information."""
        health = HealthResponse(
            status="healthy",
            gpu_available=True,
            model_loaded=True,
            memory_usage={"gpu_memory_used": "2.1GB", "gpu_memory_total": "6GB"}
        )
        assert health.status == "healthy"
        assert health.gpu_available is True
        assert health.model_loaded is True
        assert health.memory_usage["gpu_memory_used"] == "2.1GB"
    
    def test_health_response_without_memory(self):
        """Test health response without memory information."""
        health = HealthResponse(
            status="unhealthy",
            gpu_available=False,
            model_loaded=False
        )
        assert health.status == "unhealthy"
        assert health.gpu_available is False
        assert health.model_loaded is False
        assert health.memory_usage is None