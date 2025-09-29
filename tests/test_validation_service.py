"""
Unit tests for ValidationService.
"""
import pytest
from app.services.validation import ValidationService


class TestValidationService:
    """Test cases for ValidationService."""
    
    def test_validate_dimensions_valid(self):
        """Test validation of valid dimensions."""
        valid_dimensions = [256, 320, 384, 448, 512, 576, 640, 704, 768]
        
        for dim in valid_dimensions:
            is_valid, error = ValidationService.validate_dimensions(dim, dim)
            assert is_valid is True
            assert error is None
    
    def test_validate_dimensions_invalid_range(self):
        """Test validation of dimensions outside valid range."""
        # Too small
        is_valid, error = ValidationService.validate_dimensions(128, 512)
        assert is_valid is False
        assert "Width must be between 256 and 768" in error
        
        is_valid, error = ValidationService.validate_dimensions(512, 128)
        assert is_valid is False
        assert "Height must be between 256 and 768" in error
        
        # Too large
        is_valid, error = ValidationService.validate_dimensions(1024, 512)
        assert is_valid is False
        assert "Width must be between 256 and 768" in error
        
        is_valid, error = ValidationService.validate_dimensions(512, 1024)
        assert is_valid is False
        assert "Height must be between 256 and 768" in error
    
    def test_validate_dimensions_not_multiple_of_64(self):
        """Test validation of dimensions not multiples of 64."""
        is_valid, error = ValidationService.validate_dimensions(300, 512)
        assert is_valid is False
        assert "Width must be a multiple of 64" in error
        
        is_valid, error = ValidationService.validate_dimensions(512, 300)
        assert is_valid is False
        assert "Height must be a multiple of 64" in error
    
    def test_validate_steps_valid(self):
        """Test validation of valid steps."""
        valid_steps = [1, 10, 20, 30, 40, 50]
        
        for steps in valid_steps:
            is_valid, error = ValidationService.validate_steps(steps)
            assert is_valid is True
            assert error is None
    
    def test_validate_steps_invalid(self):
        """Test validation of invalid steps."""
        # Too low
        is_valid, error = ValidationService.validate_steps(0)
        assert is_valid is False
        assert "Steps must be between 1 and 50" in error
        
        # Too high
        is_valid, error = ValidationService.validate_steps(51)
        assert is_valid is False
        assert "Steps must be between 1 and 50" in error
    
    def test_sanitize_prompt_valid(self):
        """Test prompt sanitization with valid inputs."""
        # Normal prompt
        result = ValidationService.sanitize_prompt("A beautiful sunset over mountains")
        assert result == "A beautiful sunset over mountains"
        
        # Prompt with extra whitespace
        result = ValidationService.sanitize_prompt("  A   beautiful    sunset  ")
        assert result == "A beautiful sunset"
        
        # Prompt with punctuation
        result = ValidationService.sanitize_prompt("A cat, sitting on a chair!")
        assert result == "A cat, sitting on a chair!"
    
    def test_sanitize_prompt_removes_harmful_chars(self):
        """Test that prompt sanitization removes potentially harmful characters."""
        result = ValidationService.sanitize_prompt("A cat <script>alert('xss')</script> on chair")
        assert result == "A cat scriptalert('xss')script on chair"
        
        result = ValidationService.sanitize_prompt("Test & symbols @ # $ % ^ * + = | \\ / < >")
        assert result == "Test symbols"
    
    def test_sanitize_prompt_empty_after_sanitization(self):
        """Test that empty prompt after sanitization raises error."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.sanitize_prompt("   ")
        assert "Prompt cannot be empty after sanitization" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            ValidationService.sanitize_prompt("@#$%^&*+=|\\/<>")
        assert "Prompt cannot be empty after sanitization" in str(exc_info.value)
    
    def test_sanitize_prompt_too_long(self):
        """Test that overly long prompt raises error."""
        long_prompt = "A" * 501
        with pytest.raises(ValueError) as exc_info:
            ValidationService.sanitize_prompt(long_prompt)
        assert "Prompt must be at most 500 characters" in str(exc_info.value)
    
    def test_validate_seed_valid(self):
        """Test validation of valid seeds."""
        # None seed (should be valid)
        is_valid, error = ValidationService.validate_seed(None)
        assert is_valid is True
        assert error is None
        
        # Valid seeds
        valid_seeds = [0, 12345, 1000000, 2**32 - 1]
        for seed in valid_seeds:
            is_valid, error = ValidationService.validate_seed(seed)
            assert is_valid is True
            assert error is None
    
    def test_validate_seed_invalid(self):
        """Test validation of invalid seeds."""
        # Negative seed
        is_valid, error = ValidationService.validate_seed(-1)
        assert is_valid is False
        assert "Seed must be between 0 and" in error
        
        # Too large seed
        is_valid, error = ValidationService.validate_seed(2**32)
        assert is_valid is False
        assert "Seed must be between 0 and" in error
    
    def test_generate_random_seed(self):
        """Test random seed generation."""
        # Generate multiple seeds and check they're in valid range
        for _ in range(10):
            seed = ValidationService.generate_random_seed()
            assert ValidationService.MIN_SEED <= seed <= ValidationService.MAX_SEED
        
        # Check that seeds are different (very high probability)
        seeds = [ValidationService.generate_random_seed() for _ in range(10)]
        assert len(set(seeds)) > 1  # Should have at least 2 different values
    
    def test_get_valid_dimensions(self):
        """Test getting list of valid dimensions."""
        valid_dims = ValidationService.get_valid_dimensions()
        expected = [256, 320, 384, 448, 512, 576, 640, 704, 768]
        assert valid_dims == expected
    
    def test_validate_request_params_valid(self):
        """Test comprehensive validation with valid parameters."""
        result = ValidationService.validate_request_params(
            prompt="A beautiful sunset",
            steps=20,
            width=512,
            height=512,
            seed=12345
        )
        
        assert result["prompt"] == "A beautiful sunset"
        assert result["steps"] == 20
        assert result["width"] == 512
        assert result["height"] == 512
        assert result["seed"] == 12345
        assert result["is_valid"] is True
    
    def test_validate_request_params_with_sanitization(self):
        """Test comprehensive validation with prompt sanitization."""
        result = ValidationService.validate_request_params(
            prompt="  A   beautiful    sunset  ",
            steps=20,
            width=512,
            height=512,
            seed=None
        )
        
        assert result["prompt"] == "A beautiful sunset"
        assert result["steps"] == 20
        assert result["width"] == 512
        assert result["height"] == 512
        assert isinstance(result["seed"], int)
        assert ValidationService.MIN_SEED <= result["seed"] <= ValidationService.MAX_SEED
        assert result["is_valid"] is True
    
    def test_validate_request_params_invalid_prompt(self):
        """Test comprehensive validation with invalid prompt."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.validate_request_params(
                prompt="   ",
                steps=20,
                width=512,
                height=512
            )
        assert "Prompt cannot be empty after sanitization" in str(exc_info.value)
    
    def test_validate_request_params_invalid_steps(self):
        """Test comprehensive validation with invalid steps."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.validate_request_params(
                prompt="A beautiful sunset",
                steps=0,
                width=512,
                height=512
            )
        assert "Steps must be between 1 and 50" in str(exc_info.value)
    
    def test_validate_request_params_invalid_dimensions(self):
        """Test comprehensive validation with invalid dimensions."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.validate_request_params(
                prompt="A beautiful sunset",
                steps=20,
                width=300,
                height=512
            )
        assert "Width must be a multiple of 64" in str(exc_info.value)
    
    def test_validate_request_params_invalid_seed(self):
        """Test comprehensive validation with invalid seed."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.validate_request_params(
                prompt="A beautiful sunset",
                steps=20,
                width=512,
                height=512,
                seed=-1
            )
        assert "Seed must be between 0 and" in str(exc_info.value)
    
    def test_validate_request_params_multiple_errors(self):
        """Test comprehensive validation with multiple errors."""
        with pytest.raises(ValueError) as exc_info:
            ValidationService.validate_request_params(
                prompt="   ",
                steps=0,
                width=300,
                height=512,
                seed=-1
            )
        
        error_message = str(exc_info.value)
        assert "Prompt cannot be empty after sanitization" in error_message
        assert "Steps must be between 1 and 50" in error_message
        assert "Width must be a multiple of 64" in error_message
        assert "Seed must be between 0 and" in error_message