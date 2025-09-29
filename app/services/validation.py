"""
Request validation service for image generation API.
"""
import random
import re
from typing import Optional


class ValidationService:
    """Service for validating and sanitizing image generation requests."""
    
    # Valid dimension range and step
    MIN_DIMENSION = 256
    MAX_DIMENSION = 768
    DIMENSION_STEP = 64
    
    # Valid steps range
    MIN_STEPS = 1
    MAX_STEPS = 50
    
    # Prompt constraints
    MAX_PROMPT_LENGTH = 500
    
    # Seed constraints
    MIN_SEED = 0
    MAX_SEED = 2**32 - 1
    
    @staticmethod
    def validate_dimensions(width: int, height: int) -> tuple[bool, Optional[str]]:
        """
        Validate that dimensions are within valid range and multiples of 64.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check width
        if width < ValidationService.MIN_DIMENSION or width > ValidationService.MAX_DIMENSION:
            return False, f"Width must be between {ValidationService.MIN_DIMENSION} and {ValidationService.MAX_DIMENSION}"
        
        if width % ValidationService.DIMENSION_STEP != 0:
            return False, f"Width must be a multiple of {ValidationService.DIMENSION_STEP}"
        
        # Check height
        if height < ValidationService.MIN_DIMENSION or height > ValidationService.MAX_DIMENSION:
            return False, f"Height must be between {ValidationService.MIN_DIMENSION} and {ValidationService.MAX_DIMENSION}"
        
        if height % ValidationService.DIMENSION_STEP != 0:
            return False, f"Height must be a multiple of {ValidationService.DIMENSION_STEP}"
        
        return True, None
    
    @staticmethod
    def validate_steps(steps: int) -> tuple[bool, Optional[str]]:
        """
        Validate that steps are within valid range.
        
        Args:
            steps: Number of diffusion steps
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if steps < ValidationService.MIN_STEPS or steps > ValidationService.MAX_STEPS:
            return False, f"Steps must be between {ValidationService.MIN_STEPS} and {ValidationService.MAX_STEPS}"
        
        return True, None
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """
        Sanitize and clean the input prompt.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Sanitized prompt text
            
        Raises:
            ValueError: If prompt is empty after sanitization
        """
        # Remove potentially harmful characters but keep basic punctuation
        sanitized = re.sub(r'[^\w\s\-.,!?()[\]{}:;"\'`]', '', prompt)
        
        # Strip whitespace and normalize spaces
        sanitized = ' '.join(sanitized.strip().split())
        
        if not sanitized:
            raise ValueError("Prompt cannot be empty after sanitization")
        
        if len(sanitized) > ValidationService.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt must be at most {ValidationService.MAX_PROMPT_LENGTH} characters")
        
        return sanitized
    
    @staticmethod
    def validate_seed(seed: Optional[int]) -> tuple[bool, Optional[str]]:
        """
        Validate seed value if provided.
        
        Args:
            seed: Optional seed value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if seed is None:
            return True, None
        
        if seed < ValidationService.MIN_SEED or seed > ValidationService.MAX_SEED:
            return False, f"Seed must be between {ValidationService.MIN_SEED} and {ValidationService.MAX_SEED}"
        
        return True, None
    
    @staticmethod
    def generate_random_seed() -> int:
        """
        Generate a random seed value within valid range.
        
        Returns:
            Random seed integer
        """
        return random.randint(ValidationService.MIN_SEED, ValidationService.MAX_SEED)
    
    @staticmethod
    def get_valid_dimensions() -> list[int]:
        """
        Get list of all valid dimension values.
        
        Returns:
            List of valid dimension values
        """
        return list(range(
            ValidationService.MIN_DIMENSION,
            ValidationService.MAX_DIMENSION + 1,
            ValidationService.DIMENSION_STEP
        ))
    
    @classmethod
    def validate_request_params(cls, prompt: str, steps: int, width: int, height: int, seed: Optional[int] = None) -> dict:
        """
        Comprehensive validation of all request parameters.
        
        Args:
            prompt: Text prompt for generation
            steps: Number of diffusion steps
            width: Image width
            height: Image height
            seed: Optional seed value
            
        Returns:
            Dictionary with validation results and sanitized values
            
        Raises:
            ValueError: If any validation fails
        """
        errors = []
        
        # Validate and sanitize prompt
        try:
            sanitized_prompt = cls.sanitize_prompt(prompt)
        except ValueError as e:
            errors.append(str(e))
            sanitized_prompt = prompt
        
        # Validate steps
        steps_valid, steps_error = cls.validate_steps(steps)
        if not steps_valid:
            errors.append(steps_error)
        
        # Validate dimensions
        dims_valid, dims_error = cls.validate_dimensions(width, height)
        if not dims_valid:
            errors.append(dims_error)
        
        # Validate seed
        seed_valid, seed_error = cls.validate_seed(seed)
        if not seed_valid:
            errors.append(seed_error)
        
        # Generate random seed if not provided
        final_seed = seed if seed is not None else cls.generate_random_seed()
        
        if errors:
            raise ValueError("; ".join(errors))
        
        return {
            "prompt": sanitized_prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "seed": final_seed,
            "is_valid": True
        }