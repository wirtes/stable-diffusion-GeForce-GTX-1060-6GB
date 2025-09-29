#!/usr/bin/env python3
"""
Manual test script to verify model validation logic.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    GenerationMetadata,
    ErrorResponse,
    HealthResponse
)
from pydantic import ValidationError

def test_valid_request():
    """Test valid request with default values."""
    try:
        request = ImageGenerationRequest(prompt="A beautiful sunset")
        print("✓ Valid request with defaults passed")
        assert request.prompt == "A beautiful sunset"
        assert request.steps == 20
        assert request.width == 512
        assert request.height == 512
        assert request.seed is None
        print("✓ All default values correct")
    except Exception as e:
        print(f"✗ Valid request test failed: {e}")
        return False
    return True

def test_prompt_validation():
    """Test prompt validation."""
    # Test empty prompt
    try:
        ImageGenerationRequest(prompt="")
        print("✗ Empty prompt should have failed")
        return False
    except ValidationError as e:
        print("✓ Empty prompt validation passed")
    
    # Test whitespace-only prompt
    try:
        ImageGenerationRequest(prompt="   ")
        print("✗ Whitespace-only prompt should have failed")
        return False
    except ValidationError as e:
        print("✓ Whitespace-only prompt validation passed")
    
    # Test prompt sanitization
    try:
        request = ImageGenerationRequest(prompt="  A   beautiful    sunset  ")
        if request.prompt == "A beautiful sunset":
            print("✓ Prompt sanitization passed")
        else:
            print(f"✗ Prompt sanitization failed: got '{request.prompt}'")
            return False
    except Exception as e:
        print(f"✗ Prompt sanitization test failed: {e}")
        return False
    
    return True

def test_dimension_validation():
    """Test dimension validation."""
    # Test valid multiples of 64
    valid_dims = [256, 320, 384, 448, 512, 576, 640, 704, 768]
    for dim in valid_dims:
        try:
            request = ImageGenerationRequest(prompt="test", width=dim, height=dim)
            print(f"✓ Dimension {dim} passed")
        except Exception as e:
            print(f"✗ Valid dimension {dim} failed: {e}")
            return False
    
    # Test invalid dimension (not multiple of 64)
    try:
        ImageGenerationRequest(prompt="test", width=300)
        print("✗ Invalid dimension should have failed")
        return False
    except ValidationError as e:
        print("✓ Invalid dimension validation passed")
    
    return True

def test_steps_validation():
    """Test steps validation."""
    # Test valid steps
    try:
        request = ImageGenerationRequest(prompt="test", steps=25)
        print("✓ Valid steps passed")
    except Exception as e:
        print(f"✗ Valid steps failed: {e}")
        return False
    
    # Test invalid steps (too low)
    try:
        ImageGenerationRequest(prompt="test", steps=0)
        print("✗ Invalid steps (too low) should have failed")
        return False
    except ValidationError as e:
        print("✓ Invalid steps (too low) validation passed")
    
    # Test invalid steps (too high)
    try:
        ImageGenerationRequest(prompt="test", steps=51)
        print("✗ Invalid steps (too high) should have failed")
        return False
    except ValidationError as e:
        print("✓ Invalid steps (too high) validation passed")
    
    return True

def test_response_models():
    """Test response models."""
    try:
        # Test metadata
        metadata = GenerationMetadata(
            prompt="A cat",
            steps=20,
            width=512,
            height=512,
            seed=12345,
            generation_time_seconds=2.5,
            model_version="stable-diffusion-v1-5"
        )
        print("✓ GenerationMetadata creation passed")
        
        # Test response
        response = ImageGenerationResponse(
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            metadata=metadata
        )
        print("✓ ImageGenerationResponse creation passed")
        
        # Test error response
        error = ErrorResponse(
            error="Test error",
            details="Test details",
            error_code="TEST_ERROR"
        )
        print("✓ ErrorResponse creation passed")
        
        # Test health response
        health = HealthResponse(
            status="healthy",
            gpu_available=True,
            model_loaded=True
        )
        print("✓ HealthResponse creation passed")
        
    except Exception as e:
        print(f"✗ Response models test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running manual model validation tests...\n")
    
    tests = [
        test_valid_request,
        test_prompt_validation,
        test_dimension_validation,
        test_steps_validation,
        test_response_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n--- Running {test.__name__} ---")
        if test():
            passed += 1
            print(f"✓ {test.__name__} PASSED")
        else:
            print(f"✗ {test.__name__} FAILED")
    
    print(f"\n--- Results ---")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)