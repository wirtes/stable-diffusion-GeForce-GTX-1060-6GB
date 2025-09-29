#!/usr/bin/env python3
"""
Manual test script to validate Pydantic models.
"""
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from app.models.schemas import (
        ImageGenerationRequest,
        ImageGenerationResponse,
        GenerationMetadata,
        ErrorResponse,
        HealthResponse
    )
    print("✓ Successfully imported all models")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_image_generation_request():
    """Test ImageGenerationRequest validation."""
    print("\n--- Testing ImageGenerationRequest ---")
    
    # Test valid request with defaults
    try:
        request = ImageGenerationRequest(prompt="A beautiful sunset")
        print(f"✓ Valid request with defaults: {request.dict()}")
    except Exception as e:
        print(f"✗ Failed valid request: {e}")
        return False
    
    # Test valid request with all parameters
    try:
        request = ImageGenerationRequest(
            prompt="A cat in a hat",
            steps=30,
            width=768,
            height=512,
            seed=12345
        )
        print(f"✓ Valid request with all params: {request.dict()}")
    except Exception as e:
        print(f"✗ Failed valid request with all params: {e}")
        return False
    
    # Test prompt sanitization
    try:
        request = ImageGenerationRequest(prompt="  A   beautiful    sunset  ")
        expected = "A beautiful sunset"
        if request.prompt == expected:
            print(f"✓ Prompt sanitization works: '{request.prompt}'")
        else:
            print(f"✗ Prompt sanitization failed: expected '{expected}', got '{request.prompt}'")
            return False
    except Exception as e:
        print(f"✗ Prompt sanitization error: {e}")
        return False
    
    # Test empty prompt validation
    try:
        request = ImageGenerationRequest(prompt="")
        print("✗ Empty prompt should have failed")
        return False
    except Exception as e:
        print(f"✓ Empty prompt correctly rejected: {e}")
    
    # Test whitespace-only prompt validation
    try:
        request = ImageGenerationRequest(prompt="   ")
        print("✗ Whitespace-only prompt should have failed")
        return False
    except Exception as e:
        print(f"✓ Whitespace-only prompt correctly rejected: {e}")
    
    # Test dimension validation (not multiple of 64)
    try:
        request = ImageGenerationRequest(prompt="test", width=300)
        print("✗ Invalid dimension should have failed")
        return False
    except Exception as e:
        print(f"✓ Invalid dimension correctly rejected: {e}")
    
    # Test valid dimensions (multiples of 64)
    valid_dimensions = [256, 320, 384, 448, 512, 576, 640, 704, 768]
    for dim in valid_dimensions:
        try:
            request = ImageGenerationRequest(prompt="test", width=dim, height=dim)
            print(f"✓ Valid dimension {dim} accepted")
        except Exception as e:
            print(f"✗ Valid dimension {dim} rejected: {e}")
            return False
    
    # Test steps validation
    try:
        request = ImageGenerationRequest(prompt="test", steps=0)
        print("✗ Invalid steps (0) should have failed")
        return False
    except Exception as e:
        print(f"✓ Invalid steps (0) correctly rejected: {e}")
    
    try:
        request = ImageGenerationRequest(prompt="test", steps=51)
        print("✗ Invalid steps (51) should have failed")
        return False
    except Exception as e:
        print(f"✓ Invalid steps (51) correctly rejected: {e}")
    
    return True

def test_other_models():
    """Test other model classes."""
    print("\n--- Testing Other Models ---")
    
    # Test GenerationMetadata
    try:
        metadata = GenerationMetadata(
            prompt="A beautiful sunset",
            steps=20,
            width=512,
            height=512,
            seed=12345,
            generation_time_seconds=2.5,
            model_version="stable-diffusion-v1-5"
        )
        print(f"✓ GenerationMetadata created: {metadata.dict()}")
    except Exception as e:
        print(f"✗ GenerationMetadata failed: {e}")
        return False
    
    # Test ImageGenerationResponse
    try:
        response = ImageGenerationResponse(
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            metadata=metadata
        )
        print(f"✓ ImageGenerationResponse created")
    except Exception as e:
        print(f"✗ ImageGenerationResponse failed: {e}")
        return False
    
    # Test ErrorResponse
    try:
        error = ErrorResponse(
            error="Validation failed",
            details="Prompt cannot be empty",
            error_code="VALIDATION_ERROR"
        )
        print(f"✓ ErrorResponse created: {error.dict()}")
    except Exception as e:
        print(f"✗ ErrorResponse failed: {e}")
        return False
    
    # Test HealthResponse
    try:
        health = HealthResponse(
            status="healthy",
            gpu_available=True,
            model_loaded=True,
            memory_usage={"gpu_memory_used": "2.1GB", "gpu_memory_total": "6GB"}
        )
        print(f"✓ HealthResponse created: {health.dict()}")
    except Exception as e:
        print(f"✗ HealthResponse failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Running manual model validation tests...")
    
    success = True
    success &= test_image_generation_request()
    success &= test_other_models()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)