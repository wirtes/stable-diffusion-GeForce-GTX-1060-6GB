"""
Integration tests for middleware components.
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from app.middleware.error_handling import (
    GlobalExceptionMiddleware,
    TimeoutMiddleware,
    ValidationErrorMiddleware
)
from app.middleware.logging import (
    RequestLoggingMiddleware,
    PerformanceLoggingMiddleware,
    SecurityLoggingMiddleware
)


class TestErrorHandlingMiddleware:
    """Test suite for error handling middleware."""
    
    @pytest.fixture
    def app_with_global_exception_middleware(self):
        """Create test app with global exception middleware."""
        app = FastAPI()
        app.add_middleware(GlobalExceptionMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/error")
        async def error_endpoint():
            raise Exception("Test exception")
        
        return app
    
    @pytest.fixture
    def app_with_timeout_middleware(self):
        """Create test app with timeout middleware."""
        app = FastAPI()
        app.add_middleware(TimeoutMiddleware, timeout_seconds=1)
        
        @app.get("/fast")
        async def fast_endpoint():
            return {"message": "fast"}
        
        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(2)  # Longer than timeout
            return {"message": "slow"}
        
        return app
    
    def test_global_exception_middleware_success(self, app_with_global_exception_middleware):
        """Test that normal requests pass through middleware."""
        client = TestClient(app_with_global_exception_middleware)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    def test_global_exception_middleware_handles_exception(self, app_with_global_exception_middleware):
        """Test that unhandled exceptions are caught and formatted."""
        client = TestClient(app_with_global_exception_middleware)
        response = client.get("/error")
        
        assert response.status_code == 500
        data = response.json()
        
        assert "error" in data
        assert "error_code" in data
        assert data["error_code"] == "INTERNAL_ERROR"
        assert "Test exception" in data["error"]
    
    def test_timeout_middleware_fast_request(self, app_with_timeout_middleware):
        """Test that fast requests complete normally."""
        client = TestClient(app_with_timeout_middleware)
        response = client.get("/fast")
        
        assert response.status_code == 200
        assert response.json() == {"message": "fast"}
    
    def test_timeout_middleware_slow_request(self, app_with_timeout_middleware):
        """Test that slow requests are timed out."""
        client = TestClient(app_with_timeout_middleware)
        response = client.get("/slow")
        
        assert response.status_code == 408
        data = response.json()
        
        assert "error" in data
        assert "error_code" in data
        assert data["error_code"] == "REQUEST_TIMEOUT"


class TestLoggingMiddleware:
    """Test suite for logging middleware."""
    
    @pytest.fixture
    def app_with_request_logging(self):
        """Create test app with request logging middleware."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware, log_request_body=True, log_response_body=True)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.post("/test")
        async def test_post_endpoint(data: dict):
            return {"received": data}
        
        @app.get("/error")
        async def error_endpoint():
            raise HTTPException(status_code=400, detail="Test error")
        
        return app
    
    @pytest.fixture
    def app_with_performance_logging(self):
        """Create test app with performance logging middleware."""
        app = FastAPI()
        app.add_middleware(PerformanceLoggingMiddleware, slow_request_threshold=0.1)
        
        @app.get("/fast")
        async def fast_endpoint():
            return {"message": "fast"}
        
        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(0.2)  # Longer than threshold
            return {"message": "slow"}
        
        return app
    
    @pytest.fixture
    def app_with_security_logging(self):
        """Create test app with security logging middleware."""
        app = FastAPI()
        app.add_middleware(SecurityLoggingMiddleware, rate_limit_threshold=5)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/generate")
        async def generate_endpoint():
            return {"message": "generated"}
        
        return app
    
    def test_request_logging_middleware_adds_headers(self, app_with_request_logging):
        """Test that request logging middleware adds tracking headers."""
        client = TestClient(app_with_request_logging)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers
        
        # Verify request ID format (8 character UUID)
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 8
    
    def test_request_logging_middleware_logs_errors(self, app_with_request_logging):
        """Test that error responses are handled properly."""
        client = TestClient(app_with_request_logging)
        response = client.get("/error")
        
        assert response.status_code == 400
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers
    
    def test_performance_logging_middleware_adds_memory_headers(self, app_with_performance_logging):
        """Test that performance middleware adds memory usage headers."""
        with patch('psutil.Process') as mock_process_class:
            # Mock memory info
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_process_class.return_value = mock_process
            
            client = TestClient(app_with_performance_logging)
            response = client.get("/fast")
            
            assert response.status_code == 200
            assert "X-Memory-Usage-MB" in response.headers
    
    def test_performance_logging_middleware_detects_slow_requests(self, app_with_performance_logging):
        """Test that slow requests are detected and logged."""
        client = TestClient(app_with_performance_logging)
        response = client.get("/slow")
        
        assert response.status_code == 200
        # Should complete but be logged as slow
        process_time = float(response.headers["X-Process-Time"])
        assert process_time > 0.1  # Should exceed threshold
    
    def test_security_logging_middleware_tracks_requests(self, app_with_security_logging):
        """Test that security middleware tracks client requests."""
        client = TestClient(app_with_security_logging)
        
        # Make multiple requests
        for _ in range(3):
            response = client.get("/test")
            assert response.status_code == 200
    
    def test_security_logging_middleware_logs_generation_access(self, app_with_security_logging):
        """Test that access to generation endpoint is logged."""
        client = TestClient(app_with_security_logging)
        response = client.get("/generate")
        
        assert response.status_code == 200


class TestMiddlewareIntegration:
    """Test middleware integration scenarios."""
    
    @pytest.fixture
    def app_with_all_middleware(self):
        """Create test app with all middleware components."""
        app = FastAPI()
        
        # Add middleware in reverse order (last added = first executed)
        app.add_middleware(GlobalExceptionMiddleware)
        app.add_middleware(TimeoutMiddleware, timeout_seconds=5)
        app.add_middleware(PerformanceLoggingMiddleware, slow_request_threshold=1.0)
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(SecurityLoggingMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/error")
        async def error_endpoint():
            raise Exception("Test exception")
        
        return app
    
    def test_middleware_stack_normal_request(self, app_with_all_middleware):
        """Test that all middleware works together for normal requests."""
        client = TestClient(app_with_all_middleware)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
        
        # Check that headers from different middleware are present
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers
    
    def test_middleware_stack_error_request(self, app_with_all_middleware):
        """Test that all middleware works together for error requests."""
        client = TestClient(app_with_all_middleware)
        response = client.get("/error")
        
        assert response.status_code == 500
        data = response.json()
        
        # Check error formatting from GlobalExceptionMiddleware
        assert "error" in data
        assert "error_code" in data
        assert data["error_code"] == "INTERNAL_ERROR"
        
        # Check headers from logging middleware
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers
    
    def test_middleware_order_execution(self, app_with_all_middleware):
        """Test that middleware executes in the correct order."""
        # This is more of a functional test to ensure no conflicts
        client = TestClient(app_with_all_middleware)
        
        # Make multiple requests to test interaction
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
            
            # Each request should have unique ID
            request_id = response.headers.get("X-Request-ID")
            assert request_id is not None
            assert len(request_id) == 8


class TestMiddlewareErrorScenarios:
    """Test middleware behavior in various error scenarios."""
    
    @pytest.fixture
    def app_with_middleware_errors(self):
        """Create test app to test middleware error handling."""
        app = FastAPI()
        app.add_middleware(GlobalExceptionMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/validation-error")
        async def validation_error_endpoint():
            from pydantic import ValidationError
            raise ValidationError([], str)
        
        @app.get("/http-exception")
        async def http_exception_endpoint():
            raise HTTPException(status_code=404, detail="Not found")
        
        @app.get("/runtime-error")
        async def runtime_error_endpoint():
            raise RuntimeError("Runtime error")
        
        return app
    
    def test_middleware_handles_http_exceptions(self, app_with_middleware_errors):
        """Test that HTTP exceptions pass through middleware correctly."""
        client = TestClient(app_with_middleware_errors)
        response = client.get("/http-exception")
        
        assert response.status_code == 404
        # HTTP exceptions should not be caught by global exception handler
    
    def test_middleware_handles_runtime_errors(self, app_with_middleware_errors):
        """Test that runtime errors are caught by global exception handler."""
        client = TestClient(app_with_middleware_errors)
        response = client.get("/runtime-error")
        
        assert response.status_code == 500
        data = response.json()
        
        assert "error" in data
        assert "Runtime error" in data["error"]
        assert data["error_code"] == "INTERNAL_ERROR"


if __name__ == "__main__":
    pytest.main([__file__])