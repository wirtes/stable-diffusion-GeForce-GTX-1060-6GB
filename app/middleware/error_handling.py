"""
Global error handling middleware for the Stable Diffusion API.
"""
import logging
import traceback
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.stable_diffusion import response_formatter

logger = logging.getLogger(__name__)


class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """
    Global exception handling middleware to catch and format unhandled exceptions.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and handle any unhandled exceptions.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: HTTP response, either from handler or error response
        """
        try:
            # Process the request
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Log the unhandled exception with full traceback
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: {str(exc)}",
                exc_info=True
            )
            
            # Format error response
            error_response = response_formatter.format_error_response(
                exc, 
                "INTERNAL_ERROR"
            )
            
            # Add request context to error details
            error_response["details"] = f"Unhandled server error in {request.method} {request.url.path}"
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response,
                headers=response_formatter.get_response_headers()
            )


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle request timeouts for long-running operations.
    """
    
    def __init__(self, app, timeout_seconds: int = 120):
        """
        Initialize timeout middleware.
        
        Args:
            app: FastAPI application instance
            timeout_seconds: Maximum request processing time in seconds
        """
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with timeout handling.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: HTTP response or timeout error
        """
        import asyncio
        
        try:
            # Execute request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            return response
            
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout after {self.timeout_seconds}s: {request.method} {request.url.path}"
            )
            
            # Format timeout error response
            error_response = response_formatter.format_error_response(
                TimeoutError(f"Request timed out after {self.timeout_seconds} seconds"),
                "REQUEST_TIMEOUT"
            )
            
            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                content=error_response,
                headers=response_formatter.get_response_headers()
            )


class ValidationErrorMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle Pydantic validation errors and format them consistently.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and handle validation errors.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: HTTP response or formatted validation error
        """
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Check if it's a validation error from FastAPI/Pydantic
            if hasattr(exc, 'errors') and callable(getattr(exc, 'errors')):
                # This is likely a Pydantic ValidationError
                validation_errors = exc.errors()
                
                logger.warning(
                    f"Validation error in {request.method} {request.url.path}: {validation_errors}"
                )
                
                # Format validation errors into a readable message
                error_messages = []
                for error in validation_errors:
                    field = " -> ".join(str(loc) for loc in error.get('loc', []))
                    message = error.get('msg', 'Invalid value')
                    error_messages.append(f"{field}: {message}")
                
                error_response = {
                    "error": "Request validation failed",
                    "details": "; ".join(error_messages),
                    "error_code": "VALIDATION_ERROR"
                }
                
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content=error_response,
                    headers=response_formatter.get_response_headers()
                )
            
            # Re-raise if not a validation error
            raise exc