"""
Request/response logging middleware for the Stable Diffusion API.
"""
import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log HTTP requests and responses with timing information.
    """
    
    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        """
        Initialize request logging middleware.
        
        Args:
            app: FastAPI application instance
            log_request_body: Whether to log request body content
            log_response_body: Whether to log response body content
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with logging.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: HTTP response from handler
        """
        # Generate unique request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        
        # Add request ID to request state for use in handlers
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {client_ip} - User-Agent: {request.headers.get('user-agent', 'Unknown')}"
        )
        
        # Log request body if enabled and present
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await self._get_request_body(request)
                if body:
                    logger.debug(f"[{request_id}] Request body: {body}")
            except Exception as e:
                logger.warning(f"[{request_id}] Could not read request body: {e}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.3f}s"
            )
            
            # Log response body if enabled and it's an error
            if self.log_response_body and response.status_code >= 400:
                try:
                    response_body = await self._get_response_body(response)
                    if response_body:
                        logger.debug(f"[{request_id}] Response body: {response_body}")
                except Exception as e:
                    logger.warning(f"[{request_id}] Could not read response body: {e}")
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as exc:
            # Log exception with request context
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Exception: {str(exc)} - Time: {process_time:.3f}s"
            )
            raise exc
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request headers.
        
        Args:
            request: HTTP request object
            
        Returns:
            Client IP address string
        """
        # Check for forwarded headers (common in reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    async def _get_request_body(self, request: Request) -> str:
        """
        Safely read request body for logging.
        
        Args:
            request: HTTP request object
            
        Returns:
            Request body as string, truncated if too long
        """
        try:
            body = await request.body()
            if body:
                body_str = body.decode('utf-8')
                # Truncate long bodies for logging
                if len(body_str) > 1000:
                    body_str = body_str[:1000] + "... [truncated]"
                return body_str
        except Exception:
            pass
        return ""
    
    async def _get_response_body(self, response: Response) -> str:
        """
        Safely read response body for logging.
        
        Args:
            response: HTTP response object
            
        Returns:
            Response body as string, truncated if too long
        """
        try:
            if hasattr(response, 'body'):
                body = response.body
                if body:
                    body_str = body.decode('utf-8')
                    # Truncate long bodies for logging
                    if len(body_str) > 1000:
                        body_str = body_str[:1000] + "... [truncated]"
                    return body_str
        except Exception:
            pass
        return ""


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log performance metrics and slow requests.
    """
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        """
        Initialize performance logging middleware.
        
        Args:
            app: FastAPI application instance
            slow_request_threshold: Threshold in seconds to log slow requests
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with performance monitoring.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: HTTP response from handler
        """
        start_time = time.time()
        
        # Get memory usage before request (if available)
        memory_before = self._get_memory_usage()
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            process_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before if memory_before and memory_after else None
            
            # Log performance metrics
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            if process_time > self.slow_request_threshold:
                logger.warning(
                    f"[{request_id}] SLOW REQUEST: {request.method} {request.url.path} - "
                    f"Time: {process_time:.3f}s (threshold: {self.slow_request_threshold}s)"
                )
            
            # Log memory usage for generation endpoints
            if request.url.path == "/generate" and memory_delta:
                logger.info(
                    f"[{request_id}] Memory usage - Delta: {memory_delta:.1f}MB, "
                    f"After: {memory_after:.1f}MB"
                )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Memory-Usage-MB"] = str(memory_after) if memory_after else "unknown"
            if memory_delta:
                response.headers["X-Memory-Delta-MB"] = str(memory_delta)
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                f"[{request_id}] ERROR: {request.method} {request.url.path} - "
                f"Exception: {str(exc)} - Time: {process_time:.3f}s"
            )
            raise exc
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB, or None if unavailable
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return None


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log security-related events and suspicious activity.
    """
    
    def __init__(self, app, rate_limit_threshold: int = 100):
        """
        Initialize security logging middleware.
        
        Args:
            app: FastAPI application instance
            rate_limit_threshold: Number of requests per minute to flag as suspicious
        """
        super().__init__(app)
        self.rate_limit_threshold = rate_limit_threshold
        self.client_requests = {}  # Simple in-memory tracking
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with security monitoring.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: HTTP response from handler
        """
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Track request rate per client
        self._track_client_requests(client_ip, current_time)
        
        # Check for suspicious patterns
        self._check_suspicious_activity(request, client_ip)
        
        try:
            response = await call_next(request)
            
            # Log failed authentication attempts (if any)
            if response.status_code == 401:
                logger.warning(
                    f"Authentication failed - IP: {client_ip} - "
                    f"Path: {request.url.path} - User-Agent: {request.headers.get('user-agent', 'Unknown')}"
                )
            
            # Log access to sensitive endpoints
            if request.url.path in ["/generate"] and response.status_code == 200:
                request_id = getattr(request.state, 'request_id', 'unknown')
                logger.info(
                    f"[{request_id}] Successful generation request - IP: {client_ip}"
                )
            
            return response
            
        except Exception as exc:
            # Log security-related exceptions
            logger.error(
                f"Security exception - IP: {client_ip} - "
                f"Path: {request.url.path} - Exception: {str(exc)}"
            )
            raise exc
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _track_client_requests(self, client_ip: str, current_time: float):
        """Track request rate per client IP."""
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []
        
        # Add current request timestamp
        self.client_requests[client_ip].append(current_time)
        
        # Clean up old requests (older than 1 minute)
        minute_ago = current_time - 60
        self.client_requests[client_ip] = [
            timestamp for timestamp in self.client_requests[client_ip]
            if timestamp > minute_ago
        ]
        
        # Check if rate limit exceeded
        request_count = len(self.client_requests[client_ip])
        if request_count > self.rate_limit_threshold:
            logger.warning(
                f"High request rate detected - IP: {client_ip} - "
                f"Requests in last minute: {request_count}"
            )
    
    def _check_suspicious_activity(self, request: Request, client_ip: str):
        """Check for suspicious request patterns."""
        # Check for suspicious user agents
        user_agent = request.headers.get('user-agent', '').lower()
        suspicious_agents = ['bot', 'crawler', 'scanner', 'exploit']
        
        if any(agent in user_agent for agent in suspicious_agents):
            logger.warning(
                f"Suspicious user agent - IP: {client_ip} - "
                f"User-Agent: {request.headers.get('user-agent', 'Unknown')}"
            )
        
        # Check for unusual request patterns
        if request.method not in ['GET', 'POST', 'OPTIONS']:
            logger.warning(
                f"Unusual HTTP method - IP: {client_ip} - "
                f"Method: {request.method} - Path: {request.url.path}"
            )
        
        # Check for requests to non-existent endpoints
        valid_paths = ['/generate', '/health', '/docs', '/openapi.json']
        if request.url.path not in valid_paths and not request.url.path.startswith('/docs'):
            logger.info(
                f"Request to unknown endpoint - IP: {client_ip} - "
                f"Path: {request.url.path}"
            )