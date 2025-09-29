"""
Stable Diffusion API - Main application entry point
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from app.api.routes import router
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
from app.core.startup import lifespan, startup_manager
from app.core.health import health_monitor


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    try:
        # Use the comprehensive startup system
        await lifespan(app).__aenter__()
        
        # Start health monitoring
        await health_monitor.start_monitoring()
        
        yield
        
    finally:
        # Shutdown
        await health_monitor.stop_monitoring()
        await lifespan(app).__aexit__(None, None, None)


# Create FastAPI application with lifespan management
app = FastAPI(
    title="Stable Diffusion API",
    description="AI image generation API optimized for GTX 1060",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan
)

# Add CORS middleware for web client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.allowed_hosts,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Add custom middleware in order (last added = first executed)
# Security logging should be first to catch all requests
app.add_middleware(SecurityLoggingMiddleware, rate_limit_threshold=50)

# Performance monitoring
app.add_middleware(PerformanceLoggingMiddleware, slow_request_threshold=10.0)

# Request/response logging
app.add_middleware(RequestLoggingMiddleware, log_request_body=False, log_response_body=True)

# Timeout handling for long-running requests
app.add_middleware(TimeoutMiddleware, timeout_seconds=settings.api.generation_timeout_seconds + 60)

# Validation error handling
app.add_middleware(ValidationErrorMiddleware)

# Global exception handling (should be last to catch everything)
app.add_middleware(GlobalExceptionMiddleware)

# Include API routes
app.include_router(router)


# Legacy event handlers for backward compatibility
@app.on_event("startup")
async def legacy_startup_event():
    """Legacy startup event handler"""
    logger = logging.getLogger(__name__)
    logger.info("Legacy startup event triggered")


@app.on_event("shutdown")
async def legacy_shutdown_event():
    """Legacy shutdown event handler"""
    logger = logging.getLogger(__name__)
    logger.info("Legacy shutdown event triggered")


if __name__ == "__main__":
    import uvicorn
    
    # Run with configuration from settings
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        access_log=True
    )