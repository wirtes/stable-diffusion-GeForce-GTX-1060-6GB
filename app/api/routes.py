"""
API routes for the Stable Diffusion image generation service.
"""
import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    HealthResponse,
    ErrorResponse
)
from app.services.stable_diffusion import (
    model_manager,
    generation_service,
    response_formatter
)
from app.services.validation import ValidationService
from app.services.memory_manager import memory_manager
from app.services.performance_monitor import performance_monitor

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate",
    response_model=ImageGenerationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate image from text prompt",
    description="Generate an AI image from a text description using Stable Diffusion"
)
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    This endpoint accepts a text description and optional parameters to generate
    an AI image. The service is optimized for GTX 1060 6GB GPU constraints.
    
    Args:
        request: Image generation request containing prompt and parameters
        
    Returns:
        ImageGenerationResponse: Generated image as base64 with metadata
        
    Raises:
        HTTPException: For various error conditions (validation, timeout, etc.)
    """
    try:
        logger.info(f"Received image generation request: prompt='{request.prompt[:50]}...', steps={request.steps}, size={request.width}x{request.height}")
        
        # Check if model is ready
        if not model_manager.is_ready():
            logger.error("Model is not initialized or ready for inference")
            error_response = response_formatter.format_error_response(
                RuntimeError("Model is not initialized or ready for inference"),
                "MODEL_NOT_READY"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_response
            )
        
        # Additional validation using validation service
        try:
            validated_params = ValidationService.validate_request_params(
                prompt=request.prompt,
                steps=request.steps,
                width=request.width,
                height=request.height,
                seed=request.seed
            )
        except ValueError as e:
            logger.warning(f"Request validation failed: {str(e)}")
            error_response = response_formatter.format_error_response(e, "VALIDATION_ERROR")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_response
            )
        
        # Generate image
        try:
            generation_result = await generation_service.generate_image(
                prompt=validated_params["prompt"],
                steps=validated_params["steps"],
                width=validated_params["width"],
                height=validated_params["height"],
                seed=validated_params["seed"]
            )
            
            # Format response
            formatted_response = response_formatter.format_generation_response(generation_result)
            
            logger.info(f"Successfully generated image in {generation_result['metadata']['generation_time_seconds']}s")
            
            return JSONResponse(
                content=formatted_response,
                headers=response_formatter.get_response_headers()
            )
            
        except TimeoutError as e:
            logger.error(f"Image generation timed out: {str(e)}")
            error_response = response_formatter.format_error_response(e, "TIMEOUT_ERROR")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=error_response
            )
            
        except RuntimeError as e:
            logger.error(f"Image generation failed: {str(e)}")
            
            # Check if it's a memory error
            if "memory" in str(e).lower():
                error_response = response_formatter.format_error_response(e, "MEMORY_ERROR")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=error_response
                )
            else:
                error_response = response_formatter.format_error_response(e, "GENERATION_ERROR")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_response
                )
                
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error in generate_image: {str(e)}", exc_info=True)
        error_response = response_formatter.format_error_response(e, "INTERNAL_ERROR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Check the health status of the Stable Diffusion API service"
)
async def health_check():
    """
    Check the health status of the Stable Diffusion API service.
    
    This endpoint provides comprehensive information about service availability,
    GPU status, model loading status, system resources, and performance metrics.
    
    Returns:
        HealthResponse: Comprehensive service health information
    """
    try:
        logger.debug("Health check requested")
        
        # Use the comprehensive health monitoring system
        from app.core.health import health_monitor
        health_report = await health_monitor.perform_health_check()
        
        # Extract key information for backward compatibility
        overall_status = health_report['overall_status']
        components = health_report['components']
        
        # Build legacy response format
        legacy_response = {
            "status": overall_status,
            "gpu_available": components.get('gpu', {}).get('details', {}).get('cuda_available', False),
            "model_loaded": components.get('model', {}).get('details', {}).get('is_initialized', False),
            "memory_usage": None
        }
        
        # Add memory usage if available
        if health_report.get('metrics'):
            metrics = health_report['metrics']
            if metrics.get('gpu_memory_used_mb') is not None:
                legacy_response["memory_usage"] = {
                    "gpu_memory_used_mb": metrics['gpu_memory_used_mb'],
                    "gpu_memory_total_mb": metrics['gpu_memory_total_mb'],
                    "gpu_memory_percent": metrics['gpu_memory_percent']
                }
        
        # Determine HTTP status code
        if overall_status in ['healthy', 'degraded']:
            status_code = status.HTTP_200_OK
        elif overall_status == 'initializing':
            status_code = status.HTTP_202_ACCEPTED
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        logger.debug(f"Health check response: {overall_status}")
        
        return JSONResponse(
            content=legacy_response,
            status_code=status_code,
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        
        # Return unhealthy status but don't raise exception
        error_response = {
            "status": "unhealthy",
            "gpu_available": False,
            "model_loaded": False,
            "memory_usage": None
        }
        
        return JSONResponse(
            content=error_response,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers=response_formatter.get_response_headers()
        )


@router.get(
    "/health/detailed",
    summary="Detailed health check",
    description="Get comprehensive health information including all components and metrics"
)
async def detailed_health_check():
    """
    Get detailed health information including all components and system metrics.
    
    This endpoint provides comprehensive health monitoring data including:
    - Individual component health status
    - System resource usage metrics
    - GPU status and memory usage
    - Performance metrics and history
    
    Returns:
        Detailed health report with all monitoring data
    """
    try:
        from app.core.health import health_monitor
        health_report = await health_monitor.perform_health_check()
        
        # Add metrics history
        health_report['metrics_history'] = health_monitor.get_metrics_history(minutes=30)
        
        # Determine HTTP status code
        overall_status = health_report['overall_status']
        if overall_status in ['healthy', 'degraded']:
            status_code = status.HTTP_200_OK
        elif overall_status == 'initializing':
            status_code = status.HTTP_202_ACCEPTED
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content=health_report,
            status_code=status_code,
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}", exc_info=True)
        
        error_response = {
            "overall_status": "unknown",
            "components": {},
            "metrics": None,
            "error": str(e)
        }
        
        return JSONResponse(
            content=error_response,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers=response_formatter.get_response_headers()
        )


@router.get(
    "/health/summary",
    summary="Health summary",
    description="Get a quick summary of service health status"
)
async def health_summary():
    """
    Get a quick summary of service health status.
    
    Returns:
        Brief health summary with component counts and overall status
    """
    try:
        from app.core.health import health_monitor
        summary = health_monitor.get_health_summary()
        
        # Determine HTTP status code
        overall_status = summary['overall_status']
        if overall_status in ['healthy', 'degraded']:
            status_code = status.HTTP_200_OK
        elif overall_status == 'initializing':
            status_code = status.HTTP_202_ACCEPTED
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content=summary,
            status_code=status_code,
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Health summary failed: {str(e)}", exc_info=True)
        
        error_response = {
            "overall_status": "unknown",
            "error": str(e)
        }
        
        return JSONResponse(
            content=error_response,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers=response_formatter.get_response_headers()
        )


@router.get(
    "/performance/stats",
    summary="Performance statistics",
    description="Get performance statistics and metrics for the API"
)
async def get_performance_stats(time_window: int = 60):
    """
    Get performance statistics for the specified time window.
    
    Args:
        time_window: Time window in minutes for statistics calculation (default: 60)
        
    Returns:
        Performance statistics including generation times, success rates, and resource usage
    """
    try:
        stats = performance_monitor.get_performance_stats(time_window_minutes=time_window)
        
        return JSONResponse(
            content=stats.__dict__,
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {str(e)}")
        error_response = response_formatter.format_error_response(e, "STATS_ERROR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response
        )


@router.get(
    "/performance/system",
    summary="System health metrics",
    description="Get current system health and resource usage metrics"
)
async def get_system_health():
    """
    Get current system health status and resource usage metrics.
    
    Returns:
        System health information including CPU, memory, GPU usage and status
    """
    try:
        health_info = performance_monitor.get_system_health()
        
        return JSONResponse(
            content=health_info,
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        error_response = response_formatter.format_error_response(e, "HEALTH_ERROR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response
        )


@router.get(
    "/performance/queue",
    summary="Queue status",
    description="Get current request queue status and memory management info"
)
async def get_queue_status():
    """
    Get current request queue status and memory management information.
    
    Returns:
        Queue status including active requests, queue length, and memory usage
    """
    try:
        queue_status = memory_manager.get_queue_status()
        performance_metrics = memory_manager.get_performance_metrics()
        
        combined_status = {
            "queue": queue_status,
            "performance": performance_metrics
        }
        
        return JSONResponse(
            content=combined_status,
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {str(e)}")
        error_response = response_formatter.format_error_response(e, "QUEUE_ERROR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response
        )


@router.post(
    "/performance/export",
    summary="Export performance metrics",
    description="Export performance metrics to a file"
)
async def export_performance_metrics(format: str = "json", filename: str = None):
    """
    Export performance metrics to a file.
    
    Args:
        format: Export format ("json" or "csv")
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Export confirmation with file information
    """
    try:
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.{format}"
        
        performance_monitor.export_metrics(filename, format)
        
        return JSONResponse(
            content={
                "message": "Metrics exported successfully",
                "filename": filename,
                "format": format
            },
            headers=response_formatter.get_response_headers()
        )
        
    except Exception as e:
        logger.error(f"Failed to export metrics: {str(e)}")
        error_response = response_formatter.format_error_response(e, "EXPORT_ERROR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response
        )