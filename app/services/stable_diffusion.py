"""
Stable Diffusion service for image generation with GTX 1060 optimizations.
"""
import logging
import torch
import gc
import uuid
from typing import Optional, Dict, Any
from diffusers import StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging
import psutil
import os

# Import our optimization services
from .memory_manager import memory_manager
from .performance_monitor import performance_monitor

# Configure logging
logger = logging.getLogger(__name__)

# Suppress diffusers warnings for cleaner logs
diffusers_logging.set_verbosity_error()


class StableDiffusionModelManager:
    """
    Manages Stable Diffusion model initialization and GPU memory optimization for GTX 1060.
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self.device = None
        self.is_initialized = False
        self._cuda_available = False
        self._gpu_memory_gb = 0
        
    def check_cuda_availability(self) -> Dict[str, Any]:
        """
        Check CUDA availability and GPU memory information.
        
        Returns:
            Dict containing CUDA status and GPU information
        """
        cuda_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "gpu_name": None
        }
        
        if torch.cuda.is_available():
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["gpu_count"] = torch.cuda.device_count()
            
            if cuda_info["gpu_count"] > 0:
                # Get GPU memory in GB
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                cuda_info["gpu_memory_gb"] = round(gpu_memory_bytes / (1024**3), 1)
                cuda_info["gpu_name"] = torch.cuda.get_device_properties(0).name
                
        self._cuda_available = cuda_info["cuda_available"]
        self._gpu_memory_gb = cuda_info["gpu_memory_gb"]
        
        logger.info(f"CUDA availability check: {cuda_info}")
        return cuda_info
        
    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for model execution.
        
        Returns:
            Device string ("cuda" or "cpu")
        """
        if self._cuda_available and self._gpu_memory_gb >= 4.0:
            return "cuda"
        else:
            logger.warning(
                f"Using CPU fallback. CUDA available: {self._cuda_available}, "
                f"GPU memory: {self._gpu_memory_gb}GB"
            )
            return "cpu"
            
    def initialize_model(self) -> bool:
        """
        Initialize the Stable Diffusion model with GTX 1060 optimizations.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing Stable Diffusion model: {self.model_id}")
            
            # Check system requirements
            cuda_info = self.check_cuda_availability()
            self.device = self._get_optimal_device()
            
            # Configure model loading parameters
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "safety_checker": None,  # Disable safety checker to save memory
                "requires_safety_checker": False,
            }
            
            # Load the pipeline
            logger.info("Loading Stable Diffusion pipeline...")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Apply GTX 1060 specific optimizations
            if self.device == "cuda":
                self._apply_gpu_optimizations()
            
            self.is_initialized = True
            logger.info(f"Model initialized successfully on {self.device}")
            
            # Log memory usage
            self._log_memory_usage()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.is_initialized = False
            return False
            
    def _apply_gpu_optimizations(self):
        """
        Apply GTX 1060 specific optimizations to reduce memory usage.
        """
        logger.info("Applying GTX 1060 optimizations...")
        
        # Enable attention slicing to reduce memory usage
        # Slice size of 1 provides maximum memory savings
        self.pipeline.enable_attention_slicing(1)
        logger.info("Enabled attention slicing")
        
        # Enable sequential CPU offloading for memory efficiency
        # This moves model components to CPU when not in use
        self.pipeline.enable_sequential_cpu_offload()
        logger.info("Enabled sequential CPU offloading")
        
        # Enable memory efficient attention if available
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers attention: {e}")
            
    def _log_memory_usage(self):
        """Log current memory usage for monitoring."""
        # System RAM usage
        ram_usage = psutil.virtual_memory()
        logger.info(f"System RAM usage: {ram_usage.percent}% ({ram_usage.used / (1024**3):.1f}GB / {ram_usage.total / (1024**3):.1f}GB)")
        
        # GPU memory usage if CUDA available
        if self._cuda_available and torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU memory - Allocated: {gpu_memory_allocated:.1f}GB, Reserved: {gpu_memory_reserved:.1f}GB")
            
    def cleanup_memory(self):
        """
        Clean up GPU memory between requests to prevent memory leaks.
        """
        if self._cuda_available and torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("GPU memory cleanup completed")
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "cuda_available": self._cuda_available,
            "gpu_memory_gb": self._gpu_memory_gb,
        }
        
    def is_ready(self) -> bool:
        """
        Check if the model is ready for inference.
        
        Returns:
            True if model is initialized and ready
        """
        return self.is_initialized and self.pipeline is not None


class ImageGenerationService:
    """
    Service for generating images using the Stable Diffusion model.
    """
    
    def __init__(self, model_manager: StableDiffusionModelManager):
        self.model_manager = model_manager
        self.generation_timeout = 60  # seconds
        
    async def generate_image(
        self,
        prompt: str,
        steps: int = 20,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt with advanced memory management.
        
        Args:
            prompt: Text description for image generation
            steps: Number of diffusion steps (1-50)
            width: Image width in pixels (multiple of 64)
            height: Image height in pixels (multiple of 64)
            seed: Random seed for reproducible generation
            
        Returns:
            Dictionary containing generated image and metadata
            
        Raises:
            RuntimeError: If model is not initialized or generation fails
            TimeoutError: If generation exceeds timeout
        """
        if not self.model_manager.is_ready():
            raise RuntimeError("Model is not initialized or ready for inference")
            
        # Generate random seed if not provided
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        
        logger.info(f"Starting image generation [{request_id}] - prompt: '{prompt[:50]}...', steps: {steps}, size: {width}x{height}, seed: {seed}")
        
        # Start performance tracking
        performance_monitor.start_request_tracking(
            request_id=request_id,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed
        )
        
        try:
            # Use memory manager context for automatic queuing and cleanup
            async with memory_manager.request_context(request_id, width, height, steps):
                # Mark that processing has started (left queue)
                performance_monitor.mark_request_processing(request_id)
                
                # Set up generation parameters
                generator = torch.Generator(device=self.model_manager.device)
                generator.manual_seed(seed)
                
                # Record generation start time
                import time
                start_time = time.time()
                
                # Generate image with timeout handling
                image = await self._generate_with_timeout(
                    prompt=prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    generator=generator
                )
                
                # Calculate generation time
                generation_time = time.time() - start_time
                
                logger.info(f"Image generation [{request_id}] completed in {generation_time:.2f} seconds")
                
                # Complete performance tracking
                performance_monitor.complete_request_tracking(request_id, success=True)
                
                return {
                    "image": image,
                    "metadata": {
                        "prompt": prompt,
                        "steps": steps,
                        "width": width,
                        "height": height,
                        "seed": seed,
                        "generation_time_seconds": round(generation_time, 2),
                        "model_version": self.model_manager.model_id,
                        "request_id": request_id
                    }
                }
            
        except Exception as e:
            # Complete performance tracking with error
            error_type = type(e).__name__
            performance_monitor.complete_request_tracking(
                request_id, 
                success=False, 
                error_type=error_type, 
                error_message=str(e)
            )
            
            logger.error(f"Image generation [{request_id}] failed: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}")
            
    async def _generate_with_timeout(self, **kwargs):
        """
        Generate image with timeout handling.
        
        Args:
            **kwargs: Arguments to pass to the pipeline
            
        Returns:
            Generated PIL Image
            
        Raises:
            TimeoutError: If generation exceeds timeout
        """
        import asyncio
        import concurrent.futures
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                future = loop.run_in_executor(
                    executor,
                    self._run_pipeline,
                    kwargs
                )
                
                # Wait for completion with timeout
                image = await asyncio.wait_for(future, timeout=self.generation_timeout)
                return image
                
            except asyncio.TimeoutError:
                logger.error(f"Image generation timed out after {self.generation_timeout} seconds")
                raise TimeoutError(f"Image generation timed out after {self.generation_timeout} seconds")
                
    def _run_pipeline(self, kwargs: Dict[str, Any]):
        """
        Run the diffusion pipeline synchronously.
        
        Args:
            kwargs: Pipeline arguments
            
        Returns:
            Generated PIL Image
        """
        try:
            # Run the pipeline
            result = self.model_manager.pipeline(**kwargs)
            
            # Extract the first (and only) image from the result
            image = result.images[0]
            
            return image
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
            
    def get_generation_status(self) -> Dict[str, Any]:
        """
        Get the current status of the generation service.
        
        Returns:
            Dictionary containing service status information
        """
        return {
            "model_ready": self.model_manager.is_ready(),
            "model_info": self.model_manager.get_model_info(),
            "timeout_seconds": self.generation_timeout
        }


class ResponseFormattingService:
    """
    Service for formatting image generation responses and handling errors.
    """
    
    @staticmethod
    def image_to_base64(image) -> str:
        """
        Convert PIL Image to base64 encoded string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string of the image
        """
        import io
        import base64
        
        try:
            # Convert image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {str(e)}")
            raise RuntimeError(f"Image conversion failed: {str(e)}")
            
    @staticmethod
    def format_generation_response(generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the image generation result into API response format.
        
        Args:
            generation_result: Result from ImageGenerationService.generate_image()
            
        Returns:
            Formatted response dictionary
        """
        try:
            image = generation_result["image"]
            metadata = generation_result["metadata"]
            
            # Convert image to base64
            image_base64 = ResponseFormattingService.image_to_base64(image)
            
            # Format response
            response = {
                "image_base64": image_base64,
                "metadata": {
                    "prompt": metadata["prompt"],
                    "steps": metadata["steps"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "seed": metadata["seed"],
                    "generation_time_seconds": metadata["generation_time_seconds"],
                    "model_version": metadata["model_version"]
                }
            }
            
            logger.info(f"Successfully formatted response for image {metadata['width']}x{metadata['height']}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to format generation response: {str(e)}")
            raise RuntimeError(f"Response formatting failed: {str(e)}")
            
    @staticmethod
    def format_error_response(error: Exception, error_code: str = "GENERATION_ERROR") -> Dict[str, Any]:
        """
        Format an error into a standardized error response.
        
        Args:
            error: The exception that occurred
            error_code: Error code for categorization
            
        Returns:
            Formatted error response dictionary
        """
        error_message = str(error)
        
        # Categorize common errors only if no specific code provided
        if error_code == "GENERATION_ERROR":
            if isinstance(error, TimeoutError):
                error_code = "TIMEOUT_ERROR"
            elif isinstance(error, RuntimeError) and "not initialized" in error_message:
                error_code = "MODEL_NOT_READY"
            elif isinstance(error, RuntimeError) and "memory" in error_message.lower():
                error_code = "MEMORY_ERROR"
            elif isinstance(error, ValueError):
                error_code = "VALIDATION_ERROR"
            
        response = {
            "error": error_message,
            "error_code": error_code
        }
        
        # Add details for specific error types
        if error_code in ["TIMEOUT_ERROR", "REQUEST_TIMEOUT"]:
            response["details"] = "Request exceeded the maximum allowed time"
        elif error_code == "MODEL_NOT_READY":
            response["details"] = "The AI model is not initialized or ready for inference"
        elif error_code == "MEMORY_ERROR":
            response["details"] = "Insufficient GPU memory for image generation"
        elif error_code == "VALIDATION_ERROR":
            response["details"] = "Invalid parameters provided for image generation"
            
        logger.warning(f"Formatted error response: {error_code} - {error_message}")
        return response
        
    @staticmethod
    def get_response_headers(content_type: str = "application/json") -> Dict[str, str]:
        """
        Get standard HTTP headers for API responses.
        
        Args:
            content_type: MIME type for the response
            
        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Content-Type": content_type,
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }


# Global instances
model_manager = StableDiffusionModelManager()
generation_service = ImageGenerationService(model_manager)
response_formatter = ResponseFormattingService()