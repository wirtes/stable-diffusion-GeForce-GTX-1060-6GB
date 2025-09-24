"""
API routes placeholder - will be implemented in later tasks
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "stable-diffusion-api"}