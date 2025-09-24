"""
Stable Diffusion API - Main application entry point
"""
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Stable Diffusion API",
    description="AI image generation API optimized for GTX 1060",
    version="1.0.0"
)

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)