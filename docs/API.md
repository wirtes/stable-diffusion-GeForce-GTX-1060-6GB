# Stable Diffusion API Documentation

## Overview

The Stable Diffusion API is a REST service that generates images from text descriptions using AI. It's optimized for NVIDIA GTX 1060 6GB and runs in a Docker container.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Generate Image

Generate an image from a text prompt.

**Endpoint:** `POST /generate`

**Request Body:**

```json
{
  "prompt": "A beautiful sunset over mountains",
  "steps": 20,
  "width": 512,
  "height": 512,
  "seed": 12345
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of the image to generate (1-500 characters) |
| `steps` | integer | No | 20 | Number of diffusion steps (1-50) |
| `width` | integer | No | 512 | Image width in pixels (256-768, must be multiple of 64) |
| `height` | integer | No | 512 | Image height in pixels (256-768, must be multiple of 64) |
| `seed` | integer | No | random | Random seed for reproducible results (0 to 4294967295) |

**Response (200 OK):**

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "metadata": {
    "prompt": "A beautiful sunset over mountains",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": 12345,
    "generation_time_seconds": 15.42,
    "model_version": "runwayml/stable-diffusion-v1-5"
  }
}
```

**Error Responses:**

**400 Bad Request - Invalid Parameters:**
```json
{
  "error": "Validation error",
  "details": "Width must be a multiple of 64",
  "error_code": "VALIDATION_ERROR"
}
```

**408 Request Timeout:**
```json
{
  "error": "Generation timeout",
  "details": "Image generation took longer than 60 seconds",
  "error_code": "TIMEOUT_ERROR"
}
```

**503 Service Unavailable:**
```json
{
  "error": "Service unavailable",
  "details": "GPU memory exhausted, please try again later",
  "error_code": "RESOURCE_EXHAUSTED"
}
```

### Health Check

Check the service health and GPU status.

**Endpoint:** `GET /health`

**Response (200 OK):**

```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_memory_used": "2.1 GB",
  "gpu_memory_total": "6.0 GB",
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

**Response (503 Service Unavailable):**

```json
{
  "status": "unhealthy",
  "gpu_available": false,
  "gpu_memory_used": "0 GB",
  "gpu_memory_total": "0 GB",
  "model_loaded": false,
  "uptime_seconds": 120,
  "error": "Model failed to load"
}
```

## Rate Limits

- Maximum 5 concurrent requests
- Requests are queued when limit is exceeded
- Queue timeout: 300 seconds

## Image Format

- Generated images are returned as base64-encoded PNG data
- Use the `image_base64` field from the response
- Decode the base64 string to get the binary image data

## Example Usage

### Decode Base64 Image (Python)

```python
import base64
from PIL import Image
from io import BytesIO

# Get base64 string from API response
image_base64 = response_data["image_base64"]

# Decode base64 to bytes
image_bytes = base64.b64decode(image_base64)

# Convert to PIL Image
image = Image.open(BytesIO(image_bytes))

# Save to file
image.save("generated_image.png")
```

### Decode Base64 Image (JavaScript)

```javascript
// Get base64 string from API response
const imageBase64 = responseData.image_base64;

// Create image element
const img = document.createElement('img');
img.src = `data:image/png;base64,${imageBase64}`;

// Add to page
document.body.appendChild(img);
```