# Stable Diffusion API

A containerized REST API service for generating images using Stable Diffusion, optimized for NVIDIA GTX 1060 6GB.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with 6GB+ VRAM
- NVIDIA Docker runtime

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd stable-diffusion-api
   ```

2. **Start the service:**
   ```bash
   docker-compose up -d
   ```

3. **Test the API:**
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A beautiful sunset over mountains", "steps": 20}'
   ```

## Features

- **REST API** for image generation from text prompts
- **GPU Optimized** for NVIDIA GTX 1060 6GB
- **Containerized** deployment with Docker
- **Memory Efficient** with attention slicing and CPU offloading
- **Configurable** generation parameters (steps, dimensions, seed)
- **Error Handling** with comprehensive error responses
- **Health Monitoring** with GPU status reporting
- **Request Queuing** to prevent memory exhaustion

## API Endpoints

### Generate Image
`POST /generate`

Generate an image from a text description.

**Request:**
```json
{
  "prompt": "A serene lake with mountains in the background",
  "steps": 25,
  "width": 512,
  "height": 512,
  "seed": 42
}
```

**Response:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...",
  "metadata": {
    "prompt": "A serene lake with mountains in the background",
    "steps": 25,
    "width": 512,
    "height": 512,
    "seed": 42,
    "generation_time_seconds": 18.5,
    "model_version": "runwayml/stable-diffusion-v1-5"
  }
}
```

### Health Check
`GET /health`

Check service status and GPU information.

**Response:**
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

## Usage Examples

### Python Client

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Generate image
response = requests.post('http://localhost:8000/generate', json={
    'prompt': 'A cute robot in a garden',
    'steps': 20,
    'width': 512,
    'height': 512
})

data = response.json()

# Save image
image_bytes = base64.b64decode(data['image_base64'])
image = Image.open(BytesIO(image_bytes))
image.save('generated_image.png')

print(f"Generated in {data['metadata']['generation_time_seconds']:.2f} seconds")
```

### JavaScript/Node.js Client

```javascript
const fetch = require('node-fetch');
const fs = require('fs');

async function generateImage() {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: 'A magical forest with glowing mushrooms',
      steps: 25
    })
  });

  const data = await response.json();
  
  // Save image
  const buffer = Buffer.from(data.image_base64, 'base64');
  fs.writeFileSync('generated_image.png', buffer);
  
  console.log(`Generated in ${data.metadata.generation_time_seconds}s`);
}

generateImage();
```

### cURL Examples

```bash
# Basic generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "steps": 20
  }' | jq '.metadata'

# Save image to file
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cozy cabin in the woods",
    "steps": 25
  }' | jq -r '.image_base64' | base64 -d > cabin.png

# Health check
curl http://localhost:8000/health | jq '.'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |
| `MODEL_CACHE_DIR` | `./models` | Directory for model storage |
| `MAX_CONCURRENT_REQUESTS` | `3` | Maximum concurrent requests |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_PORT` | `8000` | API server port |

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  stable-diffusion:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MAX_CONCURRENT_REQUESTS=2
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Performance

### GTX 1060 6GB Optimizations

The service includes several optimizations for the GTX 1060:

- **Attention Slicing**: Reduces memory usage during generation
- **CPU Offloading**: Moves model components to CPU when not in use
- **Half Precision**: Uses float16 to reduce memory footprint
- **Request Queuing**: Prevents memory exhaustion from concurrent requests

### Typical Performance

| Image Size | Steps | Generation Time | Memory Usage |
|------------|-------|-----------------|--------------|
| 512x512    | 20    | 15-20 seconds   | ~4GB VRAM    |
| 768x512    | 25    | 25-30 seconds   | ~5GB VRAM    |
| 768x768    | 30    | 35-45 seconds   | ~5.5GB VRAM  |

## Deployment

### Development
```bash
# Start with development settings
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production
```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# With reverse proxy
docker-compose -f docker-compose.prod.yml -f docker-compose.nginx.yml up -d
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Monitoring

### Health Monitoring
```bash
# Check service health
curl http://localhost:8000/health

# Monitor GPU usage
watch -n 1 nvidia-smi

# View container logs
docker-compose logs -f
```

### Performance Monitoring
```bash
# Run performance benchmark
python scripts/benchmark_performance.py

# Memory stress test
python scripts/memory_stress_test.py

# GTX 1060 optimization test
python scripts/optimize_gtx1060.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce image dimensions or steps
   - Restart the service to clear GPU memory
   - Check for memory leaks

2. **Slow generation times**
   - Verify GPU acceleration is enabled
   - Check system resources
   - Reduce concurrent requests

3. **Service won't start**
   - Verify NVIDIA Docker is installed
   - Check GPU availability
   - Review container logs

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## API Reference

### Request Parameters

| Parameter | Type | Required | Range | Default | Description |
|-----------|------|----------|-------|---------|-------------|
| `prompt` | string | Yes | 1-500 chars | - | Text description of image |
| `steps` | integer | No | 1-50 | 20 | Number of diffusion steps |
| `width` | integer | No | 256-768 (÷64) | 512 | Image width in pixels |
| `height` | integer | No | 256-768 (÷64) | 512 | Image height in pixels |
| `seed` | integer | No | 0-4294967295 | random | Random seed for reproducibility |

### Response Format

```json
{
  "image_base64": "base64-encoded-png-data",
  "metadata": {
    "prompt": "original-prompt",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": 12345,
    "generation_time_seconds": 18.5,
    "model_version": "runwayml/stable-diffusion-v1-5"
  }
}
```

### Error Responses

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | Validation Error | Invalid request parameters |
| 408 | Timeout | Generation took too long |
| 503 | Service Unavailable | GPU memory exhausted or model unavailable |
| 500 | Internal Error | Unexpected server error |

## Development

### Project Structure
```
├── app/
│   ├── api/          # FastAPI routes
│   ├── core/         # Configuration and startup
│   ├── middleware/   # Request/response middleware
│   ├── models/       # Pydantic models
│   ├── services/     # Business logic
│   └── utils/        # Utility functions
├── docs/             # Documentation
├── scripts/          # Utility scripts
├── tests/            # Test suite
├── Dockerfile        # Container definition
└── docker-compose.yml # Service orchestration
```

### Running Tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/test_integration_*.py

# Performance tests
python -m pytest tests/test_integration_performance.py
```

### Building from Source
```bash
# Build Docker image
docker build -t stable-diffusion-api .

# Run locally
docker run --gpus all -p 8000:8000 stable-diffusion-api
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramirez

## Support

- **Documentation**: See the `docs/` directory
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join the community discussions

For detailed API documentation, see [API.md](API.md).
For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).
For troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).