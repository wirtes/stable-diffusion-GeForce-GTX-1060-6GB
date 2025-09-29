# Stable Diffusion API Documentation

Welcome to the Stable Diffusion API documentation. This service provides a REST API for generating images from text descriptions using AI, optimized for NVIDIA GTX 1060 6GB.

## 📚 Documentation Index

### Getting Started
- **[README](README.md)** - Quick start guide and overview
- **[API Reference](API.md)** - Complete API documentation with examples
- **[Deployment Guide](DEPLOYMENT.md)** - Installation and deployment instructions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

### Examples and Usage
- **[Python Client](examples/python_client.py)** - Complete Python client with examples
- **[JavaScript Client](examples/javascript_client.js)** - Node.js and browser JavaScript examples
- **[cURL Examples](examples/curl_examples.sh)** - Command-line usage examples
- **[Web Interface](examples/web_interface.html)** - Interactive HTML interface

## 🚀 Quick Start

1. **Start the service:**
   ```bash
   docker-compose up -d
   ```

2. **Test the API:**
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A beautiful sunset", "steps": 20}'
   ```

3. **Check service health:**
   ```bash
   curl http://localhost:8000/health
   ```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate image from text prompt |
| `/health` | GET | Check service health and GPU status |

## 📋 Request Parameters

| Parameter | Type | Required | Range | Default | Description |
|-----------|------|----------|-------|---------|-------------|
| `prompt` | string | Yes | 1-500 chars | - | Text description |
| `steps` | integer | No | 1-50 | 20 | Diffusion steps |
| `width` | integer | No | 256-768 (÷64) | 512 | Image width |
| `height` | integer | No | 256-768 (÷64) | 512 | Image height |
| `seed` | integer | No | 0-4294967295 | random | Random seed |

## 🎯 Example Response

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...",
  "metadata": {
    "prompt": "A beautiful sunset",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": 12345,
    "generation_time_seconds": 18.5,
    "model_version": "runwayml/stable-diffusion-v1-5"
  }
}
```

## 🛠️ Client Libraries

### Python
```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post('http://localhost:8000/generate', json={
    'prompt': 'A cute robot in a garden',
    'steps': 20
})

data = response.json()
image_bytes = base64.b64decode(data['image_base64'])
image = Image.open(BytesIO(image_bytes))
image.save('generated.png')
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'A magical forest',
    steps: 25
  })
});

const data = await response.json();
console.log(`Generated in ${data.metadata.generation_time_seconds}s`);
```

### cURL
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cozy cabin", "steps": 20}' \
  | jq -r '.image_base64' | base64 -d > image.png
```

## 🔍 Monitoring

### Health Check
```bash
curl http://localhost:8000/health | jq '.'
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Performance Benchmarking
```bash
python scripts/benchmark_performance.py
```

## ⚡ Performance

| Image Size | Steps | Generation Time | Memory Usage |
|------------|-------|-----------------|--------------|
| 512×512    | 20    | 15-20 seconds   | ~4GB VRAM    |
| 768×512    | 25    | 25-30 seconds   | ~5GB VRAM    |
| 768×768    | 30    | 35-45 seconds   | ~5.5GB VRAM  |

## 🐛 Common Issues

1. **CUDA out of memory** - Reduce image size or restart service
2. **Slow generation** - Check GPU acceleration is enabled
3. **Service won't start** - Verify NVIDIA Docker is installed

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## 🚀 Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 📞 Support

- **Documentation**: Browse the files in this directory
- **Issues**: Report bugs on GitHub
- **Examples**: Check the `examples/` directory

---

**Next Steps:**
- Read the [API Reference](API.md) for complete documentation
- Try the [Web Interface](examples/web_interface.html) for interactive testing
- Follow the [Deployment Guide](DEPLOYMENT.md) for production setup