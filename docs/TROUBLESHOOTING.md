# Troubleshooting Guide

## Common Issues and Solutions

### 1. Service Won't Start

#### Problem: Container fails to start
**Symptoms:**
- Docker container exits immediately
- Error messages about CUDA or GPU

**Solutions:**

1. **Check NVIDIA Docker runtime:**
   ```bash
   # Verify nvidia-docker is installed
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   
   # If this fails, install nvidia-docker2:
   # Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Check GPU availability:**
   ```bash
   # Check if GPU is detected
   nvidia-smi
   
   # Check CUDA version compatibility
   nvcc --version
   ```

3. **Verify Docker Compose configuration:**
   ```bash
   # Make sure docker-compose.yml has runtime: nvidia
   docker-compose config
   ```

#### Problem: Model download fails
**Symptoms:**
- Long startup time followed by failure
- Network timeout errors
- "Model not found" errors

**Solutions:**

1. **Check internet connection:**
   ```bash
   # Test connectivity to Hugging Face
   curl -I https://huggingface.co/runwayml/stable-diffusion-v1-5
   ```

2. **Pre-download model:**
   ```bash
   # Download model manually
   python -c "
   from diffusers import StableDiffusionPipeline
   pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
   pipe.save_pretrained('./models/stable-diffusion-v1-5')
   "
   ```

3. **Use local model cache:**
   ```bash
   # Mount local model directory
   docker run -v ./models:/app/models your-image
   ```

### 2. GPU Memory Issues

#### Problem: CUDA out of memory errors
**Symptoms:**
- "RuntimeError: CUDA out of memory"
- Service becomes unresponsive
- Generation fails after working initially

**Solutions:**

1. **Check GPU memory usage:**
   ```bash
   # Monitor GPU memory
   watch -n 1 nvidia-smi
   ```

2. **Reduce memory usage:**
   ```python
   # In your configuration, ensure these optimizations are enabled:
   pipe.enable_attention_slicing(1)
   pipe.enable_sequential_cpu_offload()
   pipe.enable_model_cpu_offload()
   ```

3. **Lower image resolution:**
   ```json
   {
     "prompt": "your prompt",
     "width": 512,
     "height": 512,
     "steps": 20
   }
   ```

4. **Restart service to clear memory:**
   ```bash
   docker-compose restart
   ```

#### Problem: Memory leaks over time
**Symptoms:**
- GPU memory usage increases with each request
- Eventually runs out of memory
- Performance degrades over time

**Solutions:**

1. **Enable garbage collection:**
   ```python
   import gc
   import torch
   
   # After each generation
   gc.collect()
   torch.cuda.empty_cache()
   ```

2. **Monitor memory usage:**
   ```bash
   # Check memory usage endpoint
   curl http://localhost:8000/health
   ```

3. **Restart service periodically:**
   ```bash
   # Add to crontab for automatic restart
   0 */6 * * * docker-compose restart
   ```

### 3. Performance Issues

#### Problem: Slow generation times
**Symptoms:**
- Generation takes longer than expected (>30 seconds)
- Timeouts on requests
- High CPU usage

**Solutions:**

1. **Verify GPU acceleration:**
   ```bash
   # Check if CUDA is being used
   docker logs your-container-name | grep -i cuda
   ```

2. **Optimize generation parameters:**
   ```json
   {
     "prompt": "your prompt",
     "steps": 20,
     "width": 512,
     "height": 512
   }
   ```

3. **Check system resources:**
   ```bash
   # Monitor system resources
   htop
   iotop
   ```

4. **Reduce concurrent requests:**
   ```bash
   # Set MAX_CONCURRENT_REQUESTS=1 in environment
   export MAX_CONCURRENT_REQUESTS=1
   ```

#### Problem: High memory usage on host
**Symptoms:**
- System becomes slow
- Swap usage increases
- Other applications crash

**Solutions:**

1. **Limit container memory:**
   ```yaml
   # In docker-compose.yml
   services:
     stable-diffusion:
       deploy:
         resources:
           limits:
             memory: 8G
   ```

2. **Monitor host memory:**
   ```bash
   free -h
   ```

### 4. API Request Issues

#### Problem: Validation errors
**Symptoms:**
- 400 Bad Request responses
- "Validation error" messages

**Common validation issues and fixes:**

1. **Dimensions not multiple of 64:**
   ```json
   // Wrong
   {"width": 500, "height": 500}
   
   // Correct
   {"width": 512, "height": 512}
   ```

2. **Steps out of range:**
   ```json
   // Wrong
   {"steps": 100}
   
   // Correct
   {"steps": 30}
   ```

3. **Empty or too long prompt:**
   ```json
   // Wrong
   {"prompt": ""}
   
   // Correct
   {"prompt": "A beautiful landscape"}
   ```

#### Problem: Request timeouts
**Symptoms:**
- 408 Request Timeout
- Requests hang indefinitely
- Client timeout errors

**Solutions:**

1. **Increase client timeout:**
   ```python
   # Python requests
   response = requests.post(url, json=data, timeout=120)
   ```

2. **Reduce generation complexity:**
   ```json
   {
     "prompt": "simple prompt",
     "steps": 15,
     "width": 512,
     "height": 512
   }
   ```

3. **Check service health:**
   ```bash
   curl http://localhost:8000/health
   ```

### 5. Network and Connectivity Issues

#### Problem: Connection refused
**Symptoms:**
- "Connection refused" errors
- Cannot reach API endpoint
- Service appears to be down

**Solutions:**

1. **Check if service is running:**
   ```bash
   docker-compose ps
   docker logs stable-diffusion-api
   ```

2. **Verify port mapping:**
   ```bash
   # Check if port 8000 is open
   netstat -tlnp | grep 8000
   ```

3. **Test local connectivity:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Check firewall settings:**
   ```bash
   # Ubuntu/Debian
   sudo ufw status
   
   # Allow port if needed
   sudo ufw allow 8000
   ```

#### Problem: Slow API responses
**Symptoms:**
- Long response times
- Intermittent timeouts
- Network-related errors

**Solutions:**

1. **Use local network:**
   ```bash
   # Avoid using external IPs for local testing
   curl http://localhost:8000/generate
   # Instead of
   curl http://your-external-ip:8000/generate
   ```

2. **Check network latency:**
   ```bash
   ping localhost
   ```

### 6. Docker and Container Issues

#### Problem: Permission denied errors
**Symptoms:**
- Cannot access GPU from container
- File permission errors
- Docker daemon errors

**Solutions:**

1. **Add user to docker group:**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Fix file permissions:**
   ```bash
   sudo chown -R $USER:$USER ./models
   sudo chmod -R 755 ./models
   ```

3. **Check Docker daemon:**
   ```bash
   sudo systemctl status docker
   sudo systemctl start docker
   ```

#### Problem: Container resource limits
**Symptoms:**
- Container killed by OOM killer
- Performance degradation
- Unexpected container restarts

**Solutions:**

1. **Increase memory limits:**
   ```yaml
   # docker-compose.yml
   services:
     stable-diffusion:
       deploy:
         resources:
           limits:
             memory: 12G
   ```

2. **Monitor container resources:**
   ```bash
   docker stats
   ```

## Diagnostic Commands

### System Information
```bash
# GPU information
nvidia-smi
nvcc --version

# System resources
free -h
df -h
lscpu

# Docker information
docker version
docker-compose version
```

### Service Diagnostics
```bash
# Container logs
docker-compose logs -f

# Container status
docker-compose ps

# Health check
curl http://localhost:8000/health | jq '.'

# Performance test
time curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "steps": 10}'
```

### Model and GPU Diagnostics
```bash
# Test CUDA in container
docker exec -it your-container python -c "import torch; print(torch.cuda.is_available())"

# Check model loading
docker exec -it your-container python -c "
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
print('Model loaded successfully')
"
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs:**
   ```bash
   docker-compose logs --tail=100
   ```

2. **Gather system information:**
   ```bash
   # Create a diagnostic report
   echo "=== System Info ===" > diagnostic.txt
   uname -a >> diagnostic.txt
   nvidia-smi >> diagnostic.txt
   docker version >> diagnostic.txt
   echo "=== Container Logs ===" >> diagnostic.txt
   docker-compose logs --tail=50 >> diagnostic.txt
   ```

3. **Test with minimal configuration:**
   ```json
   {
     "prompt": "test",
     "steps": 10,
     "width": 512,
     "height": 512
   }
   ```

4. **Check GitHub issues** for similar problems and solutions.

5. **Provide detailed information** when reporting issues:
   - Operating system and version
   - GPU model and driver version
   - Docker and docker-compose versions
   - Complete error messages
   - Steps to reproduce the issue