# Deployment Guide

## Overview

This guide covers deploying the Stable Diffusion API in various environments, from local development to production servers.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GTX 1060 6GB or better
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space for models and containers
- **CPU**: 4+ cores recommended

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: nvidia-docker2
- **NVIDIA Drivers**: 470+ with CUDA 11.8 support

## Environment Setup

### 1. Install NVIDIA Drivers

#### Ubuntu/Debian
```bash
# Check current driver
nvidia-smi

# Install latest drivers if needed
sudo apt update
sudo apt install nvidia-driver-470
sudo reboot
```

#### CentOS/RHEL
```bash
# Enable EPEL repository
sudo yum install epel-release

# Install NVIDIA drivers
sudo yum install nvidia-driver nvidia-settings
sudo reboot
```

### 2. Install Docker

#### Ubuntu/Debian
```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### CentOS/RHEL
```bash
# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
```

### 3. Install NVIDIA Docker

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test installation
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 4. Install Docker Compose

```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

## Deployment Configurations

### Development Environment

For local development and testing:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd stable-diffusion-api
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit environment variables:**
   ```bash
   # .env
   CUDA_VISIBLE_DEVICES=0
   MODEL_CACHE_DIR=./models
   MAX_CONCURRENT_REQUESTS=2
   LOG_LEVEL=DEBUG
   API_PORT=8000
   ```

4. **Start the service:**
   ```bash
   docker-compose up -d
   ```

5. **Verify deployment:**
   ```bash
   curl http://localhost:8000/health
   ```

### Production Environment

For production deployment with optimizations:

1. **Create production docker-compose file:**
   ```yaml
   # docker-compose.prod.yml
   version: '3.8'
   
   services:
     stable-diffusion:
       build: .
       ports:
         - "8000:8000"
       environment:
         - CUDA_VISIBLE_DEVICES=0
         - MODEL_CACHE_DIR=/app/models
         - MAX_CONCURRENT_REQUESTS=3
         - LOG_LEVEL=INFO
         - WORKERS=1
       volumes:
         - ./models:/app/models
         - ./logs:/app/logs
       deploy:
         resources:
           limits:
             memory: 10G
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 60s
   
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
       depends_on:
         - stable-diffusion
       restart: unless-stopped
   ```

2. **Configure Nginx reverse proxy:**
   ```nginx
   # nginx.conf
   events {
       worker_connections 1024;
   }
   
   http {
       upstream stable_diffusion {
           server stable-diffusion:8000;
       }
   
       server {
           listen 80;
           server_name your-domain.com;
           
           # Redirect HTTP to HTTPS
           return 301 https://$server_name$request_uri;
       }
   
       server {
           listen 443 ssl http2;
           server_name your-domain.com;
   
           ssl_certificate /etc/nginx/ssl/cert.pem;
           ssl_certificate_key /etc/nginx/ssl/key.pem;
   
           client_max_body_size 10M;
           proxy_read_timeout 120s;
           proxy_connect_timeout 10s;
           proxy_send_timeout 120s;
   
           location / {
               proxy_pass http://stable_diffusion;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
           }
   
           location /health {
               proxy_pass http://stable_diffusion/health;
               access_log off;
           }
       }
   }
   ```

3. **Deploy with production configuration:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Cloud Deployment

#### AWS EC2 with GPU

1. **Launch GPU instance:**
   - Instance type: g4dn.xlarge or p3.2xlarge
   - AMI: Deep Learning AMI (Ubuntu 20.04)
   - Security group: Allow ports 22, 80, 443, 8000

2. **Connect and setup:**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker and NVIDIA Docker (if not pre-installed)
   # Follow installation steps above
   ```

3. **Deploy application:**
   ```bash
   git clone <repository-url>
   cd stable-diffusion-api
   
   # Configure environment
   cp .env.example .env
   nano .env  # Edit as needed
   
   # Start services
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Configure security:**
   ```bash
   # Setup firewall
   sudo ufw enable
   sudo ufw allow ssh
   sudo ufw allow 80
   sudo ufw allow 443
   
   # Setup SSL with Let's Encrypt
   sudo apt install certbot
   sudo certbot certonly --standalone -d your-domain.com
   ```

#### Google Cloud Platform

1. **Create VM with GPU:**
   ```bash
   gcloud compute instances create stable-diffusion-vm \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB \
     --maintenance-policy=TERMINATE \
     --restart-on-failure
   ```

2. **Install NVIDIA drivers:**
   ```bash
   # SSH into instance
   gcloud compute ssh stable-diffusion-vm --zone=us-central1-a
   
   # Install drivers
   curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
   sudo python3 install_gpu_driver.py
   ```

3. **Deploy application** (follow standard deployment steps)

#### Azure VM

1. **Create GPU VM:**
   ```bash
   az vm create \
     --resource-group myResourceGroup \
     --name stable-diffusion-vm \
     --image UbuntuLTS \
     --size Standard_NC6 \
     --admin-username azureuser \
     --generate-ssh-keys
   ```

2. **Install NVIDIA drivers and deploy** (follow standard steps)

## Monitoring and Maintenance

### Health Monitoring

1. **Setup monitoring script:**
   ```bash
   #!/bin/bash
   # monitor.sh
   
   while true; do
     response=$(curl -s http://localhost:8000/health)
     status=$(echo $response | jq -r '.status')
     
     if [ "$status" != "healthy" ]; then
       echo "$(date): Service unhealthy - $response"
       # Send alert (email, Slack, etc.)
     fi
     
     sleep 30
   done
   ```

2. **Setup log rotation:**
   ```bash
   # /etc/logrotate.d/stable-diffusion
   /path/to/logs/*.log {
     daily
     rotate 7
     compress
     delaycompress
     missingok
     notifempty
     create 644 root root
   }
   ```

### Backup and Recovery

1. **Backup models:**
   ```bash
   # Create backup script
   #!/bin/bash
   tar -czf models-backup-$(date +%Y%m%d).tar.gz ./models
   aws s3 cp models-backup-$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
   ```

2. **Database backup** (if using persistent storage):
   ```bash
   docker exec stable-diffusion-db pg_dump -U user database > backup.sql
   ```

### Performance Optimization

1. **GPU memory optimization:**
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1
   
   # Adjust memory settings in environment
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

2. **Container resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 8G
         cpus: '4'
       reservations:
         memory: 4G
         cpus: '2'
   ```

## Security Considerations

### Network Security

1. **Firewall configuration:**
   ```bash
   # Only allow necessary ports
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow ssh
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw enable
   ```

2. **API authentication** (if needed):
   ```python
   # Add API key authentication
   from fastapi import Header, HTTPException
   
   async def verify_api_key(x_api_key: str = Header()):
       if x_api_key != "your-secret-key":
           raise HTTPException(status_code=401, detail="Invalid API key")
   ```

### Container Security

1. **Run as non-root user:**
   ```dockerfile
   # In Dockerfile
   RUN groupadd -r appuser && useradd -r -g appuser appuser
   USER appuser
   ```

2. **Limit container capabilities:**
   ```yaml
   security_opt:
     - no-new-privileges:true
   cap_drop:
     - ALL
   cap_add:
     - CHOWN
     - SETGID
     - SETUID
   ```

## Scaling

### Horizontal Scaling

1. **Load balancer configuration:**
   ```nginx
   upstream stable_diffusion_cluster {
       server stable-diffusion-1:8000;
       server stable-diffusion-2:8000;
       server stable-diffusion-3:8000;
   }
   ```

2. **Multiple GPU setup:**
   ```yaml
   services:
     stable-diffusion-gpu0:
       environment:
         - CUDA_VISIBLE_DEVICES=0
     stable-diffusion-gpu1:
       environment:
         - CUDA_VISIBLE_DEVICES=1
   ```

### Vertical Scaling

1. **Increase container resources:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G
         cpus: '8'
   ```

2. **Optimize model loading:**
   ```python
   # Use model caching and sharing
   pipe.enable_model_cpu_offload()
   pipe.enable_attention_slicing(1)
   ```

## Troubleshooting Deployment

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting steps.

Common deployment issues:
- NVIDIA Docker not properly installed
- Insufficient GPU memory
- Port conflicts
- Permission issues
- Model download failures

## Maintenance Tasks

### Regular Maintenance

1. **Weekly tasks:**
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade
   
   # Clean Docker images
   docker system prune -f
   
   # Check disk space
   df -h
   ```

2. **Monthly tasks:**
   ```bash
   # Update Docker images
   docker-compose pull
   docker-compose up -d
   
   # Review logs
   docker-compose logs --tail=1000 > monthly-logs.txt
   
   # Performance review
   # Check generation times, memory usage, error rates
   ```

3. **Backup schedule:**
   ```bash
   # Setup cron job for daily backups
   0 2 * * * /path/to/backup-script.sh
   ```