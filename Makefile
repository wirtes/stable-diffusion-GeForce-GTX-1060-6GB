# Makefile for Stable Diffusion API Docker operations

.PHONY: help build up down logs clean dev prod test

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build the Docker image"
	@echo "  dev       - Start development environment"
	@echo "  prod      - Start production environment"
	@echo "  up        - Start services (development by default)"
	@echo "  down      - Stop and remove containers"
	@echo "  logs      - Show container logs"
	@echo "  clean     - Remove containers, networks, and volumes"
	@echo "  test      - Run tests in container"
	@echo "  shell     - Open shell in running container"

# Build the Docker image
build:
	docker-compose build

# Start development environment
dev:
	docker-compose --profile development up -d

# Start production environment
prod:
	docker-compose --profile production up -d

# Start production with nginx
prod-nginx:
	docker-compose --profile production --profile nginx up -d

# Start services (development by default)
up:
	docker-compose up -d

# Stop and remove containers
down:
	docker-compose --profile development --profile production --profile nginx down

# Show container logs
logs:
	docker-compose logs -f

# Show logs for specific service
logs-api:
	docker-compose logs -f stable-diffusion-api-dev stable-diffusion-api

# Remove everything (containers, networks, volumes)
clean:
	docker-compose --profile development --profile production --profile nginx down -v --remove-orphans
	docker system prune -f

# Run tests in container
test:
	docker-compose exec stable-diffusion-api-dev python -m pytest tests/ -v

# Open shell in running development container
shell:
	docker-compose exec stable-diffusion-api-dev /bin/bash

# Check GPU availability in container
gpu-check:
	docker-compose exec stable-diffusion-api-dev nvidia-smi

# Restart services
restart:
	docker-compose restart

# Pull latest images
pull:
	docker-compose pull