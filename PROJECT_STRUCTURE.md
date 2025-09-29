# Project Structure

This document describes the organization of the Stable Diffusion API project.

## Directory Structure

```
stable-diffusion-api/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── api/                      # API layer (FastAPI routes)
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/                     # Core utilities and configuration
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── models/                   # Pydantic data models
│   │   └── __init__.py
│   ├── services/                 # Business logic and services
│   │   └── __init__.py
│   └── utils/                    # Utility functions
│       └── __init__.py
├── config/                       # Configuration files
│   ├── __init__.py
│   └── settings.py
├── tests/                        # Test files
│   ├── __init__.py
│   └── test_models.py
├── models/                       # Model weights storage (created at runtime)
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker container definition
├── main.py                       # Application entry point
├── pytest.ini                   # Pytest configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Key Components

### Application Layer (`app/`)
- **api/**: FastAPI routes and endpoint definitions
- **core/**: Core utilities, configuration, and logging
- **models/**: Pydantic models for request/response validation
- **services/**: Business logic for image generation and processing
- **utils/**: Helper functions and utilities

### Configuration (`config/`)
- **settings.py**: Application settings using Pydantic BaseSettings

### Containerization
- **Dockerfile**: Multi-stage Docker build with CUDA support
- **docker-compose.yml**: Container orchestration with GPU support
- **.dockerignore**: Files to exclude from Docker build context

### Dependencies
- **requirements.txt**: Python package dependencies with CUDA-compatible versions