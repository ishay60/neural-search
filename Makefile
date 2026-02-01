.PHONY: install dev test lint format run docker-build docker-up docker-down clean help

# Default target
help:
	@echo "Neural Search - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linter (ruff)"
	@echo "  format       Format code (ruff)"
	@echo "  run          Run development server"
	@echo "  worker       Run Celery worker"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start Docker Compose stack"
	@echo "  docker-down  Stop Docker Compose stack"
	@echo "  clean        Clean build artifacts"

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/neural_search --cov-report=term-missing --cov-report=html

# Linting and formatting
lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Type checking
typecheck:
	mypy src/neural_search

# Running
run:
	uvicorn neural_search.main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A neural_search.workers.celery_app worker --loglevel=info --queues=indexing,batch

beat:
	celery -A neural_search.workers.celery_app beat --loglevel=info

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Download models
download-models:
	python scripts/download_models.py

# Seed sample data
seed-data:
	python scripts/seed_data.py

# Benchmark
benchmark:
	python benchmarks/embedding_speed.py
	python benchmarks/search_accuracy.py
