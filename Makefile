.PHONY: build up down logs train clean help

# Default target
help:
	@echo "DGA Detection System - Docker Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build      Build all Docker images"
	@echo "  up         Start all services"
	@echo "  down       Stop all services"
	@echo "  logs       View logs from all services"
	@echo "  train      Train ML models (run once before starting)"
	@echo "  clean      Remove all containers, images, and volumes"
	@echo "  restart    Restart all services"
	@echo "  status     Show status of all services"
	@echo ""

# Build all images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d
	@echo ""
	@echo "Services started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

# Stop services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Train models
train:
	@echo "Training ML models..."
	docker-compose --profile training run --rm trainer
	@echo "Training complete! Models saved to ./models/"

# Clean everything
clean:
	docker-compose down -v --rmi all
	@echo "Cleaned all containers, images, and volumes"

# Restart services
restart: down up

# Show status
status:
	docker-compose ps

# Build and start
start: build up

# Development mode - backend only with hot reload
dev-backend:
	cd /home/kernaite/documents/mei_aas_pf && python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Development mode - frontend only with hot reload
dev-frontend:
	cd frontend && npm run dev
