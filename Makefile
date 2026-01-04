.PHONY: build up down logs train clean help monitoring-up monitoring-down test-smoke test-load test-stress

# Default target
help:
	@echo "DGA Detection System - Docker Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build           Build all Docker images"
	@echo "  up              Start all services"
	@echo "  down            Stop all services"
	@echo "  logs            View logs from all services"
	@echo "  train           Train ML models (run once before starting)"
	@echo "  clean           Remove all containers, images, and volumes"
	@echo "  restart         Restart all services"
	@echo "  status          Show status of all services"
	@echo ""
	@echo "Monitoring:"
	@echo "  monitoring-up   Start Prometheus & Grafana monitoring stack"
	@echo "  monitoring-down Stop monitoring stack"
	@echo "  monitoring-logs View monitoring stack logs"
	@echo ""
	@echo "Load Testing (requires k6: https://k6.io):"
	@echo "  test-smoke      Quick validation test (10 iterations)"
	@echo "  test-load       Load test (5 minutes, 20 VUs)"
	@echo "  test-stress     Stress test (12 minutes, up to 100 VUs)"
	@echo ""

# Build all images
build:
	docker compose build

# Start services
up:
	docker compose up -d
	@echo ""
	@echo "Services started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

# Stop services
down:
	docker compose down

# View logs
logs:
	docker compose logs -f

# Train models
train:
	@echo "Training ML models..."
	docker compose --profile training run --rm trainer
	@echo "Training complete! Models saved to ./models/"

# Clean everything
clean:
	docker compose down -v --rmi all
	@echo "Cleaned all containers, images, and volumes"

# Restart services
restart: down up

# Show status
status:
	docker compose ps

# Build and start
start: build up

# Development mode - backend only with hot reload
dev-backend:
	cd /home/kernaite/documents/mei_aas_pf && python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Development mode - frontend only with hot reload
dev-frontend:
	cd frontend && npm run dev

# ============== Monitoring ==============

# Start monitoring stack (Prometheus + Grafana)
monitoring-up:
	docker compose --profile monitoring up -d prometheus grafana
	@echo ""
	@echo "Monitoring stack started!"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3001 (admin/admin)"
	@echo "  API Metrics: http://localhost:8000/metrics"

# Stop monitoring stack
monitoring-down:
	docker compose --profile monitoring down

# View monitoring logs
monitoring-logs:
	docker compose --profile monitoring logs -f prometheus grafana

# ============== Load Testing ==============

# Smoke test - quick validation
test-smoke:
	@echo "Running smoke test..."
	docker run --rm -i --network host \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/tests/load:/scripts \
		grafana/k6 run /scripts/k6-smoke-test.js

# Load test - standard load
test-load:
	@echo "Running load test (5 minutes)..."
	docker run --rm -i --network host \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/tests/load:/scripts \
		grafana/k6 run --env TEST_TYPE=load /scripts/k6-load-test.js

# Stress test - find breaking point
test-stress:
	@echo "Running stress test (12 minutes, up to 100 VUs)..."
	docker run --rm -i --network host \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/tests/load:/scripts \
		grafana/k6 run --env TEST_TYPE=stress /scripts/k6-load-test.js

# Docker-based k6 test (no local k6 installation required)
test-docker:
	@echo "Running load test via Docker..."
	docker run --rm -i --network host \
		-v $(PWD)/tests/load:/scripts \
		grafana/k6 run /scripts/k6-smoke-test.js
