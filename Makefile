# -----------------------------
# Variables
# -----------------------------
IMAGE_NAME = llmdataparser
CONTAINER_NAME = llmdataparser
VERSION = latest

# -----------------------------
# Docker Basic Commands
# -----------------------------
# Build the Docker image
build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

# Run the container
run:
	docker run -d -p 7860:7860 --name $(CONTAINER_NAME) $(IMAGE_NAME):$(VERSION)

# Stop the container
stop:
	docker stop $(CONTAINER_NAME)

# Remove the container
rm:
	docker rm $(CONTAINER_NAME)

# Remove the image
rmi:
	docker rmi $(IMAGE_NAME):$(VERSION)

# -----------------------------
# Docker Compose Commands
# -----------------------------
# Start with docker-compose (development)
compose-up:
	docker compose up -d

# Stop and remove containers
compose-down:
	docker compose down

# View logs
compose-logs:
	docker compose logs -f

# Rebuild containers
compose-build:
	docker compose build

# Restart containers
compose-restart:
	docker compose restart

# -----------------------------
# Convenience Commands
# -----------------------------
# Build and run with docker
up: build run

# Stop and remove container
down: stop rm

# Clean everything
clean: stop rm rmi

# -----------------------------
# Monitoring Commands
# -----------------------------
# Show container logs
logs:
	docker logs $(CONTAINER_NAME)

# Follow container logs
logs-follow:
	docker logs -f $(CONTAINER_NAME)

# Show container status
status:
	docker ps -a | grep $(CONTAINER_NAME)

# Enter container shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# -----------------------------
# Production Commands
# -----------------------------
# Test nginx configuration (for production use)
nginx-test:
	docker compose run --rm nginx nginx -t

# Start with nginx test (for production use)
compose-up-prod: nginx-test compose-up

# -----------------------------
# Security Commands
# -----------------------------
security-check:
	@echo "Checking nginx configuration..."
	docker compose run --rm nginx nginx -t
	@echo "Checking exposed ports..."
	docker compose config | grep -E "ports:|127.0.0.1"

# Ensure all targets are treated as commands, not files
.PHONY: build run stop rm rmi clean up down logs shell \
        compose-up compose-down compose-logs compose-build compose-restart \
        nginx-test status logs-follow compose-up-prod
