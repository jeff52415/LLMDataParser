# Variables
IMAGE_NAME = llmdataparser
CONTAINER_NAME = llmdataparser
VERSION = latest

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

# Clean everything
clean: stop rm rmi

# Build and run
up: build run

# Stop and remove container
down: stop rm

# Show container logs
logs:
	docker logs $(CONTAINER_NAME)

# Enter container shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Optional: command to check container status
status:
	docker ps -a | grep $(CONTAINER_NAME)

logs-follow:
	docker logs -f $(CONTAINER_NAME)

.PHONY: build run stop rm rmi clean up down logs shell
