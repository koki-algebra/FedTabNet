.PHONY: help up down logs ps attach
.DEFAULT_GOAL := help

up: ## Do docker compose up
	docker compose up -d

down: ## Do docker compose down
	docker compose down

logs: ## Do docker compose logs
	docker compose logs -f

ps: ## Check container status
	docker compose ps

attach: ## Attach to running container
	docker exec -it -w /workspaces fedtabnet bash

help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'