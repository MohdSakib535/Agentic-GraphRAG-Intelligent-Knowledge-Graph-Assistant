.PHONY: help up down logs build ps seed migrate test test-unit test-integration lint samples eval clean

COMPOSE ?= docker compose

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

.env:
	cp .env.example .env

up: .env ## Build and start the full stack
	$(COMPOSE) up --build -d

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail logs
	$(COMPOSE) logs -f --tail=100

ps: ## Service status
	$(COMPOSE) ps

build: ## Build images
	$(COMPOSE) build

migrate: ## Run Alembic migrations
	$(COMPOSE) exec backend alembic upgrade head

seed: ## Create demo user (demo@techcorp.com / DemoPassw0rd) and ingest sample documents
	$(COMPOSE) exec backend python -m app.scripts.seed_demo

samples: ## Regenerate the sample corpus
	cd backend && python -m app.scripts.generate_samples

test: ## Run the full test suite (unit + integration when services are reachable)
	cd backend && python -m pytest

test-unit: ## Unit + agent tests only (no external services)
	cd backend && python -m pytest tests/unit tests/evaluation -m "not integration"

test-integration: ## Integration tests against running PostgreSQL / Neo4j / Redis
	cd backend && python -m pytest tests/integration

test-docker: ## Run tests inside the backend container against the compose services
	$(COMPOSE) exec backend python -m pytest

lint: ## Ruff lint (backend, frontend, tests)
	ruff check backend frontend tests

eval: ## Trigger an evaluation run via the API (requires TOKEN env var)
	curl -s -X POST http://localhost:8000/api/v1/evaluation/run -H "Authorization: Bearer $$TOKEN" -H 'Content-Type: application/json' -d '{}'

clean: ## Stop the stack and delete volumes
	$(COMPOSE) down -v
