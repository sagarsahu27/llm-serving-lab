# LLM serving lab - common tasks. Run on the GPU VM unless noted.
# `make help` lists targets.

SHELL := /bin/bash
COMPOSE := docker compose -f compose/docker-compose.yml --env-file compose/.env
COMPOSE_ALL := docker compose -f compose/docker-compose.yml --env-file compose/.env --env-file compose/all.env
PY ?= .venv/bin/python
BACKEND ?= vllm
ALIAS ?= $(or $(shell grep -E '^MODEL_ALIAS=' compose/.env 2>/dev/null | cut -d= -f2-),qwen2.5-1.5b)
CONCURRENCY ?= 1,2,4,8,16,32
GATEWAY ?= http://127.0.0.1:8000
BACKENDS := vllm sglang tgi ollama

.DEFAULT_GOAL := help
.PHONY: help venv lint format test env up-vllm up-sglang up-tgi up-ollama up-all down ps logs \
	gpu smoke bench bench-all report infra-build deploy destroy

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create .venv with gateway, bench and dev dependencies
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt

lint: ## Ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Auto-format with ruff
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test: ## Run unit tests (no GPU needed)
	$(PY) -m pytest -q

env: ## Create compose/.env from the example (won't overwrite)
	@test -f compose/.env && echo "compose/.env exists" || (cp compose/.env.example compose/.env && echo "created compose/.env")

up-vllm up-sglang up-tgi: up-%: env ## Start gateway + one backend (up-vllm, up-sglang, up-tgi)
	$(COMPOSE) --profile $* up -d --build --wait gateway $*

up-ollama: env ## Start gateway + Ollama and pull the GGUF model
	$(COMPOSE) --profile ollama up -d --build --wait gateway ollama
	$(COMPOSE) --profile ollama run --rm ollama-pull

up-all: env ## Start all backends on one GPU (reduced memory fractions; demo only)
	$(COMPOSE_ALL) up -d --build --wait gateway
	@for b in vllm sglang tgi ollama; do \
		echo "starting $$b"; \
		$(COMPOSE_ALL) --profile $$b up -d --wait $$b || exit 1; \
	done
	$(COMPOSE_ALL) --profile ollama run --rm ollama-pull

down: ## Stop and remove all containers (volumes/caches are kept)
	$(COMPOSE) --profile all down

ps: ## Show container status
	$(COMPOSE) --profile all ps

logs: ## Follow logs (BACKEND=vllm|sglang|tgi|ollama|gateway)
	$(COMPOSE) --profile all logs -f --tail=200 $(BACKEND)

gpu: ## Show GPU utilisation
	nvidia-smi

smoke: ## Smoke-test the gateway and every healthy backend
	./scripts/smoke_test.sh $(GATEWAY)

bench: ## Benchmark one running backend: make bench BACKEND=vllm
	$(PY) -m bench run --base-url $(GATEWAY)/v1 --model $(BACKEND)/$(ALIAS) --label $(BACKEND) \
		--concurrency $(CONCURRENCY) --out results/manual/$(BACKEND)

bench-all: ## Benchmark each backend alone, then build a combined report
	./scripts/run_benchmark.sh

report: ## Combine all results/manual runs into results/manual/report
	$(PY) -m bench report results/manual --out results/manual/report

infra-build: ## Compile Bicep (local machine)
	az bicep build --file infra/main.bicep --stdout > /dev/null && echo "bicep OK"

deploy: ## Provision the Azure GPU VM (local machine; see infra/deploy.sh)
	./infra/deploy.sh up

destroy: ## Delete the Azure resource group (local machine)
	./infra/deploy.sh down
