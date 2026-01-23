# TRM Code Repair - Makefile
# Common commands for development, data collection, and training

.PHONY: help install dev-install test lint format clean
.PHONY: collect-github collect-stackoverflow collect-synthetic collect-all
.PHONY: process-data validate-data build-vocab
.PHONY: train train-curriculum evaluate
.PHONY: docker-build docker-push serve

PYTHON := python3
PIP := pip3
DATA_DIR := ./data
MODEL_DIR := ./models
CONFIG_DIR := ./configs

# =============================================================================
# Help
# =============================================================================

help:
	@echo "TRM Code Repair - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make dev-install      Install development dependencies"
	@echo "  make setup-env        Create .env from template"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run test suite"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "Data Collection:"
	@echo "  make collect-github   Collect from GitHub commits"
	@echo "  make collect-so       Collect from Stack Overflow"
	@echo "  make collect-synthetic Generate synthetic bugs"
	@echo "  make collect-all      Run all collectors"
	@echo ""
	@echo "Data Processing:"
	@echo "  make process-data     Process raw data"
	@echo "  make validate-data    Validate processed data"
	@echo "  make build-vocab      Build BPE vocabulary"
	@echo ""
	@echo "Training:"
	@echo "  make train            Train model"
	@echo "  make train-curriculum Train with curriculum"
	@echo "  make evaluate         Evaluate model"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker-build     Build Docker images"
	@echo "  make serve            Start inference server"

# =============================================================================
# Setup
# =============================================================================

install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev,training]"
	pre-commit install

setup-env:
	@if [ ! -f .env ]; then \
		cp configs/environment-template.txt .env; \
		echo "Created .env file. Please edit with your credentials."; \
	else \
		echo ".env already exists"; \
	fi

# =============================================================================
# Development
# =============================================================================

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest tests/unit/ -v -x

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/ --ignore-missing-imports

format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Data Collection
# =============================================================================

collect-github:
	$(PYTHON) scripts/collect_data.py github \
		--config $(CONFIG_DIR)/data/collection.yaml \
		--output $(DATA_DIR)/raw/github \
		--limit 100000

collect-so:
	$(PYTHON) scripts/collect_data.py stackoverflow \
		--config $(CONFIG_DIR)/data/collection.yaml \
		--output $(DATA_DIR)/raw/stackoverflow \
		--limit 50000

collect-synthetic:
	$(PYTHON) scripts/collect_data.py synthetic \
		--config $(CONFIG_DIR)/data/collection.yaml \
		--output $(DATA_DIR)/raw/synthetic \
		--count 500000

collect-linters:
	$(PYTHON) scripts/collect_data.py linters \
		--config $(CONFIG_DIR)/data/collection.yaml \
		--output $(DATA_DIR)/raw/linters \
		--input $(DATA_DIR)/raw/github/repos

collect-all: collect-github collect-so collect-synthetic collect-linters

# =============================================================================
# Data Processing
# =============================================================================

process-data:
	$(PYTHON) scripts/process_data.py \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed \
		--config $(CONFIG_DIR)/data/processing.yaml \
		--workers 8

validate-data:
	$(PYTHON) scripts/validate_data.py \
		--input $(DATA_DIR)/processed \
		--config $(CONFIG_DIR)/data/validation.yaml \
		--report $(DATA_DIR)/validation_report.json

build-vocab:
	$(PYTHON) scripts/build_vocabulary.py \
		--input $(DATA_DIR)/processed \
		--output $(DATA_DIR)/vocabulary \
		--vocab-size 32768

deduplicate:
	$(PYTHON) scripts/deduplicate.py \
		--input $(DATA_DIR)/processed \
		--output $(DATA_DIR)/processed_dedup \
		--threshold 0.9

split-data:
	$(PYTHON) scripts/split_data.py \
		--input $(DATA_DIR)/processed_dedup \
		--output $(DATA_DIR)/final \
		--train-ratio 0.9 \
		--val-ratio 0.05 \
		--test-ratio 0.05

# =============================================================================
# Training
# =============================================================================

train:
	$(PYTHON) scripts/train.py \
		--config $(CONFIG_DIR)/training/base.yaml \
		--data $(DATA_DIR)/final \
		--output $(MODEL_DIR)/base \
		--wandb

train-curriculum:
	$(PYTHON) scripts/train.py \
		--config $(CONFIG_DIR)/training/curriculum.yaml \
		--data $(DATA_DIR)/final \
		--output $(MODEL_DIR)/curriculum \
		--curriculum \
		--wandb

train-distributed:
	torchrun --nproc_per_node=4 scripts/train.py \
		--config $(CONFIG_DIR)/training/distributed.yaml \
		--data $(DATA_DIR)/final \
		--output $(MODEL_DIR)/distributed \
		--distributed

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--model $(MODEL_DIR)/best \
		--data $(DATA_DIR)/final/test \
		--output $(MODEL_DIR)/evaluation_results.json

benchmark:
	$(PYTHON) scripts/benchmark.py \
		--model $(MODEL_DIR)/best \
		--benchmarks $(DATA_DIR)/benchmarks \
		--output $(MODEL_DIR)/benchmark_results.json

# =============================================================================
# Deployment
# =============================================================================

docker-build:
	docker build -f docker/Dockerfile.inference -t trm-inference:latest .

docker-push:
	docker tag trm-inference:latest $(DOCKER_REGISTRY)/trm-inference:latest
	docker push $(DOCKER_REGISTRY)/trm-inference:latest

serve:
	$(PYTHON) -m src.inference.server \
		--model $(MODEL_DIR)/best \
		--port 8000

serve-docker:
	docker run -p 8000:8000 -v $(MODEL_DIR):/models trm-inference:latest

# =============================================================================
# Utilities
# =============================================================================

check-env:
	@$(PYTHON) -c "from dotenv import load_dotenv; load_dotenv(); import os; \
		required = ['GITHUB_TOKEN']; \
		missing = [k for k in required if not os.getenv(k)]; \
		print('Missing:', missing) if missing else print('All required env vars set')"

stats:
	@echo "Data Statistics:"
	@find $(DATA_DIR)/processed -name "*.parquet" | wc -l | xargs -I {} echo "  Parquet shards: {}"
	@du -sh $(DATA_DIR)/processed 2>/dev/null || echo "  No processed data yet"
	@echo ""
	@echo "Model Statistics:"
	@ls -la $(MODEL_DIR)/*/checkpoint-* 2>/dev/null | tail -5 || echo "  No checkpoints yet"

gpu-status:
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

watch-training:
	tensorboard --logdir $(MODEL_DIR)/runs --port 6006
