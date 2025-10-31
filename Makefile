.PHONY: setup clean test etl train pipeline evaluate api dashboard

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
RESET := \033[0m # Reset color

help:
	@echo "$(BLUE)Sales Forecasting ML Project$(RESET)"
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(RESET) %s\n", $$1, $$2}'

setup: ## Complete setup: install uv and sync dependencies
	@echo "$(BLUE)Starting complete setup...$(RESET)"
	@echo ""
	@echo "$(BLUE)[1/3] Checking Python version...$(RESET)"
	@python_version=$$(python3 --version 2>&1 | sed 's/Python //'); \
	major=$$(echo $$python_version | cut -d. -f1); \
	minor=$$(echo $$python_version | cut -d. -f2); \
	if [ $$major -lt 3 ] || [ $$major -eq 3 -a $$minor -lt 12 ]; then \
		echo "$(RED)✗ Python 3.12+ is required (found $$python_version)$(RESET)"; \
		echo "$(YELLOW)Install with: brew install pyenv && pyenv install 3.12$(RESET)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✓ Python $$python_version detected$(RESET)"
	@echo ""
	@echo "$(BLUE)[2/3] Checking for uv...$(RESET)"
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "$(YELLOW)uv not found. Installing...$(RESET)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "$(GREEN)✓ uv installed$(RESET)"; \
	else \
		echo "$(GREEN)✓ uv already installed$(RESET)"; \
	fi
	@echo ""
	@echo "$(BLUE)[3/3] Syncing dependencies with uv...$(RESET)"
	@uv sync --all-groups
	@echo "$(GREEN)✓ Dependencies synced$(RESET)"
	@echo ""
	@echo "$(GREEN)✓ Setup complete!$(RESET)"
	@echo "$(YELLOW)Activate the environment with: source .venv/bin/activate$(RESET)"

check-python:
	@echo "$(BLUE)Checking Python version...$(RESET)"
	@python_version=$$(python3 --version 2>&1 | sed 's/Python //'); \
	major=$$(echo $$python_version | cut -d. -f1); \
	minor=$$(echo $$python_version | cut -d. -f2); \
	if [ $$major -lt 3 ] || [ $$major -eq 3 -a $$minor -lt 12 ]; then \
		echo "$(RED)✗ Python 3.12+ is required (found $$python_version)$(RESET)"; \
		echo "$(YELLOW)Install with: brew install pyenv && pyenv install 3.12$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Python $$python_version detected$(RESET)"

install-uv:
	@echo "$(BLUE)Checking for uv...$(RESET)"
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "$(YELLOW)uv not found. Installing...$(RESET)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "$(GREEN)✓ uv installed$(RESET)"; \
	else \
		echo "$(GREEN)✓ uv already installed$$(RESET)"; \
	fi

sync:
	@echo "$(BLUE)Syncing dependencies with uv...$(RESET)"
	@uv sync --all-groups
	@echo "$(GREEN)✓ Dependencies synced$(RESET)"

install: sync

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(RESET)"
	@if [ -d ".venv" ]; then \
		.venv/bin/pytest tests/; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi

clean: ## Remove virtual environment and cache files
	@echo "$(BLUE)Cleaning up...$(RESET)"
	@rm -rf .venv
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf **/__pycache__
	@rm -rf **/*.pyc
	@rm -rf **/*.pyo
	@rm -rf src/*.egg-info
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"

# ============================================================================
# ML Pipeline Commands
# ============================================================================

etl: ## Run ETL pipeline (transform, split, features)
	@echo "$(BLUE)Running ETL pipeline...$(RESET)"
	@if [ -d ".venv" ]; then \
		.venv/bin/python -m src.etl.run_pipeline; \
		echo "$(GREEN)✓ ETL pipeline completed$(RESET)"; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi

train: ## Train Random Forest model
	@echo "$(BLUE)Training model...$(RESET)"
	@if [ -d ".venv" ]; then \
		.venv/bin/python -m src.train.train_model; \
		echo "$(GREEN)✓ Model training completed$(RESET)"; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi

evaluate: ## Evaluate model on test set
	@echo "$(BLUE)Evaluating model...$(RESET)"
	@if [ -d ".venv" ]; then \
		.venv/bin/python -m src.evaluate.evaluate; \
		echo "$(GREEN)✓ Model evaluation completed$(RESET)"; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi

pipeline: ## Run complete ML pipeline (ETL + Train + Evaluate)
	@echo "$(BLUE)Running complete ML pipeline...$(RESET)"
	@$(MAKE) etl
	@$(MAKE) train
	@$(MAKE) evaluate
	@echo "$(GREEN)✓ Complete pipeline finished!$(RESET)"

# ============================================================================
# API & Dashboard Commands
# ============================================================================

api: ## Start FastAPI server
	@echo "$(BLUE)Starting FastAPI server...$(RESET)"
	@echo "$(YELLOW)API will be available at: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)API docs at: http://localhost:8000/docs$(RESET)"
	@echo ""
	@if [ -d ".venv" ]; then \
		uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi

dashboard: ## Start Streamlit dashboard
	@echo "$(BLUE)Starting Streamlit dashboard...$(RESET)"
	@echo "$(YELLOW)Dashboard will be available at: http://localhost:8501$(RESET)"
	@echo ""
	@if [ -d ".venv" ]; then \
		uv run streamlit run src/dashboard/app.py; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi
