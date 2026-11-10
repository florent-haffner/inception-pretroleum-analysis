PROJECT     := inception-petroleum-analysis
PYTHON_VER  := 3.13
ENV_DIR     := env
VENV        := .venv
PY          := $(VENV)/bin/python
IMAGE       := $(PROJECT)
MODEL_DIR   := $(CURDIR)/model

.DEFAULT_GOAL := help
.PHONY: help venv venv-dev test test-all conda-env conda-update run run-conda docker-build docker-run clean clean-models

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- Local Python (virtualenv) ---------------------------------------------

venv: $(VENV)/.installed ## Create .venv and install requirements (uses uv if available)

$(VENV)/.installed: $(ENV_DIR)/requirements.txt
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python $(PYTHON_VER) $(VENV) && uv pip install --python $(PY) -r $<; \
	else \
		python$(PYTHON_VER) -m venv $(VENV) && $(PY) -m pip install -U pip && $(PY) -m pip install -r $<; \
	fi
	@touch $@

run: venv ## Train/evaluate the models with the local .venv
	$(PY) -m scripts.models-calibration

venv-dev: $(VENV)/.installed-dev

$(VENV)/.installed-dev: $(ENV_DIR)/requirements-dev.txt $(VENV)/.installed
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(PY) -r $<; \
	else \
		$(PY) -m pip install -r $<; \
	fi
	@touch $@

test: venv-dev ## Run the fast tests (synthetic data, ~1 min)
	$(PY) -m pytest tests -m "not slow"

test-all: venv-dev ## Run all tests, including short trainings on the real dataset
	$(PY) -m pytest tests

## --- Conda -----------------------------------------------------------------

conda-env: ## Create the conda environment
	conda env create -f $(ENV_DIR)/environment.yml

conda-update: ## Update the conda environment after a requirements change
	conda env update -f $(ENV_DIR)/environment.yml --prune

run-conda: ## Train/evaluate the models with the conda environment
	conda run --no-capture-output -n $(PROJECT) python -m scripts.models-calibration

## --- Docker ----------------------------------------------------------------

docker-build: ## Build the Docker image (CPU)
	docker build -f $(ENV_DIR)/Dockerfile -t $(IMAGE) .

docker-run: ## Run the pipeline in Docker; weights are kept in ./model
	mkdir -p $(MODEL_DIR)
	docker run -it --rm --user "$$(id -u):$$(id -g)" -v $(MODEL_DIR):/app/model $(IMAGE)

## --- Cleaning --------------------------------------------------------------

clean: ## Remove caches and the local virtualenv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(VENV)

clean-models: ## Remove trained model weights
	rm -rf $(MODEL_DIR)
