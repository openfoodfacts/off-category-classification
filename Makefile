SHELL = /bin/bash

PYTHON = python3.9


.DEFAULT_GOAL := help

.PHONY: check-system
check-system: ## Check system for necessary tools
	@echo ">>> Checking python..."
	@${PYTHON} --version || { echo ">>> Python 3.9 must be installed !"; exit 1; }
	${PYTHON} -m ensurepip --upgrade
	@echo ">>> OK."
	@echo ""

.PHONY: venv
venv: check-system ## Create virtual environment with venv
	@echo ">>> Creating virtual environment : '$(PWD)/env' ..."
	${PYTHON} -m venv env
	@echo ">>> OK."
	@echo ""

	@echo "!!! Activate with : 'source env/bin/activate' !!!"
	@echo ""

.PHONY: clean-venv
clean-venv: ## Remove virtual environment
	@echo ">>> Removing virtual environment : '$(PWD)/env' ..."
	rm -rf env
	@echo ">>> OK."
	@echo ""

.PHONY: check-venv
check-venv: ## Check that virtual environment is active
	@echo ">>> Checking that this project's virtual environment is active : '$(PWD)/env' ..."
	@${PYTHON} -c "import sys; sys.exit(0) if (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)) else sys.exit(1)" || { \
		echo ">>> ERROR : Virtual environment must be active !"; \
		echo ">>> Activate with : 'source env/bin/activate' ."; \
		echo ""; \
		exit 1; \
	}
	@test "$(VIRTUAL_ENV)" = "$(PWD)/env" || { \
		echo ">>> ERROR : Virtual environment must be '$(PWD)/env' !"; \
		echo ">>> Deactivate active venv ('$(VIRTUAL_ENV)') with : 'deactivate' ."; \
		echo ">>> Activate this project's venv with : 'source env/bin/activate' ."; \
		echo ""; \
		exit 1; \
	}
	@echo ">>> OK."
	@echo ""

requirements-dev.txt: check-system check-venv ## Create requirements-dev.txt file
	@echo ">>> Creating 'requirements-dev.txt' file..."
	pip install --upgrade pip
	pip install --upgrade isort black "black[jupyter]" flake8 bandit mypy \
		pytest pytest-cov
	pip freeze | grep -v "pkg_resources" > requirements-dev.txt
	@echo ">>> OK."
	@echo ""

requirements.txt: check-system check-venv ## Create requirements.txt file
	@echo ">>> Creating 'requirements.txt' file..."
	pip install --upgrade jupyterlab ipykernel ipywidgets widgetsnbextension \
		codecarbon graphviz python-dotenv requests matplotlib seaborn plotly bokeh \
		dtale lux-api pandas-profiling autoviz great_expectations popmon pyod \
		numpy statsmodels pandas modin[ray] sklearn torch tensorflow \
		azure-cognitiveservices-language-luis
	pip freeze | grep -v "pkg_resources" > requirements.txt
	@echo ">>> OK."
	@echo ""

.PHONY: deps-dev
deps-dev: check-system check-venv requirements-dev.txt ## Install dependencies with pip
	@echo ">>> Installing dev dependancies from 'requirements-dev.txt' file..."
	pip install -r requirements-dev.txt
	@echo ">>> OK."
	@echo ""

.PHONY: deps
deps: check-system check-venv requirements.txt ## Install dependencies with pip
	@echo ">>> Installing dependancies from 'requirements.txt' file..."
	pip install -r requirements.txt
	@echo ">>> OK."
	@echo ""

	@echo ">>> Installing JupyterLab Extensions..."
	# jupyter labextension install @jupyter-widgets/jupyterlab-manager
	# jupyter labextension install jupyterlab-plotly
	@echo ">>> OK."
	@echo ""

.PHONY: install
install: check-system check-venv deps-dev deps ## Check system and venv, and install dependencies

.PHONY: isort
isort: ## Sort imports with Isort
	@echo ">>> Sorting imports..."
	isort src/ tests/
	@echo ">>> OK."
	@echo ""

.PHONY: format
format: ## Format with Black
	@echo ">>> Formatting code..."
	black notebooks/ src/ tests/
	@echo ">>> OK."
	@echo ""

.PHONY: lint
lint: ## Lint with Flake8
	@echo ">>> Linting code..."
	flake8 --count --show-source --statistics src/ tests/
	@echo ">>> OK."
	@echo ""

.PHONY: bandit
bandit: ## Check security Bandit
	@echo ">>> Checking security ..."
	bandit --recursive src/ tests/
	@echo ">>> OK."
	@echo ""

.PHONY: mypy
mypy: ## Type check with Mypy
	@echo ">>> Checking types ..."
	mypy --install-types --non-interactive src/ tests/
	@echo ">>> OK."
	@echo ""

.PHONY: clean-mypy
clean-mypy: ## Remove Mypy cache files
	@echo ">>> Removing mypy artifacts..."
	rm -rf .mypy_cache
	@echo ">>> OK."
	@echo ""

.PHONY: test
test: ## Test and produce coverage report using pytest
	@echo ">>> Running tests and processing coverage..."
	pytest --cov-report=xml:coverage.xml --cov=src tests/
	@echo ">>> OK."
	@echo ""

.PHONY: clean-test
clean-test: ## Remove test cache and coverage files
	@echo ">>> Removing test and coverage artifacts..."
	rm -rf .coverage coverage.xml .pytest_cache
	@echo ">>> OK."
	@echo ""

.PHONY: qa
qa: isort format lint bandit mypy test ## Run the full QA process

.PHONY: clean-pycache
clean-pycache: ## Remove python cache files
	@echo ">>> Removing python artifacts..."
	rm -rf ./**/*.pyc ./**/__pycache__
	@echo ">>> OK."
	@echo ""

.PHONY: dataset
dataset: ## Download and extract dataset
	@echo ">>> Downloading and saving data files..."
	@if [ ! -f "data/raw/frames.json" ] ; \
	then \
		echo "Downloading data..." ; \
		wget -q -O data/raw/frames.zip https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+10%C2%A0-+D%C3%A9veloppez+un+chatbot+pour+r%C3%A9server+des+vacances/frames.zip ; \
		unzip -u -q data/raw/frames.zip -d data/raw/ ; \
		rm -f data/raw/frames.zip ; \
		echo "Done."; \
	else \
		echo "Data files already downloaded."; \
	fi
	@echo ">>> OK."
	@echo ""

.PHONY: clean-dataset
clean-dataset: ## Delete data files
	@echo ">>> Removing data files..."
	find ./data/ -type f -not -name ".gitignore" -delete
	@echo ">>> OK."
	@echo ""

.PHONY: clean-notebook
clean-notebook: ## Remove notebook cache and checkpoint files
	@echo ">>> Removing Notebook artifacts..."
	rm -rf .ipynb_checkpoints/ ./**/.ipynb_checkpoints/ ./**/*-checkpoint.ipynb
	@echo ">>> OK."
	@echo ""

.PHONY: clean-results
clean-reults: ## Delete result files
	@echo ">>> Removing result files..."
	find ./results/ -type f -not -name ".gitignore" -delete
	@echo ">>> OK."
	@echo ""

.PHONY: clean
clean: clean-venv clean-mypy clean-test clean-pycache clean-dataset clean-notebook clean-results ## Remove all file artifacts

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
