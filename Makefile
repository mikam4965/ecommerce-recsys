.PHONY: install preprocess train evaluate api ui test lint format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make preprocess  - Run data preprocessing"
	@echo "  make train       - Train recommendation model"
	@echo "  make evaluate    - Evaluate model performance"
	@echo "  make api         - Start FastAPI server"
	@echo "  make ui          - Start Streamlit UI"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean generated files"

# Install dependencies
install:
	pip install -e ".[dev]"

# Data preprocessing
preprocess:
	python scripts/preprocess.py

# Train model
train:
	python scripts/train.py

# Evaluate model
evaluate:
	python scripts/evaluate.py

# Start FastAPI server
api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit UI
ui:
	streamlit run ui/app.py --server.port 8501

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run linter
lint:
	ruff check src/ tests/
	mypy src/

# Format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist/ build/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
