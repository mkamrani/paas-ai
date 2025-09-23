.PHONY: test test-unit test-integration test-rag test-coverage test-fast install dev-install clean

# Default target
help:
	@echo "Available commands:"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-rag      - Run RAG-related tests only"
	@echo "  test-fast     - Run tests without coverage (faster)"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  install       - Install package dependencies"
	@echo "  install-all   - Install package with all dependencies"
	@echo "  dev-install-all - Install package with all dependencies and dev dependencies"
	@echo "  dev-install   - Install package with dev dependencies"
	@echo "  clean         - Clean cache and build artifacts"

# Testing commands
test:
	poetry run pytest

test-unit:
	poetry run pytest tests/unit -v

test-integration:
	poetry run pytest tests/integration -v

test-rag:
	poetry run pytest tests/unit/test_cli/test_commands/test_rag tests/unit/test_core/test_rag -v --no-cov

test-fast:
	poetry run pytest --no-cov -x

test-coverage:
	poetry run coverage run --source=src -m pytest
	poetry run coverage report --show-missing
	poetry run coverage html

# Installation commands
install:
	poetry install

install-all:
	poetry install --with agent,mcp,api,database,integrations

dev-install-all:
	poetry install --with agent,mcp,api,database,integrations,dev

dev-install:
	poetry install --extras dev

# Cleanup
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 