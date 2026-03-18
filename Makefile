.PHONY: test test-unit test-integration test-fast test-cov test-cov-html test-verbose test-failfast clean test-install help

help:
	@echo "Test targets:"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-fast         - Run tests quickly (no coverage)"
	@echo "  make test-cov          - Run tests with coverage report to terminal"
	@echo "  make test-cov-html     - Run tests with HTML coverage report"
	@echo "  make test-verbose      - Run tests with verbose output"
	@echo "  make test-failfast     - Stop on first failure"
	@echo "  make test-install      - Install test dependencies"
	@echo "  make clean             - Remove test artifacts"

test-install:
	pip install -r requirements-test.txt

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-fast:
	pytest -p no:cov

test-cov:
	pytest --cov=triage --cov-report=term-missing

test-cov-html:
	pytest --cov=triage --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

test-verbose:
	pytest -v

test-failfast:
	pytest -x

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	rm -f test-results.xml
