.PHONY: install install-dev test lint format audit smoke serve clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[notebooks]"

install-dev:
	$(PYTHON) -m pip install -e ".[notebooks,dev]"

test:
	$(PYTHON) -m pytest --cov=bioai --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

smoke:
	$(PYTHON) -m bioai about
	$(PYTHON) -m bioai features --sequence ATGATGATGATG --output /tmp/eeia-bioai-features.npy

audit:
	mkdir -p reports
	$(PYTHON) -m bioai audit \
		--train 2-data/processed/train.csv \
		--evaluation 2-data/processed/val.csv \
		--output reports/overlap-audit.json

serve:
	uvicorn bioai.api:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info src/*.egg-info
