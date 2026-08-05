.PHONY: help setup data test lint pilot full reproduce clean check

PY      := .venv/bin/python
PHASE   ?= pilot
SEEDS   ?= 1 2 3
N       ?= 300
MAXCOST ?= 25

help:
	@echo "make setup      install pinned deps into .venv (Python 3.12)"
	@echo "make data       download + generate + checksum every benchmark"
	@echo "make test       run the full test suite (no network, no API key)"
	@echo "make lint       ruff check"
	@echo "make pilot      50-item pilot per condition   (needs ANTHROPIC_API_KEY)"
	@echo "make full       full run: N=$(N) SEEDS='$(SEEDS)' (needs ANTHROPIC_API_KEY)"
	@echo "make reproduce  regenerate every table from results/raw/ (NO network)"
	@echo ""
	@echo "Dry-run any phase without spending anything:"
	@echo "  $(PY) scripts/run.py --phase full --n $(N) --seeds $(SEEDS) --dry-run"

setup:
	uv venv --python 3.12
	uv sync --extra dev

data:
	$(PY) scripts/prepare_data.py

test:
	PYTHONPATH=src $(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests scripts

check: lint test

# --- experiments (cost money; each prints a projection and honours a ceiling)
pilot:
	$(PY) scripts/run.py --phase pilot --n 50 --seeds 1 \
		--datasets prontoqa proofwriter --max-cost $(MAXCOST)

full:
	$(PY) scripts/run.py --phase full --n $(N) --seeds $(SEEDS) \
		--datasets prontoqa proofwriter folio bbh --max-cost $(MAXCOST)

# --- analysis: offline, deterministic, regenerates every table and figure
reproduce:
	@test -d results/raw/pilot -o -d results/raw/full || \
		(echo "No raw records found. Run 'make pilot' or 'make full' first." && exit 1)
	@test -d results/raw/pilot && $(PY) scripts/analyze.py --phase pilot || true
	@test -d results/raw/full  && $(PY) scripts/analyze.py --phase full  || true
	@echo "Tables regenerated in results/tables/"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ results/tables/*.md results/tables/*.csv
