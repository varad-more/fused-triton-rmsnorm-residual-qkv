CONDA_ENV ?= triton-rmsnorm
RUN = conda run -n $(CONDA_ENV) --no-banner

.PHONY: install test benchmark clean

install:
	$(RUN) pip install -e ".[dev]"

test:
	$(RUN) python -m pytest tests/ -v

benchmark:
	$(RUN) python benchmarks/harness.py

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
