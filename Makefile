# Main environment = verdi only. Each node has its own venv (verdi setup).
UV ?= $(shell command -v uv || echo $${VERDI_HOME:-$$HOME/.cache/verdi}/bin/uv)

env:
	$(UV) venv --allow-existing .venv
	$(UV) pip install --python .venv/bin/python -e ".[cli,dev]"

test:
	.venv/bin/python -m pytest -q tests

# Real model tests on this GPU machine (serial), updates nodes/*/STATUS.toml
test-nodes:
	.venv/bin/verdi test --all

.PHONY: env test test-nodes
