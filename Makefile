# Main environment = otn-cli only. Each node has its own venv (otn-cli setup).
UV ?= $(shell command -v uv || echo $${PDEBUG_HOME:-$$HOME/.cache/pdebug}/bin/uv)

env:
	$(UV) venv --allow-existing .venv
	$(UV) pip install --python .venv/bin/python -e ".[cli,dev]"

test:
	.venv/bin/python -m pytest -q tests

# Real model tests on this GPU machine (serial), updates nodes/*/STATUS.toml
test-nodes:
	.venv/bin/otn-cli test --all

.PHONY: env test test-nodes
