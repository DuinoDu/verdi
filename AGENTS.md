# Repository Guidelines

pdebug = verified, one-command model inference nodes for AI agents, run
through `otn-cli`. Read `skills/pdebug/SKILL.md` (usage) and
`skills/pdebug/references/add-node.md` (node rules) first.

## Where things run

Nodes run on a Linux host with NVIDIA GPUs. `PDEBUG_HOME` (default
`~/.cache/pdebug`) holds upstream checkouts, venvs, weights, caches and run
outputs. Per-host settings (PyPI/HF mirrors, a local git mirror for offline
hosts, CUDA toolkit path/arch, where weights live) go in
`$PDEBUG_HOME/config.toml`; see the docstring of `pdebug/core/config.py`.
Offline hosts: run `otn-cli offline-fetch` on a machine with internet and
copy its output over. Run GPU tests serially per GPU (`--device cuda:N`).

## Layout

- `pdebug/core/`: manifest, envelope, runner, doctor, install, testing,
  offline prefetch, host config
- `pdebug/types/`: unified data types (registry, comparators, io helpers)
- `pdebug/sdk/`: imported by node `entry.py` inside node venvs
- `pdebug/cli/`: `otn-cli`
- `nodes/<name>/`: one model (or deterministic algorithm) per node,
  standalone uv project; `tests/` fixtures/references via git LFS
- `pipelines/`: multi-node pipelines (SimFoundry = official repo in
  `third_party/SimFoundry`, used unmodified)
- `tests/`: unit/e2e tests for core (CPU only)

## Commands

- `make env`: main venv `.venv` with `otn-cli`
- `make test`: core tests (fast, CPU)
- `otn-cli test <node>` / `make test-nodes`: real model tests on the GPU host

## Rules

- Main package deps stay tiny; model deps live only in node venvs.
- Never add fallback/heuristic outputs to a node; raise `NodeError`.
- Conventions are fixed: metres, OpenCV camera frame, `T_cam_obj` /
  `T_world_cam`, pixel `xyxy`, `%06d` frames, RGB.
- Commit `uv.lock`, `STATUS.toml` and `tests/` with node changes.
- Python 3.10+, Black/isort (79 cols).
