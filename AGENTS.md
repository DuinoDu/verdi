# Repository Guidelines

verdi = verified, one-command model inference nodes for AI agents, run
through `verdi`. Read `skills/verdi/SKILL.md` (usage) and
`skills/verdi/references/add-node.md` (node rules) first.

## Where things run

Nodes run on a Linux host with NVIDIA GPUs. `VERDI_HOME` (default
`~/.cache/verdi`) holds upstream checkouts, venvs, weights, caches and run
outputs. Per-host settings (PyPI/HF mirrors, a local git mirror for offline
hosts, CUDA toolkit path/arch, where weights live) go in
`$VERDI_HOME/config.toml`; see the docstring of `verdi/core/config.py`.
Offline hosts: run `verdi offline-fetch` on a machine with internet and
copy its output over. Run GPU tests serially per GPU (`--device cuda:N`).

## Layout

- `verdi/core/`: manifest, envelope, runner, doctor, install, testing,
  offline prefetch, host config
- `verdi/types/`: unified data types (registry, comparators, io helpers)
- `verdi/sdk/`: imported by node `entry.py` inside node venvs
- `verdi/cli/`: `verdi`
- `nodes/<name>/`: one model (or deterministic algorithm) per node,
  standalone uv project; `tests/` fixtures/references via git LFS
- `pipelines/`: multi-node pipelines (SimFoundry = official repo in
  `third_party/SimFoundry`, used unmodified)
- `tests/`: unit/e2e tests for core (CPU only)

## Commands

- `make env`: main venv `.venv` with `verdi`
- `make test`: core tests (fast, CPU)
- `verdi test <node>` / `make test-nodes`: real model tests on the GPU host

## Rules

- Main package deps stay tiny; model deps live only in node venvs.
- Never add fallback/heuristic outputs to a node; raise `NodeError`.
- Conventions are fixed: metres, OpenCV camera frame, `T_cam_obj` /
  `T_world_cam`, pixel `xyxy`, `%06d` frames, RGB.
- Commit `uv.lock`, `STATUS.toml` and `tests/` with node changes.
- Python 3.10+, Black/isort (79 cols).
