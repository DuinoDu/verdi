---
name: pdebug
description: Run verified model inference (segmentation, depth, detection, VLM, tracking, 3D reconstruction, mesh generation, 6D pose, ...) with one command via `otn-cli` on a GPU host. Use when a task needs the output of an existing vision/3D model instead of writing model code yourself.
---

# pdebug: model inference nodes for agents

pdebug packages existing models as **nodes**. Every node has its own uv
venv, a pinned upstream commit, pinned weights and a fixture test that
was run on a real GPU. You only call the CLI: no model background needed.

Run `otn-cli` on the GPU host where pdebug is installed (`make env` creates
`.venv/bin/otn-cli`). `PDEBUG_HOME` (default `~/.cache/pdebug`) holds venvs,
upstream checkouts, weights, caches and run outputs; host-specific mirrors
and paths live in `$PDEBUG_HOME/config.toml` (see `pdebug/core/config.py`).

## Workflow

1. `otn-cli list`: find a node. Prefer `test=pass` nodes; `never`/`fail`
   means unverified.
2. `otn-cli describe <node>`: tasks, typed inputs/outputs, params and an
   example command. Do not read node source unless describe is unclear.
3. If describe says setup is not ready: `otn-cli setup <node>` (runs
   `doctor` first; a failing check prints the exact fix command).
4. `otn-cli run <node> --task <t> -i name=path ... -p key=value`
   - stdout: result JSON; stderr: model log (`-q` to silence).
   - exit 0 = `status: ok`; outputs are at `outputs.<name>.path`, each
     with a `summary` (counts, sizes, ranges) you can sanity-check
     without opening files.
   - `--out DIR` chooses the output dir (default
     `$PDEBUG_HOME/runs/<node>/<run_id>/out`).
   - Video files are accepted wherever `image_seq` is expected (split
     into frames automatically).
5. Chain nodes by passing one node's output path as the next node's
   input. Types must match (`otn-cli types`).

## Result contract

```json
{"status": "ok" | "error",
 "outputs": {"mask": {"type": "mask", "path": "...", "summary": {...}}},
 "error": {"kind": "request|setup|timeout|crash|contract|<Exception>",
           "message": "...", "hint": "...", "log_tail": "..."},
 "provenance": {"node", "upstream_commit", "weights", "env_lock",
                "gpu", "duration_sec", "run_dir", ...}}
```

- Nodes never fabricate output: on failure you get `status=error`.
  Do not "fix" that by inventing data; read `hint` and `log_tail`.
- `kind=request`: your inputs/params are wrong; `describe` again.
- `kind=setup`: run the suggested `otn-cli setup`.
- `kind=contract`: node bug; report it, do not use the outputs.

## Unified data conventions (all nodes)

- Images RGB uint8 png/jpg; frame sequences are dirs of `%06d.png|jpg`
  (+ optional `meta.json {fps}`).
- Lengths in metres. Camera frame = OpenCV (x right, y down, z forward).
- Pose = 4x4 `T_cam_obj` (object to camera); trajectories are
  `T_world_cam`. Boxes = pixel `xyxy`.
- Masks = single-channel png, 0 background, 1..N instance ids.
- Depth = `.npy` float32 metres, 0 = invalid.
- Full list with formats: `otn-cli types`.

## Maintaining nodes

See `skills/pdebug/references/add-node.md` before adding or changing a
node. A node is only "available" after `otn-cli test <node>` passes on
the GPU host and its `STATUS.toml` is committed.
