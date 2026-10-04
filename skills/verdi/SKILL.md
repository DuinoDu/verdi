---
name: verdi
description: Run verified model inference (segmentation, depth, detection, VLM, tracking, 3D reconstruction, mesh generation, 6D pose, ...) with one command via `verdi` on a GPU host. Use when a task needs the output of an existing vision/3D model instead of writing model code yourself.
---

# verdi: model inference nodes for agents

verdi packages existing models as **nodes**. Every node has its own uv
venv, a pinned upstream commit, pinned weights and a fixture test that
was run on a real GPU. You only call the CLI: no model background needed.

Run `verdi` on the GPU host where verdi is installed (`make env` creates
`.venv/bin/verdi`). `VERDI_HOME` (default `~/.cache/verdi`) holds venvs,
upstream checkouts, weights, caches and run outputs; host-specific mirrors
and paths live in `$VERDI_HOME/config.toml` (see `verdi/core/config.py`).

## Workflow

1. `verdi list`: find a node. Prefer `test=pass` nodes; `never`/`fail`
   means unverified.
2. `verdi describe <node>`: tasks, typed inputs/outputs, params and an
   example command. Do not read node source unless describe is unclear.
3. If describe says setup is not ready: `verdi setup <node>` (runs
   `doctor` first; a failing check prints the exact fix command).
4. `verdi run <node> --task <t> -i name=path ... -p key=value`
   - stdout: result JSON; stderr: model log (`-q` to silence).
   - exit 0 = `status: ok`; outputs are at `outputs.<name>.path`, each
     with a `summary` (counts, sizes, ranges) you can sanity-check
     without opening files.
   - `--out DIR` chooses the output dir (default
     `$VERDI_HOME/runs/<node>/<run_id>/out`).
   - Video files are accepted wherever `image_seq` is expected (split
     into frames automatically).
5. Chain nodes by passing one node's output path as the next node's
   input. Types must match (`verdi types`).

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
- `kind=setup`: run the suggested `verdi setup`.
- `kind=contract`: node bug; report it, do not use the outputs.

## Unified data conventions (all nodes)

- Images RGB uint8 png/jpg; frame sequences are dirs of `%06d.png|jpg`
  (+ optional `meta.json {fps}`).
- Lengths in metres. Camera frame = OpenCV (x right, y down, z forward).
- Pose = 4x4 `T_cam_obj` (object to camera); trajectories are
  `T_world_cam`. Boxes = pixel `xyxy`.
- Masks = single-channel png, 0 background, 1..N instance ids.
- Depth = `.npy` float32 metres, 0 = invalid.
- Full list with formats: `verdi types`.

## Maintaining nodes

See `skills/verdi/references/add-node.md` before adding or changing a
node. A node is only "available" after `verdi test <node>` passes on
the GPU host and its `STATUS.toml` is committed.
