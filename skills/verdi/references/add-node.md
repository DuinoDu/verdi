# Adding or changing a node

A node is `nodes/<name>/` and is a standalone uv project.

```
nodes/<name>/
  manifest.toml     # contract: tasks, typed IO, params, system, weights, test
  pyproject.toml    # node deps; depends on verdi via path "../.."
  uv.lock           # committed; `verdi setup` runs `uv sync --frozen`
  entry.py          # verdi.sdk.main({"task": fn, ...})
  patches/*.patch   # optional, applied to [upstream] checkout
  tests/fixture/    # small inputs (< ~10 MB per node; committed, binaries via git LFS)
  tests/expected/   # optional reference outputs for metric checks
  NOTES.md          # known pitfalls, how the upstream is used
  STATUS.toml       # written by `verdi test`, committed
```

## Rules

1. **Upstream code**: if the upstream is pip-installable, depend on it
   in `pyproject.toml` with a git `rev` (locked in uv.lock). Otherwise
   declare `[upstream] repo + commit` (never a branch); setup clones it
   into `$VERDI_HOME/repos/<name>`, `ctx.repo` points there. Apply
   fixes as `patches/`, never edit the checkout by hand.
2. **Weights**: declare in `[weights.<key>]` (`hf` + `revision`,
   `url` + `sha256`, or `url` + `archive = "zip"|"tar"` + `files_sha256`);
   read them via `ctx.weight(key)`; pass `prefetch=True` for multi-GB
   files (vepfs mmap loading is very slow on a cold cache). `lazy = true` only when the upstream
   code insists on downloading itself (it then lands in HF_HOME).
3. **System deps** uv cannot provide go in `[system]` (gpu, cuda
   driver/toolkit, compilers, binaries, libs, disk, env, hf_token) so
   `verdi doctor` can check them. Build-time env (CUDA_HOME,
   TORCH_CUDA_ARCH_LIST=9.0) goes in `[build.env]`; extra build steps in
   `[setup].commands` (run after uv sync and weight download).
4. **Torch**: pin torch from the `pytorch-cu126` (or newer) index. Never
   use cu121 on the GPU host (cuBLAS 12.1 SIGFPE on m=1 GEMV).
5. **IO**: only unified types (`verdi types`). Convert upstream
   formats inside `entry.py`. Use `verdi.types.io` helpers. Use `file` /
   `dir` only when no type fits, and document the layout in the task
   description.
6. **No fabrication**: never return heuristic/fallback outputs. Raise
   `NodeError(message, hint=...)`.
7. **One node per model**: different usages are tasks of one node and
   share one venv.
8. **Test**: one `[[tests]]` case per task. Check kinds: summary
   `field` (see `verdi types` summaries), `path` into a json output
   (`objects.0.azimuth_deg`), or `metric` vs `tests/expected/`
   (`mask_iou`, `mask_class_iou`, `bbox_iou`, `depth_abs_rel` (file or
   dir), `disparity_rel`, `image_psnr` (file or dir), `normal_angle_deg`, `pose_translation_err`,
   `pose_rotation_err_deg`, `tracks_epe`, `trajectory_ate`). Each case needs checks on real
   outputs (prefer a `metric` check against `tests/expected/`, which
   you must inspect visually before committing). Run
   `verdi test <name>` on the GPU host and commit `STATUS.toml` + `uv.lock`.
   Negative cases are encouraged: `expect_error = "substring"` makes a case
   pass only if the run fails with that text in message/hint (e.g. a
   distorted camera or a wrong calibration must be refused). Inputs may
   point into the pinned upstream checkout with `{repo}/path` (files that
   cannot be committed for licence reasons).
   Prefer checks against INDEPENDENT ground truth (sensor depth, mocap
   poses, analytic synthetic scenes) over self-recorded reference outputs.

8b. **Scale contract**: a node whose depth / depth_seq / pointcloud output
   is not metric must call `io.write_scale(path, status, units, source)`
   (status relative | input_pose_scale | input_depth_scale; metric
   producers should declare `metric`). Geometry inputs require metric data
   by default; declare `scale = "any"` on ports that legitimately take
   other scales (e.g. the input of a scale alignment).

9. **Algorithm nodes** (no model, e.g. marker calibration, plane fitting)
   are allowed when deterministic and typed with the unified types; omit
   `[system].gpu` and `[weights]`, keep the same manifest/test rules.
10. **Test data in git**: everything under `tests/` is committed;
   binaries go through git LFS (`.gitattributes`), json/txt as plain git.

## Checklist

- [ ] `verdi doctor <name>` all OK on the GPU host
- [ ] `verdi setup <name>` from a clean `$VERDI_HOME/venvs/<name>`
- [ ] `verdi describe <name>` reads well without opening code
- [ ] `verdi test <name>` passes; STATUS.toml committed
- [ ] NOTES.md lists pitfalls met while integrating
