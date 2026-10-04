# foundationpose

## How upstream is used

- Upstream `NVlabs/FoundationPose` pinned to `e3d597b8` (the same commit
  SimFoundry pins), cloned into `$VERDI_HOME/repos/foundationpose`, put on
  `sys.path`. `entry.py` imports `estimater` and calls
  `FoundationPose.register` (estimate) and `register` + `track_one` (track)
  just like `run_demo.py`.
- Patches:
  - `simfoundry_FoundationPose.patch`: copied as-is from SimFoundry
    (`third_party/SimFoundry/patches/FoundationPose.patch`): mycuda
    `scalar_type()` dispatch, c++17, unpinned requirements. mycuda/kaolin
    (BundleSDF, model-free path) are NOT built: the model-based path does
    not need them.
  - `mycpp_no_boost.patch`: mycpp includes three Boost headers it never
    uses and `find_package(Boost COMPONENTS system program_options)`.
    Boost is not installed on the H20 and apt is broken there (half-installed
    `fsx` package), so the dependency was dropped instead.
- `[setup].commands` builds `mycpp` (cmake + pybind11 from the venv +
  system Eigen3) and symlinks `<repo>/weights -> $VERDI_HOME/weights/foundationpose/fp`
  (upstream hardcodes `<repo>/weights/<run_name>`).
- Weights: official refiner `2023-10-28-18-33-37` and scorer
  `2024-01-11-20-02-45`. The official release is on Google Drive (not
  reachable from the H20); the HF mirror `gpue/foundationpose-weights` has
  byte-identical files (sha256 checked against an official copy).
- nvdiffrast v0.4.0 and pytorch3d v0.7.9 are uv git deps built with
  `extra-build-dependencies` (`torch` with `match-runtime`), which needs
  `[[tool.uv.dependency-metadata]]` entries so uv can lock without building.

## Conventions

- Inputs: depth `.npy` metres (values < 1 mm treated invalid), mask any
  non-zero (or `mask_id`), mesh in metres. glb PBR textures are converted to
  trimesh `SimpleMaterial` because upstream reads `material.image`.
- Output pose = upstream `ob_in_cam` = T_cam_obj of the ORIGINAL mesh frame
  (upstream internally recenters the mesh and undoes it before returning).
- `score` in the pose_set is the scorer logit of the best hypothesis
  (relative, not a probability).

## Pitfalls

- `c++` on the H20 PATH is clang: nvdiffrast fails with
  `-Wc++11-narrowing`. `[build.env] CC=gcc CXX=g++` fixes it.
- `warp-lang` is an *optional* import in `Utils.py`, but `register` needs
  `erode_depth`/`bilateral_filter_depth` which only exist when warp
  imports (otherwise `NameError: erode_depth`). Pinned to 1.0.2 like
  upstream (its kernel cache goes to `~/.cache/warp`).
- Upstream `register` silently returns a guessed translation with identity
  rotation when < 4 masked pixels have depth; the node raises instead.
- Eigen3 headers come from the system (`libeigen3-dev`); doctor has no
  header check, so this is not verified by `verdi doctor`.
- `pytorch3d` is only needed because `Utils.py` imports it at module
  level (rendering uses nvdiffrast).

## Fixture

`tests/fixture/mustard`: 3 frames (0, 5, 10) of the official `mustard0`
demo (HF dataset `mp1704/foundationpose-demo-data@2d4c1046`), rgb as jpg,
depth converted mm png -> float32 metres npy, first-frame mask, `cam_K.txt`
-> camera.json, `textured_simple.obj` -> `mesh.glb` with the texture
downscaled to 1024x1024. Total ~5 MB (depth .npy dominates).
The expected estimate pose was checked visually (mesh silhouette overlay).

## the RTX 5090 host (RTX 5090, sm_120) migration

- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128); nvdiffrast and pytorch3d
  (same git pins) rebuilt with CUDA 12.8 for sm_120 via `{cuda_home}` /
  `{cuda_arch}` in `[build.env]`.
- the RTX 5090 host has neither system cmake nor Eigen3 headers (and no apt): mycpp is
  now built with PyPI `cmake` (venv) and PyPI `cmeel-eigen` (Eigen 3.4
  headers + `Eigen3Config.cmake` under `site-packages/cmeel.prefix`, passed
  as `CMAKE_PREFIX_PATH`). `cmake` was dropped from `[system].binaries`
  (doctor runs before the venv exists).
- hf-mirror sometimes answers 429 (rate limit) on the weight snapshot;
  re-running setup is enough.
