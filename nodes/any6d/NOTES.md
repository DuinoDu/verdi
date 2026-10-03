# any6d node notes

## How upstream is used

- `[upstream]` taeyeopl/Any6D @ 80eb486 (the commit SimFoundry pins),
  cloned into `$PDEBUG_HOME/repos/any6d`; `foundationpose/` and the repo
  root go on `sys.path`; `estimater` is imported with cwd = `foundationpose/`
  (mycpp is resolved relative to it).
- `estimate` = the official `run_demo.py` path: `Any6D(...).register_any6d`
  (box alignment, per-axis scale search, FoundationPose refiner+scorer).
  The rescaled/re-centred mesh (`est.mesh`) is returned as `mesh`, the pose
  refers to it; `input_to_mesh` is a per-axis scale+offset fitted between
  input and output vertices (exact: residual ~1e-16).
- `register_candidates` = SimFoundry stage-8 usage: `register(...,
  return_all_poses=True)` on an already metric mesh; top_k poses sorted by
  scorer logit, plus the refiner's render of each candidate and the
  observed RGB crop (for VLM / visual selection).
- Patches: `simfoundry_Any6D.patch` copied as-is from SimFoundry
  (`patches/Any6D.patch`: PBR textures, `register(return_all_poses)`,
  mycuda dtype dispatch); `mycpp_no_boost.patch` drops the unused Boost
  includes/`find_package(Boost)` from mycpp (same as the foundationpose node).
- Weights: FoundationPose refiner `2023-10-28-18-33-37` + scorer
  `2024-01-11-20-02-45` from the HF mirror `gpue/foundationpose-weights`
  (byte-identical to the official Google Drive release, see the
  foundationpose node).
- Image-to-3D (InstantMesh/SAM in run_demo `--img_to_3d`) is NOT part of
  this node: give it a mesh from an image-to-3D node (e.g. hunyuan3d).

## Build (the RTX 5090 host, RTX 5090 sm_120)

- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128); nvdiffrast + pytorch3d
  (same git pins as foundationpose) built with `extra-build-dependencies`
  (`torch` match-runtime) for `{cuda_arch}` with `{cuda_home}` (12.8).
- mycpp: no system cmake / Eigen on the RTX 5090 host and no root -> PyPI `cmake` and
  `cmeel-eigen` (Eigen 3.4 + `Eigen3Config.cmake` in
  `site-packages/cmeel.prefix`, passed as `CMAKE_PREFIX_PATH`).
- hf-mirror answered 429 (rate limit) on the snapshot listing; the
  identical snapshot (sha256 checked) was copied from the foundationpose
  weights dir with the core `.pdebug_complete` marker.

## Fixture / tests

- `tests/fixture/mustard`: the upstream `demo_data` frame (RealSense
  640x480, YCB-Video-style labels): `color.png` -> jpg, `depth.png`
  (mm) -> float32 metres npy, `labels.npz` seg id 5 -> `mask.png`, depth
  intrinsics of `836212060125_640x480.yml` -> `camera.json` (as run_demo
  does), `mustard.obj` (upstream image-to-3D mesh, vertex colours) ->
  `mesh_generated.glb`. `mesh_metric.glb` = the node's own `estimate`
  output mesh (metric, centred), used by the `candidates` test.
- `tests/expected/ycb_gt_pose.json`: GT pose of the mustard bottle from
  `labels.npz` (`pose_y`, YCB model frame). Only the translation is
  comparable (the output mesh frame differs in rotation): the RTX 5090 host estimate is
  1.3 cm from it; estimated size 0.198 x 0.102 x 0.048 m (real bottle
  0.19 x 0.096 x 0.058 m). Overlay of the output mesh at the estimated pose
  checked visually (silhouette incl. cap matches); candidate renders match
  the observed crop; the 5 top candidates agree with `estimate` within
  0.3 mm / 0.4 deg.
- `estimate_poses.json` / `candidates_poses.json` are the the RTX 5090 host outputs.

## Pitfalls

- Upstream `register` silently returns a guessed pose when < 4 masked
  pixels have depth; the node raises instead.
- glb PBR textures: upstream reads `material.image`; texture-less
  materials are replaced by vertex colours.
