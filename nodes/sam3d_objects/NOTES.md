# sam3d_objects node notes

Meta SAM 3D Objects (arXiv:2511.16624). NOT the `sam3` segmentation node.

## How upstream is used

- Upstream facebookresearch/sam-3d-objects @ f91db411 (`[upstream]`, put on
  `sys.path`; not pip-installed: its hatch requirements pull ~90 training /
  tooling packages). entry.py re-implements the thin `notebook/inference.py`
  `Inference` wrapper: `OmegaConf.load(pipeline.yaml)`,
  `rendering_engine = "pytorch3d"` (as the notebook), `compile_model = False`,
  `hydra.utils.instantiate` -> `InferencePipelinePointMap`, then per object
  `pipeline.run(rgba, None, seed, ...)` (RGBA = image + object mask in alpha,
  as `Inference.merge_mask_to_rgba`). The demo notebooks
  (`demo_single_object` / `demo_multi_object`) run exactly this once per mask
  with seed 42.
- Differences from the notebook wrapper: `with_mesh_postprocess` and
  `with_texture_baking` default to true (the defaults of
  `InferencePipelinePointMap.run`; the notebook turns both off and exports
  vertex colours / Gaussians only). Params `mesh_postprocess=false,
  texture_baking=false` reproduce the notebook. `layout_refine` maps to
  `with_layout_postprocess` (off in the notebook too).
- The point map is computed ONCE for the full image (all objects share it,
  so they share one scale) and passed as `pointmap=`; this is the same
  MoGe call upstream makes inside `run()`.
- `LIDRA_SKIP_INIT=true` (as the notebook): `sam3d_objects/__init__.py`
  otherwise imports a non-released `sam3d_objects.init`.
- `patches/0001-no-kaolin.patch`: kaolin is only imported for
  `check_tensor` (shape asserts in FlexiCubes); kaolin 0.17 has no wheel
  for torch 2.8 and its S3 wheel index is unreachable. The patch keeps the
  import if kaolin exists, otherwise uses a no-op.
- DINOv2 (ViT-L/14 + registers, image + mask conditioning) is loaded by
  upstream with `torch.hub.load("facebookresearch/dinov2", source="github")`;
  entry.py rewrites the configs to `source="local"` on the pinned dinov2
  repo archive (7764ea0f) and points `torch.hub` at the official
  `dinov2_vitl14_reg4_pretrain.pth` (same files as nodes/oneposeviagen).
- Depth model: pipeline.yaml uses MoGe **v1** (`Ruicheng/moge-vitl`) via
  `moge.model.v1.MoGeModel.from_pretrained`; entry.py replaces the HF repo id
  by the pinned local `model.pt`. MoGe pinned to a8c37341 and utils3d to
  3913c65 (upstream requirements.inference.txt / MoGe's own pin).

## Weights (ModelScope mirror, pinned by the official sha256)

- `facebook/sam-3d-objects` is gated on HuggingFace (manual approval);
  hf-mirror returns 403. With the user's approval the files come from the
  ModelScope mirror `modelscope.cn/models/facebook/sam-3d-objects`.
- Verification against the official repo (HF revision
  2e73555018d2741ccd486e56c24fac41155a1dc6): the public HF tree API hides
  the LFS sha256 of gated repos but exposes the git blob oid of every file.
  For every LFS file, sha1("blob <n>\0" + LFS pointer built from the
  ModelScope sha256 + size) equals the HF oid, i.e. the ModelScope sha256
  values ARE the official LFS sha256 (all 13 LFS files matched). For the
  small yaml files the git hash-object of the ModelScope file equals the HF
  blob oid. Every file is then pinned by sha256 in manifest.toml.
- Only the files referenced by pipeline.yaml are fetched (no ss/slat
  encoders): ss_generator (6.7 GB), slat_generator (4.9 GB), ss_decoder,
  slat_decoder_gs, slat_decoder_gs_4, slat_decoder_mesh + their yaml, LICENSE.
- **Licence: SAM License (Meta, custom; covers code AND weights)**, gated
  distribution, Trade-Controls clause; redistribution must include the
  licence text (stored as weight `license`). DINOv2 Apache-2.0, MoGe MIT.
  pytorch3d BSD; spconv Apache-2.0.

## Conventions (outputs)

- Upstream layout: `rotation` (quaternion wxyz), `translation`, `scale`
  (3 equal values) map the object's local frame (TRELLIS z-up voxel frame,
  about [-0.5, 0.5]^3) into the PyTorch3D camera frame (x left, y up,
  z forward) with pytorch3d `Transform3d().scale(s).rotate(R).translate(t)`
  (row vectors), see `SceneVisualizer.object_pointcloud` / `make_scene`.
- Upstream `to_glb` rotates the mesh z-up -> y-up (`v_glb = v_local @ A`).
  The node keeps that glb frame for `meshes/obj_<id>.glb` (glTF +Y up,
  normalised, NOT metric) and folds A and the PyTorch3D->OpenCV flip
  (diag(-1,-1,1)) into the pose: `p_cv = R @ (scale * v_glb) + t` with
  `T_cam_obj = [R | t]` rigid and `scale` a scalar. `size` =
  scale * extent of the glb along the object axes.
- Units of `t`, `scale`, `size`: those of the conditioning point map.
  The pose decoder (`ScaleShiftInvariant`) normalises the point map
  (object-centric scale/shift) and maps the predicted layout back with the
  same scale/shift, so the layout lives in point-map units:
  - default (no `depth`): MoGe v1 point map = affine-invariant up to an
    unknown global scale -> `units: "relative"` (NOT metric; MoGe v1 has no
    metric head);
  - `depth` (metres) + `camera`: back-projected point map -> `units:
    "metres"`, metric as far as the depth is. Use the `moge` node (MoGe-2,
    metric) outputs `depth` + `camera` for a phone image without a depth
    sensor, or sensor depth.
- `scene.glb`: every object mesh transformed by `[R*scale | t]` into the
  OpenCV camera frame (x right, y down, z forward); a glTF viewer shows it
  upside down (rotate 180 deg about x).
- `render_mask` / `overlay`: the posed meshes projected with `camera`;
  `metadata.projection_iou` per object = layout sanity check.

## Depth input: flying pixels

- With sensor depth, mask-border pixels mix object and background depth
  (fixture: 40 of 9489 mask pixels, up to 8.7 m deep). They bias upstream's
  object-centric point-map normalisation and shift the layout (projection
  IoU 0.45 -> 0.74 after the fix). `depth_edge_rtol` (default 0.04, same
  rule as MoGe's `depth_map_edge`: 3x3 depth range > rtol * depth) sets those
  pixels to NaN (= invalid for upstream) before building the point map.
  MoGe point maps are smooth and do not need it.

## layout_refine (upstream layout post-optimisation)

- Upstream wraps it in `try/except` and silently returns the unrefined
  layout on any error; the node raises instead. On the RTX 5090 host it runs (gsplat
  built) but reports IoU 0 before and after optimisation (its own render vs
  mask never overlaps; cause not investigated), so the node raises when the
  initial IoU is 0. Param documented as EXPERIMENTAL, default off (as in the
  upstream notebook).

## Fixture / tests

- `tests/fixture/mustard`: frame 0 of the FoundationPose demo `mustard0`
  (YCB mustard bottle, RGB-D + K + mask; same source as nodes/foundationpose).
- Visual check (the RTX 5090 host): textured mesh rendered from front / side / top next
  to the input; the French's mustard bottle with readable label; posed mesh
  projected onto the image over the input mask contour. Metric case size
  0.100 x 0.192 x 0.057 m vs the YCB model 0.097 x 0.191 x 0.067 m
  (sorted extents), depth 0.745 m. MoGe v1 case gives nearly the same numbers
  on this image, but that is not guaranteed (scale-invariant).
- References `tests/expected/*_poses.json` / `*_render_mask.png` are from
  the visually checked the RTX 5090 host runs (seed 42). Peak VRAM 17.5 GB, ~60 s per
  object on the RTX 5090.

## Build / runtime pitfalls (the RTX 5090 host, RTX 5090 sm_120)

- torch 2.8.0 (PyPI cu128) instead of upstream torch 2.5.1+cu121;
  spconv-cu126 2.3.8 (needs `SPCONV_ALGO=native` on sm_120),
  xformers 0.0.32.post2 (`ATTN_BACKEND=xformers`; flash-attn 2.8.3 has no
  torch 2.8 wheel). xformers' FA3 kernels abort on sm_120 -> entry.py turns
  FA3 off on non-Hopper GPUs (same fix as trellis2 / oneposeviagen).
- pytorch3d @ 75ebeea (upstream requirements.p3d.txt) and
  diff-gaussian-rasterization (mip-splatting dda02ab, TRELLIS v1 renderer)
  are built from git against the runtime torch with nvcc 12.8 for sm_120.
  gsplat @ 2323de59 (upstream pin) is built from git at install time
  (needs `github.com/g-truc/glm` in the git mirror for its submodule): the
  layout post-optimisation renders the Gaussians with gsplat, and the PyPI
  wheel would JIT-compile at run time (failed: no ninja on PATH).
- nvdiffrast @ 253ac4f (built for sm_120): the mesh post-process hole
  filling (`_fill_holes`, default `with_mesh_postprocess=True`) rasterises
  with `utils3d.torch.RastContext(backend="cuda")` = nvdiffrast, even though
  texture baking uses pytorch3d (`rendering_engine="pytorch3d"`).
- Missing upstream runtime deps found by import: `astor`
  (sam3d_objects.data.utils), `pkg_resources` (-> `setuptools<80`).
- hydra patch (`patching/hydra`, upstream PR 2863) is not needed for
  inference.
- hf-mirror API calls are rate-limited (HTTP 429) from the RTX 5090 host; `resolve/`
  downloads still work. Weights were placed manually where setup expects
  them (sha256 / `.pdebug_complete` marker identical to what setup writes).
