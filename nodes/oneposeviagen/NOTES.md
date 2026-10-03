# oneposeviagen node notes

## Scope

Only the 3D-generation stage of OnePoseViaGen (legacy
`oneposeviagen_3dgen`). The scale-recovery (`fpose/recover_scale`) and
pose stages (`estimate_poses`, FoundationPose) are not part of this node:
use the `foundationpose` node with the generated mesh, scaled by an
external measurement. SpaTrackerV2 / SAM2 / gradio parts of the upstream
app are not used.

## How upstream is used

- Upstream GZWSAMA/OnePoseviaGen @ ecf41c5b. `entry.py` puts
  `oneposeviagen/trellis` (Hi3DGen variant of TRELLIS) and
  `oneposeviagen/Amodal3R` on `sys.path`, as app.py does.
- `patches/no_deepspeed.patch` (local) comments out
  `from deepspeed.ops.adam import FusedAdam` in
  `trellis/pipelines/trellis_image_to_3d.py` (training only; deepspeed is
  not installed).
- Weights: HF `ZhengGeng/OnePoseViaGen` @ bba5c451, only
  `Hi3DGen_Color/` (files referenced by its pipeline.json) and
  `Amodal3R/`. FoundationPose / SpatialTracker weights of that repo are
  not downloaded.
- DINOv2 (`dinov2_vitl14_reg`): both pipelines call
  `torch.hub.load(<hub_dir>/facebookresearch_dinov2_main, source='local',
  pretrained=True)`. entry.py builds a temporary hub dir with symlinks to
  a pinned dinov2 repo archive (commit 7764ea0f, hubconf.py sha256-pinned)
  and the official `dinov2_vitl14_reg4_pretrain.pth` (sha256-pinned), and
  calls `torch.hub.set_dir` on it, so nothing is downloaded at run time.
- `generate` (fully visible object, app.py `is_occluded=False`): Hi3DGen
  `TrellisImageTo3DPipeline` with the upstream app defaults (ss 12 steps /
  cfg 7.5, slat 25 steps / cfg 15, seed 42). The image + mask are turned
  into RGBA and passed through upstream `preprocess_image` (alpha branch:
  1.2x crop, pad square, 518 px, premultiplied) so the BiRefNet branch is
  never used. Texture: `postprocessing_utils.to_trimesh` (texture baked
  from the generated Gaussians, 100 views; xatlas UVs).
- `generate_amodal` (app.py `is_occluded=True`): Amodal3R
  `run_multi_image([image], [final_mask])`. The condition mask follows
  app.py: 255 background, 188 object, 0 occluder. Occluders come from an
  `occluder` mask, or from `depth` with a port of app.py
  `generate_final_mask` (pixels in the object box closer than the
  object's 90th depth percentile, eroded, components >= 100 px, gap to
  the object filled by dilation). Differences from app.py: the image is
  cropped to a square (1.2x the box of object + nearby occluders) before
  Amodal3R's 518x518 resize; app.py resizes the full frame, which
  distorts non-square images. Texture: amodal3r `to_glb`.

## Build / runtime pitfalls

- torch 2.8.0 (PyPI cu128 build, needed for RTX 5090 / sm_120) instead of
  upstream's torch with spconv-cu121 (cu121 cuBLAS bug on H20):
  spconv-cu126 2.3.8 (runs on sm_120 with `SPCONV_ALGO=native`),
  xformers 0.0.32.post2 (the torch 2.8.0 build).
  `ATTN_BACKEND=xformers` (as app.py), `SPCONV_ALGO=native`.
- nvdiffrast (v0.4.0 253ac4f) and diff-gaussian-rasterization
  (mip-splatting dda02ab, `submodules/diff-gaussian-rasterization`, used
  for texture baking from Gaussians) are uv git deps built against the
  runtime torch (`extra-build-dependencies`, match-runtime) with static
  metadata; `CC=gcc CXX=g++` (host `c++` is clang 11).
- `amodal3r.pipelines` imports `rembg` at module level (never called):
  rembg + onnxruntime are installed only for that import.
- The legacy node's `_create_occlusion_mask` was a stub that ignored
  occlusion; this node implements the upstream logic.

## Conventions

- Output glb: TRELLIS works Z-up in [-0.5,0.5]^3; `to_trimesh` /
  `to_glb` rotate to glTF Y-up. Upstream app.py additionally rotates the
  OBJ by +90 deg about X before FoundationPose; the node does not (pose
  estimators work in whatever mesh frame they are given). Not metric.

## the RTX 5090 host (RTX 5090 32 GB, sm_120): first real test of this node
- Built on the RTX 5090 host with nvcc 12.8 (`CUDA_HOME={cuda_home}`,
  `TORCH_CUDA_ARCH_LIST={cuda_arch}` = 12.0): nvdiffrast and
  diff-gaussian-rasterization (glm submodule via the local GitHub mirror).
- xformers 0.0.32 turns on its FlashAttention-3 kernels (Hopper sm_90a only)
  for every GPU with compute capability >= 9.0. On sm_120 the first
  attention call aborts the process ("CUDA error ...
  flash_fwd_launch_template.h:188: invalid argument", exit -6). entry.py
  (`_no_fa3_off_hopper`) calls `xformers.ops.fmha.dispatch._set_use_fa3(False)`
  on anything that is not sm_90, so xformers falls back to FA2 / cutlass.
- Peak VRAM over both tests: 14.9 GB (`min_vram_gb = 16`). ~45 s per case.
- Visual check: both glbs rendered from front / side / back next to the
  inputs. generate gives the white pickup with a matching texture; generate_amodal
  rebuilds the complete truck behind the grey occluder bar, with no bar in the
  mesh. Tests are field checks (textured, faces >= 5000, max_extent).
- Weights: dinov2 ckpt + dinov2 repo zip come from dl.fbaipublicfiles /
  github archives (not reachable from the RTX 5090 host); fetched on the ubuntu laptop
  and copied into the bucket (sha256 / files_sha256 verified by setup).
