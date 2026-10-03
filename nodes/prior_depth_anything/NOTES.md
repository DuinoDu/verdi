# prior_depth_anything node notes

- Upstream: SpatialVision/Prior-Depth-Anything @ 8c029cbc (same commit as
  SimFoundry install_simfoundry.sh step 2.10) with SimFoundry's
  `patches/PriorDepthAnything.patch` (copied verbatim from
  third_party/SimFoundry/patches/; origin: SimFoundry team, see its
  PATCH_PROVENANCE.md). The patch adds the `args=` constructor argument,
  replaces the fixed 518 px network size by `H // 14 * 14` (networks run
  near the input resolution), uses `>=` for confidence thresholds in the
  plugin, and unpins the torch 2.2 requirements. Imported from the patched
  checkout via sys.path (not pip-installed).
- torch_cluster (KNN) is a CUDA extension: installed from the PyG wheel
  index (data.pyg.org, `torch_cluster-1.6.3+pt27cu126`), pinned in uv.lock.
- Weights: Rain729/Prior-Depth-Anything @ 25b3bfcf: depth_anything_v2_vitl.pth
  (frozen coarse MDE, SimFoundry frozen_model_size=vitl),
  prior_depth_anything_vitb.pth (v1.0, SimFoundry) and
  prior_depth_anything_vitb_1_1.pth (v1.1). Passed as mde_dir/ckpt_dir so
  upstream never calls hf_hub_download.
- SimFoundry usage (5_decompose_scene.py): version 1.0, vitl frozen + vitb
  conditioned, K=5, prior = current metric depth with the removed object
  region (dilated) zeroed, geometric = DA3 nested depth at the same size,
  then `np.where(prior == 0, output, prior)` -> param keep_prior (default
  true). SimFoundry's fp32 `calc_scale_shift` override is only needed under
  autocast, which the node does not use.
- Units: prior/geometric/output in metres (upstream is unit-agnostic; the
  fixture pngs are millimetres and were converted to metres).
- Fixtures: upstream assets sample-3 (405 sparse prior points, GT depth) and
  sample-1 (GT depth with a 180x220 box hole as prior + upstream geo_depth,
  i.e. the SimFoundry inpainting setting). Stored as float16 .npy to keep the
  node small; the node casts to float32. Tests compare against upstream GT
  (hole-only GT for the inpainting case).

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128). The PyG prebuilt
  `torch_cluster+pt27cu126` wheel no longer matches and data.pyg.org is not
  reachable offline, so torch-cluster 1.6.3 is now built from the PyPI
  sdist against the runtime torch (`[tool.uv.extra-build-dependencies]`
  with `match-runtime = true`, plus `[[tool.uv.dependency-metadata]]`
  because uv refuses runtime matching for packages without static
  metadata). `[build.env]` gives CUDA_HOME / TORCH_CUDA_ARCH_LIST (host
  config: cuda-12.8 / 12.0), FORCE_CUDA=1. Build takes ~15 min.
- Test metrics on the RTX 5090 host vs upstream GT: abs_rel 0.0193 / 0.0107 (H20:
  0.0193 / 0.0108); thresholds unchanged.
