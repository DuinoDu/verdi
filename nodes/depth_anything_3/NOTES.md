# depth_anything_3 node notes

- Upstream: ByteDance-Seed/Depth-Anything-3 @ 3d835ec1 (same DA3_COMMIT as
  SimFoundry install_da3.sh / install_simfoundry.sh), pip-installable, pinned
  as a uv git dependency (hatch-vcs version). torch 2.7.1 (cu126) +
  xformers 0.0.31.post1 (SimFoundry: torch 2.7.0 cu128 + xformers 0.0.30).
  opencv-python is overridden by opencv-python-headless.
- Weights: depth-anything/DA3NESTED-GIANT-LARGE-1.1 @ b2359bdf (the only
  checkpoint SimFoundry uses: da3_chunk_worker, depth_backends, PDA geometric
  backend) and depth-anything/DA3MONO-LARGE @ f465978e (relative task).
  Loaded with `DepthAnything3.from_pretrained(<local dir>)`, prefetched
  (6.3 GB safetensors on vepfs).
- SimFoundry usage mirrored:
  - `estimate_depth` = `DepthAnythingV3.infer_depth` (process_res 504,
    `upper_bound_resize`, cv2 bilinear resize back to the input size).
  - `reconstruct` = stage s2_da `DepthAnythingV3Backend`: process_res 448,
    single pass when N <= chunk_size (220), else chunks of 220 with 40-frame
    overlap, each later chunk Sim(3)-aligned (Umeyama on overlap camera
    centres) to the already merged world, its depth multiplied by the Sim(3)
    scale; overlap frames keep the earlier chunk's values. SimFoundry ran
    each chunk in a fresh subprocess to free VRAM on 24 GB cards; here chunks
    run in-process with `torch.cuda.empty_cache()` between them.
  - Heavy exports (glb, gs_ply, gs_video, colmap) are dropped (SimFoundry
    also disables them: not consumed downstream). The async npz-export race
    SimFoundry works around does not apply (export_dir=None).
- Conventions: DA3 extrinsics are world-to-camera OpenCV; we write
  T_world_cam = inv(extrinsic). Nested output is metric (`is_metric=1`);
  the node raises if not. Intrinsics are predicted at the network
  resolution and scaled per axis to the input resolution; depth is
  bilinearly resized (z values unchanged).
- Sky: NestedDepthAnything3Net overwrites sky pixels with the 99th-percentile
  non-sky depth (<= 200 m), which is not a measurement. The node wraps
  `depth_anything_3.model.da3.set_sky_regions_to_max_depth` to capture the
  sky mask and writes 0 there (param sky_invalid=false keeps the filler).
- DA3MONO predicts relative depth (not disparity, not metres); we return
  1/depth as `disparity` (unknown scale; 0 on sky/invalid).
- Licences: DA3 code Apache-2.0; NESTED/GIANT weights CC-BY-NC-4.0, MONO
  and METRIC weights Apache-2.0 (model cards).

## the RTX 5090 host (RTX 5090 32 GB, sm_120) — first tested here
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128) + xformers 0.0.32.post2
  (the build for torch 2.8). DA3 uses xformers only for the fused SwiGLU
  FFN (attention is torch SDPA), which runs fine on sm_120.
- `addict` is imported by `depth_anything_3.model.da3` but not declared in
  DA3's pyproject; on the H20 lock it came in via open3d 0.19, the the RTX 5090 host lock
  resolves open3d 0.20 (no addict) -> ModuleNotFoundError. Now pinned
  explicitly (`addict==2.4.0`).
- Verification (visual sheet of rgb / metric inverse depth / mono
  disparity / unitree recon depths, all plausible, sharp edges, correct
  ordering): bedroom0 metric median 2.62 m vs depth_pro 2.71 m (abs_rel
  0.041), fx 900 px (depth_pro 798); mono disparity vs 1/depth_pro
  Pearson 0.987. References in tests/expected were produced on the RTX 5090 host
  (depth float16, disparity float32 because core `disparity_rel` runs
  lstsq on the raw dtype and numpy linalg rejects float16; unitree
  trajectory json).
- reconstruct on the unitree fixture (15 frames, 640x360, nearly static
  camera, ~0.2 m path): single pass median depth 4.85 m. Forced chunking
  (chunk_size 10, overlap 5) gives the same trajectory shape (ATE 0.010 m
  after Sim(3)) but its median depth varied between runs (4.09 / 4.89 m;
  single pass 4.85 / 4.91 m): with only centimetres of camera
  motion in the overlap the Sim(3) scale is poorly constrained, so the
  chunk test only uses loose depth bounds. Real chunking (220/40 frames)
  has far more baseline.
