# spatracker notes

## How upstream is used
- `[upstream]` henry123-boy/SpaTrackerV2 @ 7e12274c52077860cebfe007a6290777db43b63c
  (not a package); `entry.py` puts `ctx.repo` on sys.path and follows
  upstream `inference.py` (RGB / RGBD modes, offline tracker):
  `VGGT4Track.from_pretrained(<weights/front>)` -> depth, intrinsics,
  poses (c2w), confidence; then `Predictor.from_pretrained(<weights/offline>)
  .forward(video, depth, intrs, extrs, queries, query_no_BA=True, stage=1,
  support_frame=T-1, replace_ratio=0.2)` under bf16 autocast.
- Weights: HF `Yuxihenry/SpatialTrackerV2-Offline` and
  `Yuxihenry/SpatialTrackerV2_Front` at pinned revisions, loaded from local
  dirs (no runtime downloads). The configs reference
  `checkpoints/model.pt` / `checkpoints/scaled_offline.pth`; those are only
  loaded if they exist relative to cwd (run dir), which they don't, so the
  safetensors weights are used as intended.
- Preprocessing: upstream `preprocess_image` (width 518, height rounded to
  a multiple of 14, no crop for landscape). Portrait frames are rejected
  (VGGT4Track center-crops them internally, which would break the pixel
  mapping).
- Output conversion: 2D tracks and intrinsics are rescaled from the model
  resolution to the input size; depth = point_map z with conf < 0.5 set to
  0, nearest-resized to the input size; trajectory = c2w re-expressed
  relative to frame 0 and re-orthonormalised (bf16); xyz_world uses that
  trajectory. Upstream `vis_pred` is not a probability (range ~0.25..1.2);
  `visible = vis_prob > 0.5` (raw value kept as `vis_prob`).
- RGBD mode: `depth` (any resolution, resized nearest) + `camera` (K scaled
  to the model resolution) replace VGGT4Track depth / K; `trajectory`
  (T_world_cam, = upstream `extrs`, which are c2w) replaces VGGT poses,
  otherwise VGGT poses are the initialisation.
- Scale: with VGGT4Track depth the scale is the model's prediction
  (bedroom fixture: bed / wall at ~0.6-1.4 m, plausibly under-scaled);
  only RGBD mode with metric depth guarantees metres.

## Checks done
- Overlay of the rgb case: 2D grid tracks match the cotracker node (static
  points fixed, points on the moving girl follow her); depth map ordering
  correct (wall far, people/bed near); camera nearly static (the clip is
  shot from a static camera, |t| < 1 cm); reprojection of xyz_cam with K
  vs xy: median 2.4 px, 95th pct 7.4 px.
- rgbd_queries test feeds the rgb case outputs (depth downsampled to
  240x135, camera, trajectory) back in with 5 query points.
- No tracks/trajectory comparison metric exists in core: field checks only.

## Pitfalls met
- requirements.txt is a superset (gradio, ray, SAM, xformers==0.0.28 for
  torch 2.4...). Minimal set found by import probing: einops, kornia,
  timm, jaxtyping, huggingface-hub, safetensors, opencv, scipy, matplotlib,
  easydict, decord, pycolmap 3.11.1, pyceres 2.4, utils3d (pinned git),
  scikit-learn, flow-vis, moviepy==1.0.0 (`moviepy.editor` import in the
  visualizer, imported transitively), numpy<2. xformers is optional
  (falls back with "xFormers not available").
- torch 2.7.1 (cu126) instead of upstream-tested 2.4.1 (cu121 would SIGFPE
  on H20).

## Legacy
Legacy `spatracker` (OnePoseviaGen pipeline) chunked videos into 30-frame
windows, wrote depth as uint16 mm PNG + per-frame intrinsics json and
3D tracks only with a mask. Here: one pass over the whole sequence (the
Predictor itself windows at 200 frames with overlap), depth as depth_seq
(metres .npy), intrinsics as camera + per-frame K in tracks.npz, mask
restricts the grid. Rerun / mp4 visualisation dropped.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 (PyPI cu128) / torchvision 0.23.0, `sm_120` in `gpu.arch`,
  uv.lock re-generated (aliyun mirror); upstream + utils3d come from the
  local GitHub mirror. Test passes with the H20 thresholds unchanged and
  near-identical values (rgb: visible_ratio 0.9767 vs 0.9733 on H20, depth
  median 0.9091 vs 0.9095; rgbd_queries identical).
