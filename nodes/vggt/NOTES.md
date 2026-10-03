# vggt node notes

- Upstream: facebookresearch/vggt @ a288dd0f (pip-installable, pinned as a
  uv git dependency). Weights: facebook/VGGT-1B @ 860abec7 (config.json +
  model.safetensors, loaded with `VGGT.from_pretrained(<local dir>)`).
- Inference follows upstream demo: whole `model(images)` under bf16
  autocast, `pose_encoding_to_extri_intri`, depth head output; the point
  cloud is built by unprojecting depth with the predicted cameras (the
  upstream recommendation, more accurate than the point head).
- Preprocessing: upstream "crop" mode resizes width to 518 and
  centre-crops the height for portrait frames. We keep the full frame:
  long side 518, other side rounded to a multiple of 14, no crop. Then
  intrinsics are rescaled per axis back to the input resolution and depth
  is bilinearly resized (z-depth values unchanged).
- Conventions: VGGT extrinsics are world-to-camera OpenCV with frame 0 as
  identity; we write T_world_cam = inv(extrinsic). Scale is arbitrary
  (scene normalised by the model), stated in the outputs and in the
  manifest. Not metric.
- The legacy BA / COLMAP export (pycolmap, LightGlue tracker) and the
  viser viewer are dropped (BA is an optional post-process; viser is an
  interactive UI). `camera` is the mean of the per-frame intrinsics;
  `intrinsics` (json) has the per-frame K.
- Frames must share one resolution; more than max_frames (64) raises with
  a hint to subsample rather than silently subsampling.
- Test fixture is sam2's bedroom (6 frames, nearly static camera, two
  jumping children): the moving children get low confidence and drop out
  of the point cloud; depth maps are sharp. No depth_seq or trajectory
  comparison metric exists in core, so the test uses tight field ranges
  (fx, depth median, point count) taken from the visually checked run.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 (PyPI cu128) / torchvision 0.23.0, `sm_120` in `gpu.arch`,
  uv.lock re-generated (aliyun mirror). Test passes with the H20 ranges
  unchanged: fx 945.0 (H20 939.9), depth median 0.9165 (H20 0.9148), same
  point count. HF download of VGGT-1B needs `HF_HUB_DISABLE_XET=1`
  (set by core when an HF mirror is configured).
