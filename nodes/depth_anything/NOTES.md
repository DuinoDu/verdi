# depth_anything

## How upstream is used
- Images: Depth Anything V2 through transformers 4.57.1
  (`AutoModelForDepthEstimation` + `post_process_depth_estimation`, which
  resizes to the input size). Checkpoints pinned by HF revision:
  - `estimate_depth`: Depth-Anything-V2-Metric-{Indoor,Outdoor}-Large-hf
    (`depth_estimation_type = metric`, output already in metres; indoor
    max_depth 20 m, outdoor 80 m).
  - `estimate_relative_depth`: Depth-Anything-V2-Large-hf (relative).
- Sequences: Video-Depth-Anything (ByteDance) from the pinned `[upstream]`
  checkout (not pip-installable; its requirements pin torch 2.1 /
  xformers 0.0.23). `VideoDepthAnything.infer_video_depth` is called with
  the frames as one array (vitl, input_size 518, fp16 autocast like
  run.py). Checkpoints: Metric-Video-Depth-Anything-Large (metres) and
  Video-Depth-Anything-Large (relative).

## Metric vs relative (important)
- Relative models output affine-invariant DISPARITY (inverse depth up to
  unknown scale and shift, larger = closer). They are exposed as
  `file` / `dir` of .npy and never as `depth`, because `depth` means
  metres. The video relative model shares one scale/shift over the clip
  (upstream aligns overlapping 32-frame windows with least squares).
- Metric VDA: upstream skips the scale/shift alignment (`metric=True`),
  so windows are stitched only by interpolation.
- Legacy `depth-anything-video` turned relative output into fake
  "depth_mm" pngs (1 / disparity scaled so the median = 1 m); this is
  dropped, since it invents a scale.

## Pitfalls
- xformers is optional for VDA (falls back to torch attention); it is not
  installed to avoid a torch-version pin.
- VDA needs no minimum length: short clips are padded with the last frame
  up to the 32-frame window; outputs are truncated to the input length.
- Frames larger than max_res (1280) are downscaled for inference (as in
  upstream `read_video_frames`) and the depth is resized back.
- Legacy `depth_anything` used a HuggingFace gradio Space API for the
  visualisation path and `Depth-Anything-V2-Small-hf` (relative) for raw
  output; both replaced by local inference with pinned weights.
- hf-mirror.com is slow on cold files (several minutes per 1.3 GB
  checkpoint, sometimes stalling; re-running setup resumes).
- Large relative checkpoints are CC-BY-NC-4.0 (see known_issues).

## Verification (bedroom fixture frame 0, vs depth_pro metric depth)
- estimate_depth (indoor): median 2.55 m vs depth_pro 2.71 m, abs_rel 0.076.
- estimate_depth_seq (Metric-VDA): median 3.34 m, abs_rel 0.27 vs
  depth_pro: the video metric model reads ~25% farther on this indoor
  clip (trained on VKITTI + IRS); prefer depth_pro / indoor V2 when
  absolute scale matters, VDA when temporal consistency matters.
- relative disparity (image / video): Pearson correlation with
  1/depth_pro 0.990 / 0.977.
- Expected references are stored as float16 (1 MB each); the relative
  disparity is compared with depth_abs_rel too (it just loads both .npy).
- First run after setup: ~7 min for the image tasks (cold vepfs cache of
  the 1.3 GB checkpoints); warm runs ~20 s.

## the RTX 5090 host (RTX 5090 32 GB, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128); transformers unchanged.
- Video-Depth-Anything OOMed on 32 GB: the vendored DINOv2 `Attention`
  (used when xformers is absent) materialises a B x heads x N x N matrix
  for the 32-frame window (11.4 GiB allocation). Adding xformers 0.0.32
  did not help: it dispatches to its FA3 (hopper) kernel, which crashes on
  sm_120 (`flash_fwd_launch_template.h:188: invalid argument`).
  `patches/0001-vda-sdpa-attention.patch` (ours) replaces the explicit
  attention by `torch.nn.functional.scaled_dot_product_attention` (same
  math and scale; fused memory-efficient kernel). xformers stays out.
- Results unchanged vs the H20 references: image abs_rel 0.0002 / 0.0003,
  seq_metric median 3.344 m (H20 3.34).
