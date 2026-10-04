# depth_pro

- Upstream: the official transformers port of Apple Depth Pro
  (`DepthProForDepthEstimation` + `DepthProImageProcessorFast`,
  transformers 4.57.1), weights `apple/DepthPro-hf` pinned by revision.
  No git checkout is needed. torch 2.7.1 (cu126 PyPI wheel).
- The model outputs a canonical inverse depth and the horizontal FOV.
  `post_process_depth_estimation(target_sizes=[(H, W)])` converts it to
  metric depth with focal = 0.5 W / tan(0.5 fov) and resizes to the input
  size. Depth is proportional to the focal, so a known fx is applied as
  `depth * fx_known / f_est` (identical to upstream's f_px path apart from
  the 1e-4..1e4 clamp).
- Camera output: fx = fy = focal, cx, cy = image centre; Depth Pro does not
  estimate principal point or distortion. Given camera is echoed instead.
- `estimate_depth_seq` runs frames independently; the default
  `focal=median` shares the median focal so per-frame scale does not jump
  (per-frame focal estimates on the bedroom fixture vary 787..820 px, ~4%).
- fp16 (upstream default) vs the seq run on the same frame: abs_rel 0.007.
  `outputs.predicted_depth/field_of_view` are cast to fp32 before
  post-processing to avoid fp16 overflow in 1/depth.
- Legacy `ml_depth_pro` also supported Lance datasets and a json with
  depth statistics; dropped (verdi nodes take files; stats are in the
  output summary).
- Fixture expected depth is stored as float16 (1 MB) for the abs_rel check.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build, needed for sm_120);
  re-locked against aliyun PyPI. No code changes. abs_rel vs the H20
  reference 0.0003, so the H20 references were kept.
- hf-mirror downloads of Xet-backed files need HF_HUB_DISABLE_XET=1
  (core sets it now); otherwise huggingface_hub goes to the unreachable
  us.aws.cdn.hf.co.
