# foundation_stereo node notes

- Upstream: NVlabs/FoundationStereo @ 6e880681 (SimFoundry
  FOUNDATIONSTEREO_COMMIT), no setup.py: imported from the checkout via
  sys.path (`core.foundation_stereo`, `core.utils.utils.InputPadder`).
- Patch `patches/offline_init.patch` (ours): model construction needed the
  network: `torch.hub.load('facebookresearch/dinov2', ...)` (GitHub clone)
  and `timm.create_model('edgenext_small', pretrained=True)` (HF download).
  Both sets of weights are overwritten by the FoundationStereo checkpoint,
  so the patch loads DINOv2 code from the vendored `dinov2/` hub dir
  (source='local') and sets timm pretrained=False. Inference unchanged.
- Weights: official 23-51-11 (ViT-L) is an unversioned Google Drive folder
  (gdown in SimFoundry download_checkpoints.sh; Drive is unreachable from the
  H20). Pinned an HF mirror (pablovela5620/foundation-stereo @ 560e9077,
  cfg.yaml + model_best_bp2.pth); its sha256 60e79bde...87c5f1 is identical
  in 5 independent HF mirrors.
- flash-attn: SimFoundry installs it next to FoundationStereo, but no
  FoundationStereo module imports it; omitted. xformers is optional in the
  vendored DINOv2 (falls back to torch attention).
- Inference = SimFoundry FoundationStereoBackend / upstream run_demo.py:
  optional downscale (`scale`, SimFoundry 0.5), InputPadder(divis_by=32),
  autocast, forward(iters=valid_iters, test_mode=True) or run_hierachical.
  The node resizes the disparity back to the input size (x W/w) so depth
  matches the given camera. depth = fx * baseline / disparity;
  remove_invisible (x - disp < 0) -> 0, depth <= 0 or > max_depth (100 m,
  SimFoundry clip) -> 0. Raw disparity (px, unmasked) is the `disparity_px`
  file output (there is no metric-disparity type; `disparity` is relative).
- Point clouds / denoising (open3d) of SimFoundry are dropped: use the depth
  + camera with a pointcloud tool downstream.
- Fixture: upstream assets left.png/right.png (960x540) with K.txt
  (fx 754.67, baseline 0.063 m).
- Licence: NVIDIA Source Code License (LICENSE in the upstream checkout),
  section 3.3: code, weights and derivatives for non-commercial (research)
  use only; 3.1: redistribution must include the licence. The fixture
  images are upstream assets/ (same licence; not committed, *.png ignored).

## the RTX 5090 host (RTX 5090 32 GB, sm_120) — first tested here
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build, has sm_120); lock
  regenerated. xformers absent -> vendored DINOv2 uses torch attention
  (harmless "xFormers is not available" warnings).
- Peak VRAM (nvidia-smi on GPU 0, 960x540, scale 1, 32 iters): ~5.2 GB;
  ~34 s cold run incl. model load (5 s), ~12 s per test case.
- Verification on the upstream demo pair (assets/left.png, right.png,
  K.txt, baseline 0.063 m): disparity 40-137 px, median depth 0.51 m
  (tabletop scene), valid 90.6% (9.4% removed by remove_invisible at the
  left border). Warping the right image by the predicted disparity gives
  mean |L - warp(R)| 7.5 grey levels vs 37.9 unwarped / 20.1 for a constant
  median shift; depth == fx*b/disp exactly. scale=0.5 vs full: abs_rel
  0.006. References tests/expected/demo_{depth,disparity_px}.npy are the
  full-scale the RTX 5090 host outputs (float32); disparity_px is compared with the
  depth_abs_rel comparator (relative error of positive values).

## 2026-10 real2sim round (metric checks, validity, sequences)

- New outputs `valid` (mask) and `info` (fx, baseline, doffs, input /
  inference size, K at inference size, disparity resample factor, invalid
  reasons), optional `lr_error_px` with `lr_check` (right-view disparity
  from the mirrored pair: d_R(x) = d'(W-1-x)), param `doffs`, optional
  `valid` input (rectification border). New task `estimate_depth_seq`.
- Demo fixture images are read from the upstream checkout (`{repo}/assets`),
  not committed (NVIDIA licence).
- Metric verification against ANALYTIC truth: synthetic fisheye rig of
  stereo_rectify (baseline 60.5 mm) rendered by the ideal rectified pinhole
  cameras: depth abs-rel 0.0027 (single, lr_check) / 0.0018 (seq, 2
  frames), valid 97 %. Distorted cameras are refused (negative test).
