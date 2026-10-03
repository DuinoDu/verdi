# void: notes

## Upstream use
- netflix/void-model @ e3914f8 (SimFoundry install_void.sh), with SimFoundry
  `patches/void-model.patch` (no sudo for rp auto-installs, libx264 bitrate fix).
- entry.py runs the upstream scripts per chunk exactly like SimFoundry auto_bg
  `2_run_void_pass1.py` (predict_v2v.py, quadmask config, void_pass1) and
  `3_run_void_pass2.py` (inference_with_pass1_warped_noise.py, void_pass2,
  RAFT warped noise), with the same chunk plans and cross-fades.
- Weights: CogVideoX-Fun-V1.5-5b-InP (base), netflix/void-model (public, not
  gated), torchvision raft_large (pinned url+sha256, linked into TORCH_HOME).
- rp `git_import('CommonSource')` would clone an unpinned repo at run time:
  a setup command pre-clones it at a pinned commit.

## Pitfalls
- rp auto-`pip install`s missing modules at run time; on offline hosts this
  fails: Pass 2 needs `py3nvml` (rp.get_gpu_count), now a declared dependency.
- ffmpeg is not installed on the RTX 5090 host: a setup command links the static ffmpeg of
  the pinned imageio-ffmpeg wheel into the venv bin (entry.py puts it on PATH).
- VRAM: peak ~20.6 GB on a 5090 with model_cpu_offload (Pass 1 VAE encode);
  OOMs if another ~15 GB job shares the GPU. min_vram_gb = 24.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.7.1 -> 2.8.0, torchvision 0.23.0, `sm_120` in gpu.arch, re-locked.
- Weights (~45 GB) were downloaded with parallel ranged requests from
  hf-mirror to local disk and copied into the bucket (bucket forbids rename),
  with `.pdebug_complete` markers so setup skips them.
- Test reference `tests/expected/bigben_inpainted` = the visually verified the RTX 5090 host
  run (tower and its reflection removed), JPEG q95. Re-runs on the same GPU
  scored 28.4 and 26.9 dB `image_psnr` (not bit-exact), threshold 25 dB.
