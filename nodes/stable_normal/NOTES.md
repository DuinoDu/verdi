# stable_normal

## How upstream is used
- Upstream https://github.com/hugoycj/StableNormal pinned at 1b7203c9 as
  `[upstream]` checkout; `hubconf.StableNormal` / `StableNormal_turbo`
  build the pipelines from local weight dirs (`local_cache_dir =
  $weights`, keys named like the HF repos `yoso-normal-v1-5`,
  `stable-normal-v0-1`, fp16 variants only).
- Its `setup.py` installs the full gradio app requirements (torch 2.0.1,
  diffusers 0.28, transformers 4.36, xformers ...), so it is NOT
  pip-installed; own minimal deps: torch 2.7.1, diffusers 0.30.3,
  transformers 4.46.3, accelerate 1.1.1.
- The pipeline is called directly (not `Predictor.__call__`) to get float
  normals instead of an 8-bit PNG; image resized with upstream
  `resize_image` (longer side 1024, multiple of 64), output bilinearly
  resized to input size and re-normalised.

## Convention (verified)
- StableNormal's raw output is x LEFT, y up, z towards the viewer (not
  the Marigold docs' x-right). Verified on the bedroom fixture against
  normals computed from depth_pro metric depth: median cosine 0.33 with
  (x, -y, -z) and 0.97 with (-x, -y, -z). The node therefore returns
  (-x, -y, -z) = OpenCV camera frame (x right, y down, z forward).
  Checks on the output: bed top (0.03, -0.97, -0.18) (points up = -y),
  back wall z = -0.82 (faces camera). turbo: median cosine 0.95.

## Pitfalls
- `DINOv2_Encoder` (prior of the full model) calls
  `torch.hub.load('facebookresearch/dinov2')`, which needs GitHub access.
  `patches/0001-local-dinov2.patch` loads it from a pinned dinov2 checkout
  (`[setup].commands`, commit 7764ea0f, cloned into the weights dir) with
  `source="local", pretrained=False` plus the pinned
  `dinov2_vitl14_pretrain.pth` (sha256-checked). The patch also drops a
  stray `import pdb`.
- Upstream's outdoor/object masking (Mask2Former / BiRefNet downloaded at
  runtime) is not exposed; RGBA alpha masking is kept (background = zero
  vector).
- First run after setup took ~13 min (cold vepfs page cache reading ~6 GB
  of weights); warm runs are ~21 s (full) / ~7 s (turbo) on H20.
- xFormers is not installed (warnings only).
- Legacy `stable_normal` (image2normal.py) used turbo and wrote 8-bit
  PNGs; now default is the full model and output is float .npy.
