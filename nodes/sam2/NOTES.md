# sam2

- Upstream: `sam-2` (facebookresearch/sam2 @ 2b90b9f) as a uv git
  dependency; SAM2_BUILD_CUDA=0 (optional connected-components kernel not
  built). Weights: HF `facebook/sam2.1-hiera-{tiny,small,base-plus,large}`.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build); uv.lock regenerated
  on the RTX 5090 host (git dep served by the local GitHub mirror); `sm_120` added to
  gpu.arch. No code changes; H20 references pass (mask_iou >= 0.9995).
