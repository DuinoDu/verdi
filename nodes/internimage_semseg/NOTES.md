# internimage_semseg

- Upstream: OpenGVLab/DCNv4 @ 4b848f7 checked out as [upstream];
  `segmentation/` (mmseg 0.27-style code: mmseg_custom, mmcv_custom,
  configs) is put on sys.path. The DCNv4 CUDA op is the same commit's
  `DCNv4_op` subdirectory as a uv git dependency, built against the runtime
  torch (`extra-build-dependencies` with `match-runtime`; needs a
  `[[tool.uv.dependency-metadata]]` entry for dcnv4 or uv refuses
  match-runtime). Build env: CUDA_HOME=/usr/local/cuda-12.9,
  TORCH_CUDA_ARCH_LIST=9.0 (setup ~9 min incl. weights).
- Stack: torch 2.7.1 (cu126), mmcv 1.7.2 *lite* (MMCV_WITH_OPS=0),
  mmsegmentation 0.30.0, mmdet 2.28.2 (mmseg_custom imports mmdet.utils),
  timm 0.9.16, yapf 0.40.1 (mmcv 1.x Config breaks on newer yapf), numpy<2,
  python 3.10.
- mmcv 1.7.2 is sdist-only and its setup.py imports pkg_resources:
  `build-constraint-dependencies = ["setuptools<70"]`, plus static
  dependency-metadata for mmcv (drops its opencv-python dep in favour of
  opencv-python-headless).
- No mmcv._ext: entry.py installs an `mmcv.ops` shim (exact torch
  point_sample; raising stubs for CrissCrossAttention, PSAMask, MSDA, ...)
  before importing mmseg. UPerNet + FlashInternImage never calls them.
  Mask2Former checkpoints (need MSDA from mmcv-full) are not exposed.
- torch>=2.6 defaults torch.load(weights_only=True); mmcv checkpoints carry
  pickled meta, so entry.py patches torch.load to weights_only=False
  (checkpoints are pinned HF files).
- Checkpoint `init_cfg` points at an HF URL for the ImageNet backbone; it is
  never fetched because init_segmentor does not call init_weights.
- mmseg wants BGR ndarray input (config to_rgb=True): entry passes rgb[..., ::-1].
- Label map: mask value = ADE20K class index + 1 (mmseg get_classes('ade20k')
  order, 150 classes; names stripped, e.g. "bed"). mask_iou on a full
  semseg map is meaningless (every pixel non-zero), so metric tests use the
  `classes` filter (car / person).
- Legacy `remove_dynamic` (fill class regions with their mean colour) is a
  task: trivial, and can reuse a mask from `semseg` or run it itself.
  Legacy vis output / video writer / cache flags dropped.
- Default model `s` (legacy "small" = upernet_flash_internimage_s_512);
  legacy "big" mapped to mask2former_l which is not supported (see above);
  upernet `l` @640 is the largest option.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128); [build.env] now uses
  `{cuda_home}` / `{cuda_arch}` (the RTX 5090 host: CUDA 12.8, arch 12.0); `sm_120` added
  to gpu.arch. DCNv4 builds fine for sm_120 with nvcc 12.8 (~8 min setup).
- DCNv4 `functions/flash_deform_attn_func.py` raises NotImplementedError at
  import when the GPU's compute capability is missing from its
  shared-memory table (no "12.0"). entry.py installs an import hook that
  rewrites that module's source to add `"12.0": 99000` (sm_120 opt-in
  shared memory per block = 101376 B, same as sm_86/89 -> same budget as
  upstream uses there). FlashDeformAttn is not used by UPerNet anyway;
  DCNv4Function results match the H20 references (mask_iou 1.0).
