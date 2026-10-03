# segment_anything (SAM v1)

- Upstream: HF `transformers` `SamModel`/`SamProcessor` (transformers 4.57.1,
  locked) instead of the original `segment_anything` repo; weights are the HF
  conversions `facebook/sam-vit-{base,large,huge}` pinned by revision
  (only `model.safetensors` + json configs are downloaded).
- Default checkpoint is `huge` (original SAM quality); legacy used `base`.
- `segment_image`: image embedding computed once, then one decoder pass per
  obj_id (points + box of an obj_id are one prompt). Multimask off by
  default (SAM recommends single-mask output for box / multi-point prompts).
- `auto_masks`: transformers `mask-generation` pipeline (no crop layers,
  `points_per_crop` = points_per_side). It has no `min_mask_region_area`
  post-processing like the original AMG; `min_area` just drops small masks.
  Masks overlap; they are painted largest first so the id map keeps
  nested parts (later/smaller wins).
- Legacy node fed 3 fixed points (center + 2 diagonals) and returned 3 RLE
  masks in a json: that behaviour is covered by `segment_image` with an
  explicit prompt_set; the RLE/Lance output format was dropped.
- torch 2.7.1 (PyPI cu126 build).

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build); uv.lock regenerated
  on the RTX 5090 host (aliyun index); `sm_120` added to gpu.arch. No code changes; the
  H20 references / checks pass unchanged on sm_120.
- Pitfalls: an old H20 uv.lock left on the RTX 5090 host makes `uv sync --frozen` fetch
  from the unreachable H20 PyPI mirror (delete it and re-lock); hf-mirror
  redirects Xet-backed files to an unreachable CDN (HF_HUB_DISABLE_XET=1,
  now set by core) and sometimes answers 429 on the tree API (retry later).
