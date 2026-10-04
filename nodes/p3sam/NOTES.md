# p3sam: notes

## Role in SimFoundry
Stage 9 (`--detect-articulation`) runs `deps/articulate-anything/simfoundry/complete_workflow.py`
(fork `nadunRanawaka1/articulate-anything-sf`, checked at `dec3a618`). Step
`s3_segment_mesh` with `segment_method: hunyuan` (the default template,
`simfoundry/cfg/hunyuan_template.yaml`) calls `segment_mesh_hunyuan.py`:
`AutoMask(ckpt).predict_aabb(mesh, point_num=100000, prompt_num=400, prompt_bs=8,
threshold=0.85, post_process=True, seed=42, clean_mesh_flag=False)` and writes
`face2label.json` + a tab20-coloured `<obj>_segmented.glb`. This node covers that call.
Not covered (plain geometry in the fork, not a model): `smooth_segment_boundaries`
(3 iterations), `remove_small_disconnected_regions` (off by default),
`hierarchical_merge_segments` (max_segments=15), rendering/labelling of exploded views.

## Upstream use
- Hunyuan3D-Part at public commit `e96be06` (PR #15 merge). SimFoundry's
  `installation_hunyuan.sh` pins `df0c911`, which only exists in their old
  mirror; their own script falls back to HEAD + 3-way apply. The patch applies
  cleanly on `e96be06` (the public "switch to safetensors" merge).
- `patches/hunyuan-simfoundry.patch` is verbatim from the fork's `patches/`.
  It makes P3-SAM importable as a package, adds `clean_mesh_flag`, `prompt_bs`,
  `release()`, safetensors loading.
- No pip install of the upstream: entry.py puts `P3-SAM/` and `XPart/partgen/`
  on `sys.path` and imports `demo.auto_mask.AutoMask`.
- Sonata backbone: upstream calls `sonata.load("sonata", repo_id=..., download_root=~/.cache/sonata)`,
  a runtime HF download. entry.py wraps `models.sonata.load` to load the pinned
  `weights/sonata/sonata.pth` instead (same file, `facebook/sonata@df99897`).
- `is_parallel=False` (DataParallel on one GPU is identical; avoids wrapper).

## Build pitfalls
- flash-attn: no PyPI wheel. Uses the official v2.8.3 release wheel
  (cu12 / torch2.7 / cxx11abiTRUE / cp310). uv URL sources are not rewritten by
  the verdi GitHub proxy, so the URL carries the `gh-proxy.com` prefix.
- torch-scatter 2.1.2 is sdist-only: static `dependency-metadata` +
  `extra-build-dependencies` (torch match-runtime), compiled for sm_90 (~10 min).
- spconv-cu126 2.3.8 wheel works with torch 2.7.1 (cu126).
- gh-proxy returned HTTP 429 once during `git fetch`; re-running setup was enough.

## Fixture
`tests/fixture/beetle_car.glb`: upstream demo asset `P3-SAM/demo/assets/4.glb`
(500k faces, 30 MB) decimated to 40k faces with fast-simplification
(geometry only). A wheeled beetle car: wheels/legs/antennae/shell are
clear parts.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.7.1 -> 2.8.0 (cu128). flash-attn: GitHub release wheels are not
  reachable from the RTX 5090 host, so flash-attn 2.8.3 is built from the PyPI sdist
  (`FLASH_ATTENTION_FORCE_BUILD=TRUE`, `FLASH_ATTN_CUDA_ARCHS=90;120`,
  static dependency-metadata + extra-build-dependencies torch match-runtime);
  ~15 min with MAX_JOBS=32.
- spconv-cu126 2.3.8 works on sm_120 with torch 2.8 (no cu128 spconv wheel).
- torch-scatter built for `{cuda_arch}` (12.0).
- Hunyuan3D-Part mirror clone needed `http.version=HTTP/1.1` on the laptop
  (HTTP/2 framing errors).
- Visual check on the RTX 5090 host: 29-30 parts, wheels / legs / antennae / shell / pylon
  separated; num_parts varies by +-1 between runs (atomic ops), so the test
  checks a range.
