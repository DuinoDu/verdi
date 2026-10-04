# genpose2 node notes

## How upstream is used

- `[upstream]` Omni6DPose/GenPose2 @ d0993c0 (2025-08). entry.py imports
  `runners/infer.py:GenPose2` (score -> energy -> scale networks, upstream
  defaults: ODE sampler, T0 0.55, 50 samples, clustering) and
  `datasets/datasets_infer.py:InferDataset` in-process, with `sys.argv`
  emptied around construction (upstream `get_config()` parses argv).
- Mask: unified masks use 0 = background, GenPose++ uses 255 = background
  (it treats every other value, including 0, as an object) -> remapped.
- Output pose is the upstream 4x4 (canonical Omni6DPose object frame ->
  camera, metres); `size` is the ScaleNet box length, `bbox3d_cam` the 8
  corners. Category is not predicted (labels are `instance_<id>`).
- `cutoop==0.1.0` (Omni6DPose API) from PyPI provides data types/drawing.
- Checkpoints: only on Dropbox (folder link in the upstream README).
  `[setup]` downloads the folder zip, unpacks it into
  `$VERDI_HOME/weights/genpose2/ckpts` and verifies per-file sha256
  (`ckpts.sha256`). Dropbox is blocked on the H20: the files were
  downloaded on another machine and copied into that directory; setup then
  only verifies them. The DINOv2 ViT-S/14 tensors are part of the
  checkpoints (strict load), so only the DINOv2 *code* is needed: pinned
  checkout facebookresearch/dinov2 @ 7764ea0 in
  `weights/genpose2/dinov2_repo`, loaded with `torch.hub.load(...,
  source='local', pretrained=False)`.

## Pitfalls / fixes

- Patch `infer_offline.patch`: drop `pyrealsense2` / `flask` imports from
  `runners/infer.py` (camera demo only); DINOv2 from a local pinned
  checkout instead of the unpinned torch.hub GitHub download (+ its
  pretrained-weight download from fbaipublicfiles, slow from China).
- The Dropbox folder zip contains a `/` entry; `unzip` returns non-zero
  ("stripped absolute path"), so setup ignores the unzip status and relies
  on the sha256 check.
- torch >= 2.6 `weights_only=True` default -> `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- Depth beyond 4 m is zeroed by upstream; the node raises if an instance
  has < 10 valid depth pixels instead of hitting upstream `ipdb.set_trace()`.
- Legacy features dropped: video/tracking (prev-pose init, ICP tracking),
  rerun/vis video, sam6d-json read/write bridges (`save_to_sam6d`,
  `sam6d_mask_path`), vggt camera conversion. Use the `mask` input
  (e.g. sam6d's mask output) instead.
- Test fixture = SAM-6D example (LINEMOD watering can) with the sam6d
  node's mask; GenPose++ translation agrees with the sam6d pose within
  ~3 mm and the box/axes match the can visually.

## the RTX 5090 host (RTX 5090, sm_120) migration

- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128); pointnet2 rebuilt for
  sm_120 (`{cuda_home}`, `{cuda_arch}`).
- Checkpoints are now a `[weights.ckpts]` entry (Dropbox folder zip,
  `archive = "zip"`, per-file `files_sha256`; no archive sha256 because
  Dropbox builds the zip on the fly). The curl/unzip setup command is gone
  (setup must not download). Dropbox is unreachable from the RTX 5090 host and from the
  ubuntu fetch host; the sha256-identical H20 copy was placed in
  `<weights>/genpose2/ckpts/` with a `.verdi_extracted` marker, so setup
  only verifies it.
- The pinned DINOv2 code checkout moved from `{weights}/dinov2_repo` to
  `{venv}/dinov2_repo`: the the RTX 5090 host weights bucket refuses rename/unlink, so
  git cannot work there. A broken partial clone remains at
  `<weights>/genpose2/dinov2_repo` (it cannot be deleted; unused).
- the RTX 5090 host test: translation within 0.1 mm of the H20 reference.
