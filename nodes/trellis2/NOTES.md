# trellis2 node notes

## How upstream is used

- Upstream microsoft/TRELLIS.2 @ 75fbf018 (same pin as SimFoundry
  `scripts/installation/install_trellis.sh`), checked out with submodules
  (o-voxel's eigen, from gitlab.com, reachable from the H20). `entry.py`
  puts the checkout on `sys.path` (SimFoundry `Trellis2.set_repo_path`)
  and runs `Trellis2ImageTo3DPipeline` + `o_voxel.postprocess.to_glb` with
  the SimFoundry `Trellis2.generate_mesh` settings (aabb [-0.5,0.5]^3,
  decimation 300k, remesh, remesh_project 0.9, `mesh.simplify(16777216)`).
- The CUDA extensions are uv git dependencies pinned to the SimFoundry
  commits: nvdiffrast v0.4.0 (253ac4f), nvdiffrec renderutils (b296927,
  package `nvdiffrec_render`), CuMesh (12289e1, with cubvh submodule),
  FlexGEMM (6dd94a8), o-voxel (subdirectory of the TRELLIS.2 commit),
  utils3d (9a4eb15). Each has static `[[tool.uv.dependency-metadata]]`
  so `uv lock` never builds them, and `extra-build-dependencies` with
  `torch` `match-runtime = true` so they compile against torch 2.7.1
  (cu126) with nvcc 12.9 inside the build isolation. o-voxel's own
  unpinned `cumesh @ git+...` / `flex_gemm @ git+...` requirements are
  overridden by its static metadata.
- `pipeline.json` of TRELLIS.2-4B references the sparse-structure decoder
  as `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16` and
  DINOv3 as an HF repo id. entry.py writes a temporary pipeline.json
  pointing every model at the pinned local weight dirs, so nothing is
  fetched at run time.
- `rembg.BiRefNet` is replaced by a stub before `from_pretrained`: the
  upstream background remover is briaai/RMBG-2.0 (gated, CC-BY-NC-4.0).
  SimFoundry does the same (`_RejectingBackgroundRemover`). Inputs must be
  RGBA or image + mask; the RGBA goes through upstream `preprocess_image`
  (alpha branch: crop to alpha bbox, premultiply).

## DINOv3 weights (ModelScope mirror)

- `facebook/dinov3-vitl16-pretrain-lvd1689m` is gated on HuggingFace
  (manual approval by Meta); the configured HF token got 403.
- With the user's explicit approval the files are downloaded from the
  ModelScope mirror of the same repo
  (`modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m`).
  `model.safetensors` is pinned by sha256
  dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179, which
  is the official HF LFS sha256, so the file is bit-identical to Meta's.
  config.json / LICENSE.md are pinned by sha256 too.
- **Meta's DINOv3 License still applies to anyone using this node.**
  To use the official channel, get HF access and switch
  `[weights.dinov3]` to `hf = "facebook/dinov3-vitl16-pretrain-lvd1689m"`.

## Build / runtime pitfalls

- Host `c++` is clang 11: `[build.env]` sets `CC=gcc CXX=g++`.
- nvdiffrec links `-lcuda`: `LIBRARY_PATH=/usr/local/cuda-12.9/lib64/stubs`.
- flash-attn (upstream default attention) has no prebuilt wheel for
  torch 2.7 at the upstream version, and a source build is very slow;
  xformers (0.0.32.post2 for torch 2.8.0 on the RTX 5090 host) is used instead
  (`ATTN_BACKEND=xformers`, supported by both dense and sparse attention).
  Sparse convolutions use FlexGEMM (upstream default).
- pillow-simd (upstream) is replaced by plain pillow (speed only).
- hf-mirror sometimes times out on the 2.5 GB checkpoints
  (`ReadTimeout`) or answers 429 (rate limit); rerunning
  `otn-cli setup trellis2` resumes (`HF_HUB_DOWNLOAD_TIMEOUT=120` helps).
- `pipe.low_vram` defaults to True upstream (moves sub-models CPU<->GPU);
  the node sets it from the `low_vram` param (default false, H20 has
  enough memory).

## Conventions

- Output glb: TRELLIS.2 works Z-up in [-0.5,0.5]^3; `to_glb` converts to
  glTF Y-up (`(x, y, z) -> (x, z, -y)`). Not metric.

## the RTX 5090 host (RTX 5090 32 GB, sm_120): first real test of this node
- torch 2.8.0 (PyPI cu128) / torchvision 0.23.0 / xformers 0.0.32.post2.
  Every CUDA extension (nvdiffrast, nvdiffrec, CuMesh + cubvh, FlexGEMM,
  o-voxel) builds with nvcc 12.8 for sm_120 (`CUDA_HOME={cuda_home}`,
  `TORCH_CUDA_ARCH_LIST={cuda_arch}`, `LIBRARY_PATH={cuda_home}/lib64/stubs`
  for nvdiffrec's `-lcuda`).
- xformers 0.0.32 turns on its FlashAttention-3 kernels (Hopper sm_90a only)
  for every GPU with compute capability >= 9.0, and these abort on sm_120
  ("CUDA error ... invalid argument"). entry.py `_no_fa3_off_hopper()`
  turns FA3 off on anything that is not sm_90, so FA2 / cutlass is used.
- Git sources on the offline host: TRELLIS.2 has a gitlab submodule
  (o-voxel/third_party/eigen, also cubvh/third_party/eigen inside CuMesh).
  The the RTX 5090 host mirror has `gitlab.com/libeigen/eigen.git` (full) and
  `github.com/microsoft/TRELLIS.2.git`. TRELLIS.2 was first fetched shallow
  and then deepened to the full history of the pinned commit 75fbf01.
  Shallow mirrors break `git clone --filter=blob:none`, which the [upstream]
  checkout uses. Nested submodules need the rewrite through
  `GIT_CONFIG_GLOBAL` (core 2d0b295), because git drops GIT_CONFIG_* env for
  submodules of submodules.
- DINOv3 comes from ModelScope (reachable from the RTX 5090 host), sha256-pinned to the
  official HF LFS hash. See the DINOv3 section above for the licence.
- Fixture `tests/fixture/example.png` = upstream `assets/example_image/T.png`
  (RGBA, MIT), downscaled to 768 px. The H20 fixture was never committed.
- Test (pipeline_type 512, 100k faces, 1024 texture): 210 s, peak VRAM
  18.9 GB on the 5090. Visual check: the glb rendered front, side and back
  next to the input. The steampunk "T" is reproduced with its texture
  (brass gears, glass tubes), +Y up, front towards +Z.
