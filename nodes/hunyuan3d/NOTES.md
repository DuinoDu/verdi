# hunyuan3d node notes

## How upstream is used

- Upstream: Tencent-Hunyuan/Hunyuan3D-2.1 @ 82920d64 (same pin as
  SimFoundry `scripts/installation/install_hunyuan.sh`), checked out by
  setup into `$PDEBUG_HOME/repos/hunyuan3d`. `entry.py` puts
  `hy3dshape/`, `hy3dpaint/` and `hy3dpaint/custom_rasterizer/` on
  `sys.path` exactly like upstream `demo.py`.
- `patches/Hunyuan3D-2.1.patch` = SimFoundry `patches/Hunyuan3D-2.1.patch`
  (trimesh `simplify_quadric_decimation(face_count=...)`, pipeline
  `components`, `.to(device)` of the DiT output) plus two local changes:
  - `hy3dpaint/utils/multiview_utils.py`: accept a local directory for
    `multiview_pretrained_path` (upstream always calls
    `snapshot_download(repo_id=...)`, which bypasses our pinned weights).
  - `hy3dpaint/DifferentiableRenderer/mesh_utils.py`: `import bpy` made
    optional. bpy 4.0 (Blender, ~350 MB, cp310 only) is only used to
    convert the textured OBJ to GLB; entry.py does that conversion with
    trimesh instead (glTF PBR: baseColor + metallicRoughness, roughness in
    G and metallic in B), calling the pipeline with `save_glb=False`.
- Weights (all pinned, under `$PDEBUG_HOME/weights/hunyuan3d`):
  `tencent/Hunyuan3D-2.1` (only `hunyuan3d-dit-v2-1` and
  `hunyuan3d-paintpbr-v2-1`), `facebook/dinov2-giant` (paint conditioning),
  `RealESRGAN_x4plus.pth` (paint view super-resolution, sha256 pinned),
  `u2net.onnx` (rembg default session, sha256 pinned; `U2NET_HOME` is
  pointed at it so rembg never downloads).
- Shape loads through `smart_load_model`: an absolute path is used as is
  (`os.path.join(HY3DGEN_MODELS, abs)` returns `abs`).

## Build

- `[setup].commands` build `custom_rasterizer_kernel` in place
  (`python setup.py build_ext --inplace`) and the pybind11
  `mesh_inpaint_processor` with g++ (upstream `compile_mesh_painter.sh`
  uses `python3-config`, which is not the venv's python).
- The H20 `c++` is clang 11, which rejects the narrowing conversions in
  `grid_neighbor.cpp` / `rasterizer.cpp` (`-Wc++11-narrowing` errors):
  `[build.env]` sets `CC=gcc CXX=g++`.
- `basicsr==1.4.2` is sdist-only and imports torch in `setup.py`: built
  with `extra-build-dependencies` (torch matched to runtime), which needs
  static metadata, provided through `[[tool.uv.dependency-metadata]]`.
  The same mechanism drops `opencv-python` (clashes with the headless
  build), `tb-nightly`, and realesrgan's `gfpgan`/`facexlib` (only used
  by its face-restoration CLI).
- basicsr imports `torchvision.transforms.functional_tensor` (removed in
  torchvision 0.17); entry.py aliases it to `transforms.functional`
  (what upstream `torchvision_fix.py` does).
- numpy pinned to 1.26.4 (numpy-1 ABI for basicsr/pymeshlab/onnxruntime);
  deepspeed, gradio, bpy, cupy, open3d from upstream requirements are not
  needed for inference and are not installed.

## Conventions / pitfalls

- Meshes are NOT metric: the shape VAE decodes in a normalised box
  (`box_v = 1.01`, i.e. coordinates within about [-1, 1]), +Y up, object
  facing +Z. Paint keeps the vertex frame of its input mesh.
- `paint` remeshes to <= 40k faces and re-unwraps UVs (xatlas), so its
  output has a different topology than the input mesh.
- `shape`/`paint` run rembg automatically when the image has no
  transparent pixels (`remove_background=auto`), like upstream demo.py.

## the RTX 5090 host (RTX 5090 32 GB, sm_120)
- torch 2.8.0 (PyPI cu128) / torchvision 0.23.0; `custom_rasterizer` built
  with nvcc 12.8 for sm_120 (`CUDA_HOME={cuda_home}`,
  `TORCH_CUDA_ARCH_LIST={cuda_arch}` from the host config); g++ 11.4.
  basicsr builds against torch 2.8 unchanged. uv.lock re-generated (aliyun).
- VRAM: measured with `nvidia-smi -lms 500` over the full test (rembg,
  shape, paint with defaults: 6 views, 512 px, 4096 texture): peak
  21.6 GB, so paint fits one 32 GB 5090. `min_vram_gb` lowered 32 -> 24
  (doctor reports the 5090 as slightly under 32 GB, so 32 refused it).
- Real-ESRGAN / u2net url weights fetched offline (ubuntu -> bucket).
- Visual check on the RTX 5090 host: shape (shaded) and paint (texture sampled) outputs viewed front + side next to demo.png; both correct (penguin + HY3D sign, texture aligned).
