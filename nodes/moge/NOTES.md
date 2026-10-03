# moge

- Upstream: microsoft/MoGe @ 925b8ed835a7a9cdb7578ba15c658a0afc969030 (last
  MoGe-2-era commit, before the MoGe-3 "V3" refactor that adds
  FlexGEMM/gradio deps), installed as a uv git dependency (no checkout);
  utils3d @ 3fab839f. Only `moge.model.v2.MoGeModel.infer` is used; the app /
  training deps are dropped via `tool.uv.dependency-metadata`.
- Licence: MoGe code MIT (Microsoft), MoGe-2 weights MIT; DINOv2 backbone
  Apache-2.0.
- Weights (HF, pinned revisions in the manifest), `model.pt` sha256:
  - vitl `Ruicheng/moge-2-vitl-normal` 280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01
  - vitb `Ruicheng/moge-2-vitb-normal` 16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6
  - vits `Ruicheng/moge-2-vits-normal` 79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc
- Conventions: metres, OpenCV camera frame. MoGe's intrinsics are normalised
  (fx/W, fy/H, cx/W, cy/H); converted to pixels. The point map is
  projection-consistent (upstream `force_projection`): on the fixture
  P.z == depth exactly and P.xy == depth * K^-1 [u+0.5, v+0.5, 1] to 4e-7 m.
  Normals are unit length (mean |n| 1.0000) and point towards the camera.
- A known camera only fixes the horizontal FOV (fx); cx/cy/distortion are
  ignored (MoGe assumes a centred principal point).
- Visual check (bedroom/000000, vitl): floor nearest at the bottom with
  up-facing normals, far wall at the left-centre, walls with opposite x
  normals; mask is all-valid (indoor scene, no sky).
- References `tests/expected/bedroom0_{depth,normal}.npy` (float16) come from
  the the RTX 5090 host vitl fp16 run; rerun: depth abs_rel 0.0002, normal angle 0.13 deg.

## Comparison with depth_pro (bedroom/000000, same fixture)
- Estimated focal: MoGe-2 vitl 938.5 px (fov_x 54.2 deg), Depth Pro 797.9 px.
- Median depth: MoGe 2.770 m, Depth Pro 2.711 m (MoGe ~2% farther).
- Per-pixel vs the depth_pro reference: abs_rel 0.068; 0.051 after median
  scale alignment; inverse-depth correlation 0.949.
- With the known camera (fx = 1000): MoGe vits median 3.483 m, Depth Pro
  3.397 m.
- `estimate_seq` (median FOV over 4 frames): per-frame fov_x 53.7..54.6 deg
  (1.5% spread), median depth 2.76 m.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0+cu128 / torchvision 0.23.0 (PyPI), Python 3.12. No custom CUDA
  extensions; no code changes needed for sm_120.
- Peak VRAM (one 960x540 image, resolution_level 9, fp16, torch max
  allocated / reserved): vitl 1.39 / 2.26 GB, vits 0.78 / 1.08 GB.
- Timings in `otn-cli test` (includes model load): image 24 s (first run,
  cold weight cache), image_known_camera 4 s, seq (8 forward passes) 8 s.
- Note: `otn-cli test --device cuda:N` sets CUDA_VISIBLE_DEVICES itself; an
  outer CUDA_VISIBLE_DEVICES is overridden.
