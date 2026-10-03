# foundpose

## How upstream is used

- Upstream `facebookresearch/foundpose@3103473b` (+ submodules
  `external/dinov2@e1277af2`, `external/bop_toolkit@e7ba9f23`), cloned into
  `$PDEBUG_HOME/repos/foundpose`; repo root and `external/dinov2` are put
  on `sys.path`.
- The upstream scripts (`gen_templates.py`, `gen_repre.py`, `infer.py`)
  are tied to the BOP dataset layout (bop_toolkit config paths, test
  targets, CNOS detections). `entry.py` contains thin ports of their main
  loops for one mesh / one image and calls the upstream `utils/` modules
  for everything else (renderer, crop cameras, DINOv2 extractor, PCA,
  faiss k-means, tf-idf, cyclic-buddy matching, PnP). Options default to
  the released `configs/*/lmo.json`.
- Units: upstream is millimetre-based (BOP); the node feeds the metric
  mesh to the pyrender rasteriser directly (it expects metres internally),
  keeps cameras/depth in mm inside, and converts poses/depth to metres at
  its boundary.
- DINOv2 weights: official `dinov2_vits14_reg4_pretrain.pth` from
  dl.fbaipublicfiles.com (url + sha256). The dinov2 hub code would
  download it itself; entry.py points `torch.hub` at a temp dir that
  symlinks the pinned file.
- `estimate` with `mesh` onboards on the fly (798 templates + repre);
  pass `templates` from `render_templates` to skip that.

## Pitfalls

- NVIDIA EGL on the H20 creates a context but rasterises nothing
  (colour all 0, depth = znear everywhere), for any pyrender scene. The
  Mesa llvmpipe EGL device works. entry.py probes each EGL device in a
  subprocess (switching EGL devices inside one process fails with
  EGL_BAD_ACCESS) and uses the first that renders correctly;
  `EGL_DEVICE_ID` overrides. With llvmpipe 798 templates at 4x SSAA take
  a few minutes (CPU).
- `patches/renderer_zfar.patch`: upstream sets the pyrender camera
  `zfar=3000.0` (metres); changed to 100 m for float depth precision.
- Core `checkout()` returns early when a fresh clone is already at the
  pinned commit and never runs `git submodule update`; done in
  `[setup].commands` instead.
- dl.fbaipublicfiles.com is very slow/unstable from the H20 (first try
  ~25 kB/s, a resumed try finished in seconds).
- `faiss-gpu-cu12` (PyPI) aborts on the H20 (`Faiss assertion
  err__ == cudaSuccess` in runL2Norm). The node uses `faiss-cpu` and runs
  upstream `cluster_util.kmeans` on CPU tensors (upstream would use the
  GPU because the features live on CUDA). k-NN indices were CPU upstream
  already.
- Coarse stage only (upstream does not release the featuremetric
  refinement). Expect cm-level translation error and occasional
  symmetric/flipped solutions on weakly textured objects.

## Dropped legacy manifests

`foundpose_to_linemod`, `templates_to_linemod` (pure format converters);
`foundpose_subprocess` (wrapper around an external script) is replaced by
`estimate`; `cad_to_templates` is `render_templates`.

## Fixture

mustard0 frame 0 from the official FoundationPose demo (same as the
foundationpose node fixture): rgb jpg, mask, camera, textured mesh glb.
`tests/expected/mustard_pose.json` is this node's own (visually checked)
estimate. Against the FoundationPose RGB-D estimate of the same frame it
is off by 8.4 cm translation (almost all along the viewing ray, i.e.
depth/scale ambiguity of RGB-only PnP at 0.8 m) and 7.8 deg rotation; the
2D silhouettes coincide. `tests/expected/fp_reference_pose.json` keeps the
FoundationPose pose for reference.
Timing on the H20 (llvmpipe rendering, CPU k-means): onboarding 798
templates ~15 min (render ~10 min, DINOv2 features, k-means ~3 min);
matching + PnP < 5 s.

## the RTX 5090 host (RTX 5090, sm_120) migration

- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128). No CUDA extensions.
- NVIDIA EGL works on the RTX 5090 host (unlike the H20). The probe now tries the EGL
  device of the assigned GPU first (`CUDA_VISIBLE_DEVICES`), so rendering
  runs on that GPU: 798 templates now take ~1 min instead of ~10.
- The relocked pyglet (2.1.x, imported by pyrender) loads its X11 backend,
  which needs libXrender (not installed on the RTX 5090 host) -> `PYGLET_HEADLESS=true`
  in `[run].env`.
- Submodules (`external/dinov2`, `external/bop_toolkit`) come through the
  file:// GitHub mirror; git needs `protocol.file.allow=always` for that
  (core b7322db sets it).
- The estimate is not bit-reproducible on GPU: two the RTX 5090 host runs differ by
  ~1.5 cm / 12 deg (k-means init, RANSAC, rendering). The estimate checks
  now compare against the independent RGB-D FoundationPose pose
  (`fp_reference_pose.json`, max 6 cm / 15 deg; the RTX 5090 host runs: 3.3 cm / 6.3 deg
  and 1.9 cm / 8.5 deg). Overlay checked: silhouette matches the bottle.
  `mustard_pose.json` = first the RTX 5090 host estimate (kept for reference only).
