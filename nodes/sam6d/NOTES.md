# sam6d node notes

## How upstream is used

- `[upstream]` JiehongLin/SAM-6D @ 1c2543b (latest master, 2024-07). The
  three official demo steps (`SAM-6D/demo.sh`) are run as subprocesses of
  the node venv, exactly like the legacy `sam6d_docker` manifest did inside
  `lihualiu/sam-6d:1.0`, but without Docker:
  1. `blenderproc run Render/render_custom_templates.py` (42 templates)
  2. `Instance_Segmentation_Model/run_inference_custom.py --segmentor_model sam`
     (skipped when the optional `mask` input is given; the mask ids are
     written as `detection_ism.json` with score 1.0)
  3. `Pose_Estimation_Model/run_inference_custom.py`
- entry.py converts unified types to upstream conventions: mesh metres ->
  mm (`mesh_mm.ply`, or textured `.obj` for obj/glb), depth `.npy` metres ->
  uint16 mm png (depth_scale 1.0), camera -> `{cam_K, depth_scale}`;
  output `t` mm -> metres, `T_cam_obj` in the input mesh frame.
- Weights (HF mirrors of the official files):
  SAM ViT-H `ybelkada/segment-anything` (`sam_vit_h_4b8939.pth`, same sha as
  dl.fbaipublicfiles), DINOv2 ViT-L/14 `hdtech/dinov2_vitl14_pretrain`
  (size identical to the official file), PEM `jiiihuang/SAM-6D`
  (`sam-6d-pem-base.pth`; official copy is on Google Drive, unreachable
  from mainland China). Setup symlinks them into the paths the upstream
  scripts hard-code (`Instance_Segmentation_Model/checkpoints/...`,
  `Pose_Estimation_Model/checkpoints`).
- Blender 3.3.1 (the version blenderproc 2.6.1 hard-codes) is downloaded
  by `[setup]` from the TUNA mirror (sha256 pinned from
  download.blender.org) into `$VERDI_HOME/weights/sam6d/blender/` and
  used via `--custom-blender-path`.

## Pitfalls / fixes

- pointnet2 `setup.py` uses a relative include dir; with torch's ninja
  build that breaks (`ball_query.h not found`) -> patched to an absolute
  path. `build_ext --inplace` also needs `pointnet2/` to exist (mkdir).
- `trimesh==4.0.8` (upstream pin) is incompatible with numpy 2
  (`ndarray.ptp`) -> venv pins numpy 1.26.4.
- `blenderproc pip install trimesh` pulls numpy 2 into Blender's custom
  package dir, shadowing Blender 3.3's bundled numpy 1.22 -> setup pins
  `numpy==1.22.0` there as well.
- blenderproc resolves the script path with `$PWD`, not the process cwd ->
  absolute script path and `PWD` set per subprocess.
- torch >= 2.6 `weights_only=True` default -> `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  in `[run.env]` for the gorilla PEM checkpoint.
- Patch `pem_inference.patch`: PEM feature extractor `pretrained: False`
  (upstream would download MAE ViT-B init weights from fbaipublicfiles at
  every run; they are fully overwritten by the PEM checkpoint),
  `--det_score_thresh` parsed as float (upstream compares str vs float),
  `np.random.seed` (point sampling was unseeded -> run-to-run jitter),
  keep the caller's `CUDA_VISIBLE_DEVICES` (gorilla forced GPU 0).
- FastSAM segmentor not offered (needs ultralytics==8.0.135 + Google Drive
  weights); SAM ViT-H is the official demo default.
- Rendering ~1 min per call (templates are not cached between runs).

## the RTX 5090 host (RTX 5090, sm_120) migration

- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128); pointnet2 rebuilt for
  sm_120 (`{cuda_home}`, `{cuda_arch}`).
- Blender 3.3.1 is now a `[weights.blender]` url+sha256 entry (official
  download.blender.org tarball, same sha256 as before; the TUNA mirror and
  setup-time curl are gone: setup must not download). `[setup]` extracts
  it into `{venv}/blender/` (local disk): the weights bucket on the RTX 5090 host refuses
  rename/unlink, and `blenderproc pip install` writes into the Blender tree.
- FreeImage: raw.githubusercontent.com was unreachable from the ubuntu
  fetch host; the identical file (sha256 checked) was taken from the
  imageio/imageio-binaries git mirror.
- Blender's pip (`blenderproc pip install`) uses `PIP_INDEX_URL`, which
  core sets from the host's pypi index.
- the RTX 5090 host has no root/apt and lacks the X11 client libs Blender 3.3 links
  against (libXi, libXxf86vm, libXfixes, libXrender, libxkbcommon, libSM,
  libICE). They are pinned Ubuntu 22.04 debs (`[weights.deb_*]`, aliyun
  Ubuntu mirror + sha256), extracted by `[setup]` into
  `{venv}/blender/syslib` and put on `LD_LIBRARY_PATH` (build + run env).
- First setup on the RTX 5090 host takes ~10+ min: the blenderproc warm-up pip-installs
  blenderproc's own Blender-side packages (opencv-contrib 67 MB,
  scikit-learn...) from the aliyun mirror at ~0.4 MB/s.
- the RTX 5090 host test: 0.7 mm / 0.5 mm from the H20 references, mask IoU 1.0.
