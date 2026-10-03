# mast3r_slam node notes

## Upstream use
- rmurai0610/MASt3R-SLAM @ e6f4e3d4 (`[upstream]`, no submodules). mast3r,
  dust3r/croco and asmk are vendored in that repo; its only submodules are
  Eigen (gitlab, unreachable from the RTX 5090 host) and pyimgui (GUI only).
- Eigen headers come from PyPI `cmeel-eigen` (3.4); setup symlinks them to
  `thirdparty/eigen` and to lietorch's `eigen/`.
- lietorch (princeton-vl @ e7df8655, the unpinned git dependency in upstream's
  pyproject) is cloned in setup (GitHub mirror) and built in place. It is not
  a uv git dependency because uv would try to clone its gitlab Eigen
  submodule.
- All extensions are built in place (`build_ext --inplace`): lietorch, curope,
  asmk.hamming, mast3r_slam_backends. entry.py puts these dirs on sys.path, so
  `uv sync` never removes them.
- Weights: the three naver checkpoints (MASt3R ViT-L metric 512 catmlpdpt,
  retrieval trainingfree, codebook) are `url + sha256` entries. the RTX 5090 host cannot
  reach download.europe.naverlabs.com, so they were fetched on ubuntu with
  `~/bin_fetchthe RTX 5090 host.sh` into `<weights>/mast3r_slam/<key>/`. Upstream finds the
  codebook from the retrieval checkpoint's name, so entry.py symlinks both
  into one temp dir.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set in entry.py: the checkpoints
  pickle argparse Namespaces, and torch 2.8 loads with weights_only by default.
  The files are trusted because their sha256 is pinned.

## patches/build_archs.patch (torch 2.8 / sm_120)
- setup.py hard-codes `-gencode` sm_60..86, so it can't build for sm_120. The
  patch drops that list so that TORCH_CUDA_ARCH_LIST (`{cuda_arch}`) applies.
  `has_cuda` also accepts FORCE_CUDA=1.
- curope setup.py: same fix (drops `get_gencode_flags`, which builds every
  arch).
- `x.type()` -> `x.scalar_type()` in AT_DISPATCH (curope kernels.cu,
  matching_kernels.cu); `torch::linalg::linalg_norm` -> `at::linalg_norm`
  in gn_kernels.cu. Without these the code does not compile with torch 2.8.

## entry.py vs upstream main.py
- One process and no GUI. The driver runs upstream's main loop, `run_backend`
  and `relocalization` synchronously, which is upstream's `single_thread`
  behaviour: after each new keyframe, retrieval + factor-graph construction +
  global GN. Functions are copied because main.py imports the GUI (in3d/imgui).
  `SharedKeyframes` gets an in-process stand-in for mp.Manager, and
  `share_memory_` is skipped while it is created (no /dev/shm needed).
- Calibration follows `dataloader.Intrinsics.from_calib` (undistort to
  `getOptimalNewCameraMatrix`, alpha 0, principal point centred) and is
  reimplemented in entry.py. The reason: `mast3r_slam.dataloader` imports
  pyrealsense2.
- Per-frame trajectory (upstream only saves keyframe poses): each tracked
  frame stores its Sim3 pose relative to its reference keyframe
  (`T_CkCf`). At the end that pose is composed with the keyframe's final
  optimised pose, so loop-closure corrections reach every frame. Sim3 ->
  SE3 via upstream `as_SE3` (the scale is dropped; the camera centre is the
  translation). Frames lost before relocalisation get no pose. They are
  listed in `metadata.lost_frames`; nothing is fabricated.
- Point cloud = upstream `save_reconstruction` (per keyframe, average conf >
  1.5, rays constrained in calibrated mode), randomly capped at max_points.
  Optional per-keyframe depth = z of the keyframe pointmap, mapped back
  through upstream's resize+crop to the input resolution (0 outside the crop).
- Overflowing the keyframe buffer (`max_keyframes`, upstream 512) raises
  NodeError and does not crash.

## Validation (the RTX 5090 host, RTX 5090, cuda:7)
- Fixture: TUM RGB-D fr1_desk (CC BY 4.0), every 3rd frame (205 frames,
  ~10 Hz), resized to 512x384 (the network resolution), JPEG q80, 6.0 MB.
  The camera json holds TUM fr1 intrinsics x0.8 and the distortion.
  Reference = TUM mocap GT (`tests/expected/gt_trajectory.json`, nearest
  stamp).
- Results (ATE = RMSE after Sim3 alignment, metres):
  | run | all frames | keyframes |
  |---|---|---|
  | fixture, calibrated | 0.021 (205/205 frames) | 0.015 (14 kf) |
  | fixture, uncalibrated | 0.068 (204/205, frame 119 lost, 1 reloc) | 0.101 (16 kf) |
  | fr1_desk 640x480, subsample 2, calibrated (upstream eval setting) | 0.034 | 0.018 |
  | fr1_room 640x480, 1362 frames, calibrated | 0.065 | 0.066 (55 kf) |
  | fr1_room 640x480, 1362 frames, uncalibrated | 0.110 | 0.113 (53 kf) |

  Paper (keyframes): fr1_desk calibrated 0.016 / uncalibrated 0.035; fr1_room
  0.061 / 0.118. The port reproduces upstream accuracy.
- Visual check (/tmp/h20w/mast3r_slam_vis.jpg locally): aligned
  trajectories against GT (the calibrated one lies on GT, the uncalibrated
  one follows it with jitter), a top view of the fused cloud (desks,
  monitors, chairs consistent, no double walls), and a keyframe depth map.
- A 320x240 fixture (first attempt) tracked much worse: 0.19 m uncalibrated,
  0.06 m calibrated, because MASt3R upsamples to 512. Use at least ~512 px
  input.
- Runs are deterministic on one GPU (same ATE on repeated runs). Both tests
  compare against the full 205-frame GT; core `trajectory_ate` matches poses
  by `frame_index` (the uncalibrated run loses frame 119 before a
  relocalisation).

## Runtime / VRAM (RTX 5090, 512x384 network resolution)
- fr1_room, 1362 frames 640x480: SLAM 76.8 s uncalibrated (17.7 fps), 87.0 s
  calibrated (15.7 fps), plus ~7 s model load and ~20 s otn-cli/venv overhead.
  That is about 56 s (uncalibrated) / 64 s (calibrated) of SLAM per 1000
  frames on a hand-held indoor walk (~40 keyframes per 1000 frames).
- Peak VRAM: 8.3 GB allocated / 9.3-9.6 GB reserved (nvidia-smi ~10.2-10.5
  GB) for 1362 frames. On the 205-frame fixture it is already 8.1 GB, so the
  memory is mostly fixed: model ~2.7 GB plus the pre-allocated 512-keyframe
  buffer (~9 MB per keyframe at 512x384, ~4.5 GB). Growth with sequence
  length is only the factor-graph edges (~5 MB each, ~1-4 per keyframe).
  Frames are streamed from disk. Thousands of frames fit in 32 GB. For
  >512 keyframes, raise `max_keyframes` or `subsample`.
