# cotracker notes

## How upstream is used
- `cotracker` is installed from git (facebookresearch/co-tracker @
  82e02e8029753ad4ef13cf06be7f4fc5facdda4d, locked in uv.lock) and
  `cotracker.predictor.CoTrackerPredictor(checkpoint, offline=True,
  window_len=60)` is used directly (no torch.hub, so no network at run
  time). Weights: `facebook/cotracker3` `scaled_offline.pth` (pinned rev).
- The predictor resizes internally to 384x512 and rescales tracks to the
  input pixel size; visibility is upstream's `vis > 0.9` (query points are
  forced visible on their own frame).
- Whole sequence in one pass (offline model). The legacy code split videos
  into independent 60-frame chunks with re-seeded grids (no continuity);
  that is not reproduced. For long videos use `max_frames` or subsample.
- Query points (`track_points`) are prompt_set points; `frame` is the
  frame where `xy` holds. `obj_id` / `query_frame` are kept in the npz.

## Legacy manifests
- `cotracker` -> `track_grid` (grid / mask) and `track_points`.
- `video_kps_to_all` (CVAT keypoint annotations -> tracked keypoints pkl,
  cotracker2) is covered by `track_points`; the CVAT reader / pkl writer
  is a format converter and was dropped.
- Online model (`cotracker3_online`) not ported (legacy raised
  NotImplementedError for it on real input).

## Checks
No `tracks` comparison metric exists in core, so tests check frame/point
counts and visibility; the overlay of both test runs was inspected
(static points stay put, points on the moving girl follow her, queries
given on frames 3 and 5 are tracked backwards correctly).

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 (PyPI cu128 build) / torchvision 0.23.0; `sm_120` added to
  `gpu.arch`; uv.lock re-generated against the aliyun PyPI mirror.
- Output identical in structure to the H20 run (visible_ratio 0.9767 for the
  grid case, same as H20). Overlay re-inspected on the RTX 5090 host (static points fixed,
  points on the girl follow her). The the RTX 5090 host outputs are now references in
  `tests/expected/` (`grid_tracks.npz`, `points_tracks.npz`) with a
  `tracks_epe` check (max 1.5 px; core gained this metric after the H20 run).
