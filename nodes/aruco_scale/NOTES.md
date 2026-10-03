# aruco_scale

Algorithm node (no model, no weights, CPU only). Upstream = OpenCV
`cv2.aruco` from `opencv-contrib-python-headless==4.11.0.86` (PyPI; the
headless build has the same aruco module without GUI/libGL deps) and
`plyfile` for point-cloud transforms. Tested on the RTX 5090 host with `--device cpu`.

## Conventions
- OpenCV >= 4.6 board frame: origin top-left of the pattern, x right, y DOWN,
  z INTO the board (verified on 4.11: GridBoard marker 0 corners
  (0,0)..(L,L), ChArUco chessboard corner 0 at (sq, sq)). The node converts
  to its documented board frame: origin = pattern centre, x right, y up,
  z out of the printed side (`Board.to_world`). A board lying face-up gives
  z = up, so `board_orientation=horizontal` means gravity = -z.
- ChArUco marker count = floor(rows*cols/2) (white squares); boards printed
  with OpenCV < 4.6 and an even row count need `legacy_pattern=true`.
- Trajectory output contains only frames where the pose was accepted, with
  `frame_index` / `frames` / `timestamps` (= frame_index, same as
  mast3r_slam). `scale_align` matches frames by `frame_index`, else integer
  stems of `frames` (vggt writes names), else position.

## Pose estimation
`solvePnPGeneric(SOLVEPNP_IPPE)` gives both planar solutions; each is
refined with `solvePnPRefineLM` and the lower reprojection error wins.
Grids also use `refineDetectedMarkers` with the board.

## Fixture (tests/fixture/make_fixture.py, reproducible)
Images are ray-cast per pixel through the full OpenCV camera model
(distortion k1=-0.08, k2=0.03, p1/p2 small; 4x supersampling) onto the
board plane, plus vignetting and noise, so the test checks against ground
truth poses (tests/expected/*_gt*.json):
- charuco: 7x5 squares, 40/30 mm, DICT_5X5_100, 10 frames, frame 6 looks
  away (no board). Observed 0.1-0.7 mm translation error, 0.04 px reproj.
- apriltag: 4x3 DICT_APRILTAG_36h11, 50 mm tags, 15 mm gap, ids 10..21,
  vertical board. Marker-corner refinement matters: none 2-7 mm, subpix
  1-5 mm (0.35 px), contour 2.4-3.5 mm, apriltag 0.2-4 mm (0.14 px) ->
  `corner_refinement=auto` uses apriltag for apriltag grids, subpix else.
  ChArUco chessboard corners are far more accurate than marker corners:
  prefer ChArUco boards for metric scale.
- scale_align: source = GT in the frame-0 camera frame divided by 0.4, with
  1 mm / 0.2 deg noise, all 10 frames; reference =
  `charuco_board_trajectory.json` (the verified detect_board output, 9
  frames). Recovered scale 0.4015 (0.36 % error from the noise over a
  ~0.5 m baseline), centre rmse 1.3 mm; frame 6 (not in the reference) is
  placed within 1.2 mm of its GT. method=orientation gives the same scale.
- Overlays (axes at the board centre, x red right, y green up) were checked
  visually.

## Pitfalls
- `cv2.undistortPoints` has no `criteria` kwarg in 4.11; use
  `cv2.undistortPointsIter` (fixture renderer).
- Video input relies on the core runner's ffmpeg split; ffmpeg is NOT
  installed on the RTX 5090 host (`which ffmpeg` empty), so pass image_seq dirs there.
- Umeyama on (nearly) collinear camera centres leaves the rotation about
  the motion line undetermined (common for hand-held side-steps):
  `method=auto` then takes the rotation from the camera orientations.
