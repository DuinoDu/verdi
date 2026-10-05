# stereo_rectify node notes

- Deterministic OpenCV algorithm node (CPU): cv2.fisheye.stereoRectify /
  cv2.stereoRectify with CALIB_ZERO_DISPARITY, initUndistortRectifyMap,
  remap. Calibration convention = cv2.stereoCalibrate (X_right = R X_left + T).
- SIFT epipolar check on frame 0 (ratio test 0.7, inliers |dy - median| < 3 px)
  reports median |dy|; the run fails above max_epipolar_dy_px (default 1 px).
- Fixture (tests/fixture/make_fixture.py): ray-cast textured box room with a
  table, Kannala-Brandt fisheye pair 640x360 per eye (f ~205 px, ~100 deg
  half-diagonal FOV), baseline 60.5 mm, 0.6 deg relative rotation, 2 SBS
  frames. expected/ holds the SAME scene rendered directly by the ideal
  rectified pinhole cameras + analytic z-depth (used by foundation_stereo).
  Result: remap vs direct render PSNR 37.2 / 37.1 dB, median |dy| 0.054 px;
  R perturbed by 1.7 deg -> |dy| 6.4 px -> refused (negative test).
- Not done here: calibration itself, temporal sync, other camera models
  (omnidirectional / double-sphere): convert to OpenCV fisheye first.
