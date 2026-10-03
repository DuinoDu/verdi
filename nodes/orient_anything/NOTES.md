# orient_anything

## How upstream is used
- Upstream https://github.com/SpatialVision/Orient-Anything @ 759282c2
  as `[upstream]` checkout (not a package). entry.py imports its
  `vision_tower.DINOv2_MLP`, `inference.get_3angle(_infer_aug)` and
  `utils.background_preprocess` (cwd temporarily set to the repo because
  utils.py loads `./assets/axis.obj` at import).
- Checkpoint `ronormsigma1/dino_weight.pt` (the one used by the current
  upstream app.py, out_dim 360+180+360+2) from Viglong/Orient-Anything,
  backbone `facebook/dinov2-large`, both pinned by HF revision. The
  backbone path is injected by setting `paths.DINO_LARGE` before
  importing vision_tower (no patch needed).
- Legacy used `croplargeEX2` with out_dim 360+180+180+2; the current
  upstream checkpoint has 360 rotation bins, so rotation is in
  [-180, 180) (upstream UI label "-90~90" is stale).
- deps: torch 2.7.1, transformers 4.46.3, rembg 2.0.61 + onnxruntime
  (utils.py imports rembg unconditionally); `numba>=0.60` constraint is
  needed, otherwise uv picks numba 0.53 (py<3.10 only) via pymatting.
- rembg's u2net.onnx is pinned as a url weight (sha256) and found through
  `U2NET_HOME` so rembg does not download at run time.

## Angle semantics / R_cam_obj
- Derived from upstream `get_proj2D_XYZ` (its 2D axis drawing):
  front axis projects to (-sin az cos g - cos az sin p sin g, ...). The
  3D form used here is R_S = Rroll(-rot) Rx(polar) Ry(az) in a screen
  frame (x right, y up, z to viewer), with object axes front/left/up =
  S z / x / y at zero angles; numerically checked to reproduce upstream's
  2D projections for random angles. R_cam_obj = diag(1,-1,-1) R_S B
  (OpenCV), object frame x = front, y = left, z = up (right-handed).
- Verified visually on the truck fixture: front axis points to the truck
  front (image right, azimuth ~261 = 270 - 9), up axis vertical.
- R_cam_obj ignores perspective (object assumed near the optical axis);
  no translation is produced, so no pose_set output (would need depth).

## Pitfalls
- `test_time_aug` uses random crops; seeded with 0 per object so it is
  reproducible. Upstream TTA always removes the background for half of
  the crops.
- First run after setup is slow (~3 min, cold weight cache); warm ~40 s
  mostly model loading.
- Interactive gradio app and the rendered-axis video of legacy dropped.
