# render_depth — minimal interface plan (R2S-DIAG-20261005 v2-D; NOT implemented)

Gap shown by the diagnostic: no node persists a metric depth render of the posed mesh
in the input camera. The offline `render_depth_*.npy` in the R2S-DIAG report dir are a
diagnostic computation, not a node output.

## Fields (both nodes)
| output | type | content |
|---|---|---|
| `render_depth` | depth (`.npy` float32 HxW) | metric z-depth of the posed mesh along the optical axis of the SAME camera as the inputs (OpenCV axes, K of `camera`), 0 = no model surface (outside the render domain) |
| `render_depth.json` | json | `K`, `width`, `height`, `camera_axes: "OpenCV x right y down z forward"`, `depth_type: "z"`, `pose_source` (`sam3d_initial` / `foundationpose_refined`), `T_cam_obj`, `mesh` (path + sha256 of meshes_metric/obj_k.glb), `invalid_value: 0`, `compare_domain: "render_depth > 0 and input depth > 0"`, `residual_sign: "obs_z - render_z"`, input hashes (from provenance once SC-03 lands) |
| scale sidecar | `render_depth.npy.scale.json` | metric only if the node ran in metric mode (sam3d depth + camera; FP always metric inputs), else relative |

* sam3d_objects: new z-buffer raster of `meshes_metric` posed by the SAM3D pose (current
  `render_mask` is an instance silhouette by fillPoly without a z-buffer, so depth cannot
  be read from it); `render_mask` / `overlay` unchanged. Per mask_id; multi-object: one
  depth map with the nearest surface.
* foundationpose: `_Diag` already renders the depth (nvdiffrast) for its diagnostics: save
  it (`estimate`: one map; `track`: optional `render_depths` depth_seq behind a param, off
  by default to keep outputs small). `valid` / diagnostics unchanged; no new quality model.

## CPU-checkable acceptance (no GPU)
* shape / K / axes metadata equal the input camera; `render_depth > 0` ⇔ `render_mask > 0`
  (sam3d, same raster) up to rasterisation edge pixels;
* equivalence with the offline diagnostic renderer (`basket_ev.py` raster) on the saved
  pose + mesh within a stated tolerance;
* sidecar / scale_status rules; the compare-domain statistics reproduce FP's own
  depth_inlier_ratio from saved arrays.

## Dependencies / regression scope (GPU only in a coordinated later window)
No new dependency (cv2/numpy raster for sam3d; nvdiffrast already in FP). Regression:
sam3d_objects 2 fixture cases (~3 min) + foundationpose 5 cases (~2 min) on one free GPU.
Estimate 0.5 developer-day + that window. Not started in this batch.
