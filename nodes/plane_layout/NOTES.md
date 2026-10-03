# plane_layout

Algorithm node (no model, no weights, CPU only): `open3d-cpu==0.19.0` from
PyPI (CPU wheel, manylinux_2_31; 0.20 needs glibc 2.35). Tested on the RTX 5090 host
with `--device cpu`.

## Algorithm
1. Points: ply, or depth + camera unprojected in the OpenCV camera frame
   (iterative undistortion of the 5-coefficient model in numpy; camera
   rescaled if only the resolution differs). Voxel downsample (0.02 m).
2. Iterative `segment_plane` (seed fixed via `o3d.utility.random.seed`;
   two runs gave byte-identical planes.json), SVD least-squares refit,
   re-collect inliers, keep the largest DBSCAN component (eps 3 voxels).
3. Normals oriented towards the input origin (camera), so d > 0.
4. Floor = lowest plane within `gravity_tol_deg` (45) of the gravity hint
   (default +y = upright OpenCV camera), facing up, with extent product
   >= 0.5 m^2 and <= 2 % of the points below it. Its normal defines up.
   15 deg was too strict: a hand-held camera pitched 20 deg down already
   failed, hence the separate, wide `gravity_tol_deg`.
5. World: z up, origin = foot of the input origin on the floor, x =
   camera forward projected on the floor. Labels from orientation, height
   and area in that frame.
6. `anchor=support|auto`: table-top views without floor anchor the world
   on the largest up-facing horizontal plane (labelled table).

## Fixtures
- room.ply (make_fixture.py): synthetic room in a camera frame (1.5 m
  height, 20 deg pitch, 5 deg roll), 5 mm noise, 1 % outliers. Recovered:
  camera height 1.4999 m, up vector within 1e-4, table 0.7498 m, ceiling
  2.5998 m, 2 walls, box faces not mislabelled.
- bedroom_depth.npy: depth_pro reference depth of the bedroom frame
  (`nodes/depth_pro/tests/expected/bedroom0_depth.npy`, 2x subsampled) +
  its camera. The floor is NOT visible in this image (bed fills the bottom),
  so the default anchor=floor correctly fails; the test uses anchor=auto ->
  bed = support plane, camera 0.54 m above it, walls found. Labelled cloud
  projected on the image was checked visually (bed red, walls blue; the
  striped back wall is split into several near-parallel "other" pieces
  because monocular depth bends large planes).
