# sfm — registration notes (SC-01)

> **Scope (VD-SCOPE-20261005):** SfM is handed over to the consumer. This node is kept as is for reuse; it is not a Verdi deliverable or acceptance item, and no further business development is committed.


## Source (pinned)
| item | value |
|---|---|
| pycolmap | 4.2.1, `pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl`, sha256 `7627f0bab5746d04d94f91fec0dcb4b8ae0d5592853ab7b549cbd7dd021dd90f` (uv.lock; the entry refuses any other version) |
| build | `pycolmap.has_cuda == False`: CPU only |
| licence | COLMAP / pycolmap BSD-3-Clause; the wheel bundles third-party libraries (their licence files ship in the wheel) |

## Contract
* One camera for all frames (CameraMode.SINGLE); given intrinsics fixed unless
  `refine_intrinsics="true"`; models PINHOLE / OPENCV (k3 = 0) / OPENCV_FISHEYE (camera json
  `"model": "fisheye"`, dist = k1..k4); k3 / rational / per-frame K refused.
* Output frames = undistorted REGISTERED frames with ONE pinhole K (dist = 0); trajectory
  T_world_cam = inverse(cam_from_world), OpenCV axes, `scale_status = relative`, world = COLMAP
  gauge; points sidecar relative. Unregistered frames: listed in registration.json with a
  reason, no pose, nothing interpolated. Multiple COLMAP models are not merged (largest kept;
  frames only in other models reported).
* Errors (status error, `verdi run` rc = 1): < 3 frames, mixed resolution, no SIFT features
  (e.g. all-black input), no model, < 3 registered, ratio < `min_registered_ratio`, > 1 camera.
* BLAS pinned to 1 thread: the wheel's bundled OpenBLAS overflowed its thread table on the
  063 host (SIGSEGV in `blas_memory_alloc`; before the crash fix the same setting produced a
  model with one camera 0.70 m off on the TUM fixture).

## Fixture evidence (CPU only, 063, 2026-10-06)
| case | result |
|---|---|
| TUM fr1_desk 26 frames, known pinhole K | 24/26 registered (frames 0, 1 not), Sim3 ATE vs mocap 0.0146 m (repeats 0.0129 / 0.0131), max 0.021 m |
| same frames re-rendered through a synthetic OPENCV_FISHEYE camera (`tests/fixture/make_fixture.py`, 52 % of fisheye pixels have content) | 24/26 registered, ATE 0.0219 m (repeats 0.0193 / 0.0193), max 0.046-0.053 m; output one pinhole K |
| all-black / mixed resolution / k3 / differing per-frame K | refused with NodeError |
| sfm outputs -> splatfacto `_load_inputs` + `resolve_scale` (CPU, no training) | 24 frames / 24 poses accepted; scale resolves to `relative` |

ATE is an alignment metric against the external TUM mocap trajectory of this public fixture
only; it is not an accuracy statement for real2sim / SceneAgent scenes. Without guided
matching only 18/26 registered (default guided_matching = true). splatfacto training on
sfm outputs (GPU) has NOT been run.
