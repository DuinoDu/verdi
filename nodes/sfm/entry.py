"""CPU Structure-from-Motion (pycolmap 4.2.1, COLMAP SIFT + incremental mapping)
-> undistorted frames with ONE shared pinhole K, T_world_cam per registered frame,
sparse points. Scale is relative (SfM gauge); nothing is interpolated.

Conventions:
* T_world_cam = inverse(COLMAP cam_from_world), OpenCV camera axes (x right,
  y down, z forward) - COLMAP uses the same axes, so no axis change.
* world = the gauge of the selected COLMAP model (arbitrary origin / rotation /
  scale); scale_status = relative everywhere.
* All frames share one camera (CameraMode.SINGLE). Undistortion maps it to one
  PINHOLE camera; the undistorted frames keep the poses (COLMAP undistortion only
  re-projects pixels).
"""
from __future__ import annotations

import os

# The pycolmap wheel bundles an OpenBLAS built for <= 24 threads; with COLMAP's
# matcher threads x OpenMP on a many-core host it overflows its thread table
# (observed: SIGSEGV in blas_memory_alloc, and silently corrupted models before
# that). BLAS runs single-threaded; COLMAP parallelises over its own workers.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

PYCOLMAP = "4.2.1"
WHEEL_SHA256 = "7627f0bab5746d04d94f91fec0dcb4b8ae0d5592853ab7b549cbd7dd021dd90f"  # cp311 manylinux_2_28 (uv.lock)
WORLD = ("gauge of the selected COLMAP incremental model: arbitrary origin, rotation and "
         "scale (relative); OpenCV camera axes")


# ---------------------------------------------------------------- helpers
def _call(x):
    return x() if callable(x) else x


def camera_to_colmap(cam: Dict[str, Any], w: int, h: int):
    """verdi camera json -> (COLMAP model name, params list, record)."""
    if (int(cam["width"]), int(cam["height"])) != (w, h):
        raise NodeError(f"camera is {cam['width']}x{cam['height']} but frames are {w}x{h}; "
                        "intrinsics are not rescaled implicitly")
    K = np.asarray(cam["K"], dtype=float)
    if abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9 or abs(K[2, 2] - 1) > 1e-9:
        raise NodeError("camera K has skew / is not normalised; COLMAP models have no skew")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = [float(x) for x in np.asarray(cam.get("dist", []), dtype=float).ravel()]
    model = str(cam.get("model", "pinhole")).lower()
    if model == "fisheye":
        if len(d) != 4:
            raise NodeError(f"fisheye camera needs dist = [k1, k2, k3, k4] (Kannala-Brandt / "
                            f"cv2.fisheye), got {len(d)} values")
        return "OPENCV_FISHEYE", [fx, fy, cx, cy] + d, {"model": "OPENCV_FISHEYE", "dist": d}
    if model not in ("pinhole", "opencv"):
        raise NodeError(f"camera model {model!r} not supported (pinhole / opencv / fisheye)")
    if not d or max(abs(x) for x in d) == 0.0:
        return "PINHOLE", [fx, fy, cx, cy], {"model": "PINHOLE", "dist": []}
    d = (d + [0.0] * 5)[:max(5, len(d))]
    if len(d) > 5 or abs(d[4]) > 0:
        raise NodeError(f"distortion {d} has k3 / rational terms; only OPENCV "
                        "(k1, k2, p1, p2) or fisheye (k1..k4) are supported",
                        hint="FULL_OPENCV is not in scope; pass k3 = 0 or a fisheye model")
    return "OPENCV", [fx, fy, cx, cy] + d[:4], {"model": "OPENCV", "dist": d[:4]}


def _features_per_image(db: Path) -> Dict[str, Optional[int]]:
    try:
        con = sqlite3.connect(str(db))
        rows = con.execute("SELECT images.name, keypoints.rows FROM images LEFT JOIN "
                           "keypoints ON images.image_id = keypoints.image_id").fetchall()
        con.close()
        return {n: (int(r) if r is not None else 0) for n, r in rows}
    except sqlite3.Error:
        return {}


def _num_db_cameras(db: Path) -> Optional[int]:
    try:
        con = sqlite3.connect(str(db))
        n = con.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        con.close()
        return int(n)
    except sqlite3.Error:
        return None


def _T_world_cam(img) -> np.ndarray:
    cfw = _call(img.cam_from_world)
    M = np.eye(4)
    M[:3, :4] = np.asarray(cfw.matrix(), dtype=float)
    return np.linalg.inv(M)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """dst ~ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def pose_eval(T_pred: Dict[int, np.ndarray], ref_path: Path, n_frames: int) -> Dict[str, Any]:
    ref = io.read_json(ref_path)
    Ts = [np.asarray(T, dtype=float) for T in ref["T_world_cam"]]
    idx = ref.get("frame_index") or list(range(len(Ts)))
    if len(idx) != len(Ts):
        raise NodeError("reference_poses: frame_index and T_world_cam differ in length")
    R_ = dict(zip([int(i) for i in idx], Ts))
    common = [i for i in sorted(T_pred) if i in R_]
    if len(common) < 3:
        raise NodeError(f"reference_poses overlap only {len(common)} registered frames (< 3)")
    P = np.array([T_pred[i][:3, 3] for i in common])
    G = np.array([R_[i][:3, 3] for i in common])
    s, R, t = umeyama(P, G, True)
    err = np.linalg.norm((s * (R @ P.T)).T + t - G, axis=1)
    rot = []
    for i in common:
        Rp = R @ T_pred[i][:3, :3]
        c = np.clip((np.trace(Rp.T @ R_[i][:3, :3]) - 1) / 2, -1, 1)
        rot.append(float(np.degrees(np.arccos(c))))
    return {"note": "alignment of the SfM camera centres to an EXTERNAL trajectory (Sim3, "
                    "Umeyama); a consistency metric against that reference, not an "
                    "accuracy statement beyond it",
            "frames": common, "n_frames_input": n_frames,
            "ate_rmse_after_sim3": float(np.sqrt((err ** 2).mean())),
            "ate_max_after_sim3": float(err.max()),
            "sim3_scale_ref_per_pred": s,
            "rotation_err_deg_after_sim3_median": float(np.median(rot)),
            "rotation_err_deg_after_sim3_max": float(np.max(rot)),
            "per_frame_translation_err": [float(e) for e in err]}


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["r"], rec["g"], rec["b"] = rgb.T
        f.write(rec.tobytes())


# ------------------------------------------------------------------- task
def reconstruct(ctx: Context) -> None:
    import pycolmap

    if pycolmap.__version__ != PYCOLMAP:
        raise NodeError(f"pycolmap {pycolmap.__version__} != pinned {PYCOLMAP}", kind="setup")
    t_all = time.time()
    frames_dir = Path(ctx.input("frames"))
    files = io.list_frames(frames_dir)
    if len(files) < 3:
        raise NodeError(f"need >= 3 frames, got {len(files)}")
    if any(f.parent != frames_dir for f in files):
        raise NodeError("frames must be files directly inside the frames directory")
    shapes = {}
    for f in files:
        im = io.read_image(f)
        shapes.setdefault(im.shape[:2], []).append(f.name)
    if len(shapes) != 1:
        raise NodeError(f"frames have {len(shapes)} resolutions {sorted(shapes)}; one shared "
                        "camera (one K) per call", hint="split per camera / resolution")
    (h, w), = shapes.keys()
    names = [f.name for f in files]

    if ctx.has_input("cameras"):
        cams = io.read_json(ctx.input("cameras"))["cameras"]
        Ks = {tuple(np.round(np.asarray(c["K"], float).ravel(), 9)) + tuple(c.get("dist", []))
              for c in cams}
        if len(Ks) != 1:
            raise NodeError(f"cameras has {len(Ks)} different intrinsics; this node supports "
                            "ONE shared camera (per-frame K is not in scope)",
                            hint="pass `camera` (shared) or split the call per camera")
        cam_in = dict(cams[0])
    elif ctx.has_input("camera"):
        cam_in = io.read_json(ctx.input("camera"))
    else:
        cam_in = None
    refine_p = str(ctx.param("refine_intrinsics", "auto"))
    if cam_in is not None:
        model, params, cam_rec = camera_to_colmap(cam_in, w, h)
        refine = refine_p == "true"
        cam_rec.update(source="input camera", params=params)
    else:
        if refine_p == "false":
            raise NodeError("refine_intrinsics=false needs a given camera (without one the "
                            "intrinsics must be estimated)")
        model = str(ctx.param("camera_model", "OPENCV"))
        params, refine = None, True
        cam_rec = {"model": model, "source": "estimated by COLMAP (focal prior "
                   "1.2 x max(w, h)); refined in bundle adjustment"}

    work = ctx.output_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    nthreads = int(ctx.param("num_threads", 0)) or min(os.cpu_count() or 1, 16)
    seed = int(ctx.param("seed", 0))

    # 1. features (SIFT, CPU)
    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = model
    if params is not None:
        reader.camera_params = ",".join(repr(float(p)) for p in params)
    ext = pycolmap.FeatureExtractionOptions()
    ext.use_gpu = False
    ext.num_threads = nthreads
    ext.max_image_size = int(ctx.param("max_image_size", 3200))
    ext.sift.max_num_features = int(ctx.param("max_num_features", 8192))
    dsp = bool(ctx.param("sift_dsp", False))
    ext.sift.domain_size_pooling = dsp
    ext.sift.estimate_affine_shape = dsp
    pycolmap.extract_features(db, frames_dir, image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext, device=pycolmap.Device.cpu)
    nfeat = _features_per_image(db)
    n_db_cams = _num_db_cameras(db)
    if n_db_cams is not None and n_db_cams != 1:
        raise NodeError(f"feature database has {n_db_cams} cameras; expected one shared camera")
    t_feat = time.time() - t0
    if nfeat and sum(nfeat.values()) == 0:
        raise NodeError("no SIFT features in any frame (blank / constant / unreadable images)",
                        hint="check the frames; nothing to reconstruct")

    # 2. matching
    t0 = time.time()
    mo = pycolmap.FeatureMatchingOptions()
    mo.use_gpu = False
    mo.num_threads = nthreads
    guided = bool(ctx.param("guided_matching", True))
    mo.guided_matching = guided
    matcher = str(ctx.param("matcher", "exhaustive"))
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=mo, device=pycolmap.Device.cpu)
    else:
        po = pycolmap.SequentialPairingOptions()
        po.overlap = int(ctx.param("sequential_overlap", 10))
        po.loop_detection = False
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po,
                                  device=pycolmap.Device.cpu)
    t_match = time.time() - t0

    # 3. incremental mapping
    t0 = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = nthreads
    opts.random_seed = seed
    opts.multiple_models = True
    opts.min_model_size = 3
    opts.ba_refine_focal_length = refine
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = refine
    models_dir = work / "models"
    models_dir.mkdir()
    recs = pycolmap.incremental_mapping(db, frames_dir, models_dir, options=opts)
    t_map = time.time() - t0
    if not recs:
        raise NodeError("incremental mapping produced no model (not enough matched / "
                        "geometrically verified views)",
                        hint=f"features per frame: {nfeat}")
    order = sorted(recs, key=lambda k: -recs[k].num_reg_images())
    best = recs[order[0]]
    models = [{"model": int(k), "num_reg_images": int(recs[k].num_reg_images()),
               "num_points3D": int(recs[k].num_points3D())} for k in order]
    if best.num_reg_images() < 3:
        raise NodeError(f"largest model registers only {best.num_reg_images()} frames (< 3)")
    if best.num_cameras() != 1:
        raise NodeError(f"model has {best.num_cameras()} cameras; one shared camera expected")
    other = {}
    for k in order[1:]:
        for iid in _call(recs[k].reg_image_ids):
            other[recs[k].image(iid).name] = int(k)

    reg = {}  # name -> image
    for iid in _call(best.reg_image_ids):
        im = best.image(iid)
        reg[im.name] = im
    ratio = len(reg) / len(names)
    min_ratio = float(ctx.param("min_registered_ratio", 0.0))
    if ratio < min_ratio:
        raise NodeError(f"only {len(reg)}/{len(names)} frames registered ({ratio:.2f} < "
                        f"min_registered_ratio {min_ratio})")
    sparse = ctx.output_path("sparse")
    best.write(sparse)
    ctx.set_output("sparse", sparse)
    cam = next(iter(best.cameras.values()))
    cam_rec["refined_params"] = [float(x) for x in cam.params]
    cam_rec["refine_intrinsics"] = refine
    cam_rec["model_after_ba"] = cam.model_name if hasattr(cam, "model_name") else str(cam.model)

    # 4. undistortion -> one PINHOLE camera
    t0 = time.time()
    und = work / "undistorted"
    uo = pycolmap.UndistortCameraOptions()
    uo.blank_pixels = float(ctx.param("undistort_blank_pixels", 0.0))
    reg_names = [n for n in names if n in reg]  # input order
    pycolmap.undistort_images(und, sparse, frames_dir, image_names=reg_names,
                              output_type="COLMAP", undistort_options=uo)
    urec = pycolmap.Reconstruction(str(und / "sparse"))
    ucams = list(urec.cameras.values())
    if len(ucams) != 1:
        raise NodeError(f"undistortion produced {len(ucams)} cameras; one shared K expected")
    ucam = ucams[0]
    Kp = np.asarray(ucam.calibration_matrix(), dtype=float)
    uw, uh = int(ucam.width), int(ucam.height)
    t_und = time.time() - t0

    out_frames = ctx.output_path("frames")
    T_by_index, Ts, fidx = {}, [], []
    for n in reg_names:
        src = und / "images" / n
        if not src.exists():
            raise NodeError(f"undistorted image {n} missing")
        shutil.copyfile(src, out_frames / n)
        i = names.index(n)
        T = _T_world_cam(reg[n])
        T_by_index[i] = T
        Ts.append(T.tolist())
        fidx.append(i)
    ctx.set_output("frames", out_frames)
    io.write_camera(ctx.output_path("camera", "camera.json"), Kp, uw, uh)
    ctx.set_output("camera", ctx.output_dir / "camera.json")
    io.write_json(ctx.output_path("trajectory", "trajectory.json"), {
        "T_world_cam": Ts, "frame_index": fidx, "frames": reg_names,
        "units": "relative (arbitrary SfM scale)", "scale_status": "relative",
        "world": WORLD, "pose_source": "pycolmap incremental SfM (CPU); unregistered frames "
        "have no pose (not interpolated)",
        "convention": "T_world_cam = inverse(COLMAP cam_from_world), OpenCV axes"})
    ctx.set_output("trajectory", ctx.output_dir / "trajectory.json")

    pts = list(best.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=np.float64).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)
    pp = ctx.output_path("points", "points.ply")
    _write_ply(pp, xyz, rgb)
    io.write_scale(pp, "relative", "arbitrary (SfM gauge)", "pycolmap incremental SfM",
                   note="same world as trajectory.json")
    ctx.set_output("points", pp)

    regrec = []
    for i, n in enumerate(names):
        r = {"frame_index": i, "frame": n, "registered": n in reg,
             "n_features": nfeat.get(n)}
        if n in reg:
            im = reg[n]
            r["n_points3D"] = int(_call(im.num_points3D))
        elif n in other:
            r["reason"] = f"registered only in separate model {other[n]} (not merged; no pose)"
        else:
            r["reason"] = "not registered by incremental mapping (no pose, not interpolated)"
        regrec.append(r)
    io.write_json(ctx.output_path("registration", "registration.json"), {
        "frames": regrec, "num_registered": len(reg), "num_frames": len(names),
        "registered_ratio": ratio, "partial": len(reg) < len(names)})
    ctx.set_output("registration", ctx.output_dir / "registration.json")

    info = {
        "pycolmap": pycolmap.__version__, "wheel_sha256": WHEEL_SHA256,
        "device": "cpu (pycolmap built without CUDA)" if not pycolmap.has_cuda else "cpu (forced)",
        "scale_status": "relative", "world": WORLD,
        "camera_in": cam_rec,
        "camera_out": {"model": "PINHOLE", "K": Kp.tolist(), "width": uw, "height": uh,
                       "dist": [0, 0, 0, 0, 0],
                       "note": "one shared undistorted K for every output frame (COLMAP "
                               "undistortion, blank_pixels = %.2f)" % uo.blank_pixels},
        "camera_mode": "SINGLE (one camera for all frames)",
        "matcher": matcher, "models": models, "selected_model": models[0],
        "num_points3D": int(best.num_points3D()),
        "mean_reprojection_error_px": float(best.compute_mean_reprojection_error()),
        "mean_track_length": float(best.compute_mean_track_length()),
        "num_registered": len(reg), "num_frames": len(names),
        "params": {"matcher": matcher, "max_image_size": ext.max_image_size,
                   "max_num_features": ext.sift.max_num_features, "num_threads": nthreads,
                   "openblas_threads": os.environ["OPENBLAS_NUM_THREADS"],
                   "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                   "sift_dsp": dsp, "guided_matching": guided, "seed": seed, "refine_intrinsics": refine,
                   "min_registered_ratio": min_ratio,
                   "undistort_blank_pixels": uo.blank_pixels},
        "timing_sec": {"features": round(t_feat, 2), "matching": round(t_match, 2),
                       "mapping": round(t_map, 2), "undistortion": round(t_und, 2),
                       "total": round(time.time() - t_all, 2)},
        "determinism": "multi-threaded COLMAP is not bit-reproducible; seed fixes the "
                       "RANSAC / mapper seeds only",
        "not_provided": ["metric scale", "per-frame intrinsics", "poses of unregistered frames"],
    }
    io.write_json(ctx.output_path("info", "info.json"), info)
    ctx.set_output("info", ctx.output_dir / "info.json")
    if ctx.has_input("reference_poses"):
        io.write_json(ctx.output_path("pose_eval", "pose_eval.json"),
                      pose_eval(T_by_index, ctx.input("reference_poses"), len(names)))
        ctx.set_output("pose_eval", ctx.output_dir / "pose_eval.json")
    if not bool(ctx.param("keep_work", False)):
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main({"reconstruct": reconstruct}))
