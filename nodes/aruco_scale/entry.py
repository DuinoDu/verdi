"""aruco_scale node: metric camera poses from a printed marker board
(ArUco grid / ChArUco / AprilTag grid, OpenCV cv2.aruco) and similarity
alignment of up-to-scale trajectories to that metric board frame.

World frame of every output (the "board frame"):
  origin = centre of the board pattern, x = along the columns (to the right
  as printed), y = towards the first row (up as printed, marker/square row 0
  is the top row), z = x cross y = board normal pointing OUT of the printed
  side (towards a camera that sees the board).
OpenCV's own board frame (origin top-left, y down, z into the board) is
converted to this frame internally.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

BOARD_TYPES = ("aruco_grid", "charuco", "apriltag_grid")


# ------------------------------------------------------------------ board
class Board:
    """Board spec (json) -> cv2.aruco board + conversion to the world frame."""

    def __init__(self, spec: Dict[str, Any]):
        import cv2

        A = cv2.aruco
        self.spec = spec
        btype = spec.get("type")
        if btype not in BOARD_TYPES:
            raise NodeError(f"board.type {btype!r} not in {BOARD_TYPES}",
                            hint="see `verdi describe aruco_scale`")
        self.type = btype
        try:
            self.rows, self.cols = int(spec["rows"]), int(spec["cols"])
        except (KeyError, TypeError, ValueError):
            raise NodeError("board needs integer rows and cols") from None
        default_dict = ("DICT_APRILTAG_36h11" if btype == "apriltag_grid"
                        else None)
        dname = str(spec.get("dictionary") or default_dict or "")
        if not dname:
            raise NodeError("board.dictionary missing",
                            hint="e.g. DICT_5X5_100, DICT_4X4_50, "
                            "DICT_APRILTAG_36h11")
        if not dname.upper().startswith("DICT_"):
            dname = "DICT_" + dname
        names = {n.upper(): n for n in dir(A) if n.startswith("DICT_")}
        if dname.upper() not in names:
            raise NodeError(f"unknown dictionary {dname!r}",
                            hint="one of " + ", ".join(sorted(
                                n for n in names.values()
                                if n == n.upper() or "APRILTAG" in n)))
        self.dictionary_name = names[dname.upper()]
        self.dictionary = A.getPredefinedDictionary(
            getattr(A, self.dictionary_name))
        self.marker_length = _pos(spec, "marker_length_m")
        # ChArUco: one marker per white square = floor(rows*cols/2)
        n_markers = (self.rows * self.cols // 2 if btype == "charuco"
                     else self.rows * self.cols)
        if spec.get("ids") is not None:
            ids = np.asarray(spec["ids"], np.int32).ravel()
            if ids.size != n_markers:
                raise NodeError(f"board.ids has {ids.size} ids, the board "
                                f"has {n_markers} markers")
        else:
            ids = np.arange(n_markers, dtype=np.int32) + int(
                spec.get("first_id", 0))
        dict_size = self.dictionary.bytesList.shape[0]
        if ids.max() >= dict_size:
            raise NodeError(f"marker id {int(ids.max())} exceeds dictionary "
                            f"{self.dictionary_name} size {dict_size}",
                            hint="use a larger dictionary or lower first_id")
        self.ids = ids
        if btype == "charuco":
            self.square_length = _pos(spec, "square_length_m")
            if self.marker_length >= self.square_length:
                raise NodeError("charuco needs marker_length_m < "
                                "square_length_m")
            self.board = A.CharucoBoard((self.cols, self.rows),
                                        self.square_length,
                                        self.marker_length,
                                        self.dictionary, ids)
            self.board.setLegacyPattern(bool(spec.get("legacy_pattern")))
            self.width = self.cols * self.square_length
            self.height = self.rows * self.square_length
        else:
            self.separation = float(spec.get("separation_m", 0.0))
            if self.separation < 0:
                raise NodeError("board.separation_m must be >= 0")
            self.board = A.GridBoard((self.cols, self.rows),
                                     self.marker_length, self.separation,
                                     self.dictionary, ids)
            self.width = (self.cols * self.marker_length
                          + (self.cols - 1) * self.separation)
            self.height = (self.rows * self.marker_length
                           + (self.rows - 1) * self.separation)

    def to_world(self, pts_cv: np.ndarray) -> np.ndarray:
        """OpenCV board coords (origin top-left, y down, z in) -> world."""
        p = np.asarray(pts_cv, np.float64).reshape(-1, 3)
        return np.stack([p[:, 0] - self.width / 2,
                         self.height / 2 - p[:, 1], -p[:, 2]], 1)

    def describe(self) -> Dict[str, Any]:
        out = {"type": self.type, "dictionary": self.dictionary_name,
               "rows": self.rows, "cols": self.cols,
               "marker_length_m": self.marker_length,
               "ids": [int(i) for i in self.ids],
               "size_m": [self.width, self.height]}
        if self.type == "charuco":
            out["square_length_m"] = self.square_length
            out["legacy_pattern"] = bool(self.spec.get("legacy_pattern"))
        else:
            out["separation_m"] = self.separation
        return out


def _pos(spec: Dict[str, Any], key: str) -> float:
    try:
        v = float(spec[key])
    except (KeyError, TypeError, ValueError):
        raise NodeError(f"board.{key} (metres) missing") from None
    if not v > 0:
        raise NodeError(f"board.{key} must be > 0 metres")
    return v


# ------------------------------------------------------------- detection
def _detector_params(ctx: Context, board_type: str):
    import cv2

    A = cv2.aruco
    p = A.DetectorParameters()
    method = ctx.param("corner_refinement", "auto")
    if method == "auto":
        method = "apriltag" if board_type == "apriltag_grid" else "subpix"
    p.cornerRefinementMethod = {
        "none": A.CORNER_REFINE_NONE, "subpix": A.CORNER_REFINE_SUBPIX,
        "contour": A.CORNER_REFINE_CONTOUR,
        "apriltag": A.CORNER_REFINE_APRILTAG}[method]
    return p


def _frame_indices(files: List[Path]) -> List[int]:
    """Frame index = integer file stem (%06d) if all stems are integers."""
    try:
        return [int(f.stem) for f in files]
    except ValueError:
        return list(range(len(files)))


def _pnp(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """Planar PnP: IPPE candidates, keep the lower error, LM refine."""
    import cv2

    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not n:
        return None
    best = None
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv.copy(),
                                      tv.copy())
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2),
                             axis=1)
        if best is None or err.mean() < best[2].mean():
            best = (rv, tv, err)
    rv, tv, err = best
    R, _ = cv2.Rodrigues(rv)
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R
    T_cam_world[:3, 3] = tv.ravel()
    return T_cam_world, err


def _detect_frame(board: Board, detector, charuco_detector, gray: np.ndarray,
                  K: np.ndarray, dist: np.ndarray):
    """-> dict with marker detections and (optionally) the pose."""
    A = __import__("cv2").aruco
    rec: Dict[str, Any] = {"marker_ids": [], "marker_corners": []}
    if board.type == "charuco":
        ch_corners, ch_ids, m_corners, m_ids = charuco_detector.detectBoard(
            gray)
        if m_ids is not None:
            rec["marker_ids"] = [int(i) for i in m_ids.ravel()]
            rec["marker_corners"] = [c.reshape(4, 2).tolist()
                                     for c in m_corners]
        if ch_ids is None or len(ch_ids) == 0:
            rec["points"] = 0
            return rec, None, None
        rec["charuco_ids"] = [int(i) for i in ch_ids.ravel()]
        rec["charuco_corners"] = ch_corners.reshape(-1, 2).tolist()
        obj, img = board.board.matchImagePoints(ch_corners, ch_ids)
    else:
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids):
            corners, ids, _, _ = detector.refineDetectedMarkers(
                gray, board.board, corners, ids, rejected, K, dist)
        if ids is None or len(ids) == 0:
            rec["points"] = 0
            return rec, None, None
        keep = np.isin(ids.ravel(), board.ids)
        corners = [c for c, k in zip(corners, keep) if k]
        ids = ids[keep]
        rec["marker_ids"] = [int(i) for i in ids.ravel()]
        rec["marker_corners"] = [c.reshape(4, 2).tolist() for c in corners]
        if not len(ids):
            rec["points"] = 0
            return rec, None, None
        obj, img = board.board.matchImagePoints(corners, ids)
    if obj is None or len(obj) == 0:
        rec["points"] = 0
        return rec, None, None
    rec["points"] = int(len(obj))
    return rec, board.to_world(obj), img.reshape(-1, 2).astype(np.float64)


def _min_points(board: Board, ctx: Context) -> int:
    v = int(ctx.param("min_points", 0) or 0)
    if v > 0:
        return v
    return 6 if board.type == "charuco" else 8   # 6 corners / 2 markers


def _read_board(ctx: Context) -> Board:
    spec = io.read_json(ctx.input("board"))
    if not isinstance(spec, dict):
        raise NodeError("board json must be an object")
    return Board(spec)


def _read_camera(ctx: Context, width: int, height: int):
    cam = io.read_camera(ctx.input("camera"))
    K, dist = cam["K"].astype(np.float64), cam["dist"].astype(np.float64)
    cw, ch = int(cam["width"]), int(cam["height"])
    if (cw, ch) != (width, height):
        if abs(cw / ch - width / height) > 0.01:
            raise NodeError(
                f"camera is {cw}x{ch} but frames are {width}x{height} "
                "(different aspect ratio)",
                hint="pass intrinsics of the frame resolution")
        sx, sy = width / cw, height / ch
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        ctx.log(f"camera rescaled from {cw}x{ch} to {width}x{height}")
    return K, dist


def _gravity(orientation: str) -> Optional[List[float]]:
    return {"horizontal": [0.0, 0.0, -1.0],
            "vertical": [0.0, -1.0, 0.0]}.get(orientation)


def detect_board(ctx: Context) -> None:
    import cv2

    A = cv2.aruco
    board = _read_board(ctx)
    files = io.list_frames(ctx.input("frames"))
    if not files:
        raise NodeError("no frames")
    idx = _frame_indices(files)
    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape
    K, dist = _read_camera(ctx, w, h)
    params = _detector_params(ctx, board.type)
    detector = A.ArucoDetector(board.dictionary, params)
    charuco_detector = None
    if board.type == "charuco":
        cp = A.CharucoParameters()
        cp.cameraMatrix = K
        cp.distCoeffs = dist
        cp.tryRefineMarkers = True
        charuco_detector = A.CharucoDetector(board.board, cp, params)
    min_pts = _min_points(board, ctx)
    max_err = float(ctx.param("max_reproj_px", 2.0))
    overlay = bool(ctx.param("overlay", False))
    ov_dir = ctx.output_path("overlay") if overlay else None

    det_frames, Ts, used_idx, used_names, errs_mean, errs_max = (
        [], [], [], [], [], [])
    dists, npts = [], []
    for f, fi in zip(files, idx):
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            raise NodeError(f"cannot read frame {f}")
        if bgr.shape[:2] != (h, w):
            raise NodeError("all frames must have the same resolution")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rec, obj, img = _detect_frame(board, detector, charuco_detector,
                                      gray, K, dist)
        rec = {"frame": fi, "file": f.name, **rec, "used": False}
        pose = None
        if obj is None or len(obj) < min_pts:
            rec["reason"] = f"{rec['points']} points < min_points {min_pts}"
        else:
            pose = _pnp(obj.astype(np.float64), img, K, dist)
            if pose is None:
                rec["reason"] = "PnP failed"
            else:
                T_cw, err = pose
                rec["reproj_err_px"] = {"mean": float(err.mean()),
                                        "rms": float(np.sqrt((err ** 2)
                                                             .mean())),
                                        "max": float(err.max())}
                rec["T_cam_board"] = T_cw.tolist()
                if err.mean() > max_err:
                    rec["reason"] = (f"mean reprojection error "
                                     f"{err.mean():.2f}px > max_reproj_px "
                                     f"{max_err}")
                    pose = None
                else:
                    rec["used"] = True
                    T_wc = np.linalg.inv(T_cw)
                    Ts.append(T_wc)
                    used_idx.append(fi)
                    used_names.append(f.name)
                    errs_mean.append(float(err.mean()))
                    errs_max.append(float(err.max()))
                    dists.append(float(np.linalg.norm(T_wc[:3, 3])))
                    npts.append(int(len(obj)))
        det_frames.append(rec)
        if ov_dir is not None:
            vis = bgr.copy()
            if rec["marker_corners"]:
                A.drawDetectedMarkers(
                    vis, [np.asarray(c, np.float32).reshape(1, 4, 2)
                          for c in rec["marker_corners"]],
                    np.asarray(rec["marker_ids"], np.int32).reshape(-1, 1))
            if rec.get("charuco_corners"):
                A.drawDetectedCornersCharuco(
                    vis, np.asarray(rec["charuco_corners"],
                                    np.float32).reshape(-1, 1, 2),
                    np.asarray(rec["charuco_ids"], np.int32).reshape(-1, 1))
            if rec["used"]:
                T_cw = np.asarray(rec["T_cam_board"])
                rv, _ = cv2.Rodrigues(T_cw[:3, :3])
                cv2.drawFrameAxes(vis, K, dist, rv, T_cw[:3, 3],
                                  0.5 * min(board.width, board.height), 3)
            cv2.imwrite(str(ov_dir / io.frame_name(fi, ".jpg")), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not Ts:
        raise NodeError(
            f"board not detected (with >= {min_pts} points and reprojection "
            f"error <= {max_err}px) in any of {len(files)} frames",
            hint="check board.type/dictionary/rows/cols/first_id against the "
            "printed board (overlay=true shows what was detected); for "
            "ChArUco boards printed by OpenCV < 4.6 with an even row count "
            "set legacy_pattern=true")

    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {"T_world_cam": [T.tolist() for T in Ts],
                         "frame_index": used_idx, "frames": used_names,
                         "timestamps": [float(i) for i in used_idx]})
    ctx.set_output("trajectory", traj)
    dpath = ctx.output_path("detections", "detections.json")
    io.write_json(dpath, {"board": board.describe(),
                          "camera": {"K": K.tolist(),
                                     "dist": dist.tolist(),
                                     "width": w, "height": h},
                          "frames": det_frames})
    ctx.set_output("detections", dpath)

    orientation = ctx.param("board_orientation", "unknown")
    grav = _gravity(orientation)
    centres = np.array([T[:3, 3] for T in Ts])
    summary = {
        "frames_total": len(files),
        "frames_detected": sum(1 for r in det_frames if r["points"] > 0),
        "frames_used": len(Ts),
        "frames_rejected": [r["frame"] for r in det_frames
                            if r["points"] > 0 and not r["used"]],
        "mean_reproj_px": float(np.mean(errs_mean)),
        "max_reproj_px": float(np.max(errs_max)),
        "mean_points": float(np.mean(npts)),
        "camera_board_distance_m": {"min": float(np.min(dists)),
                                    "mean": float(np.mean(dists)),
                                    "max": float(np.max(dists))},
        "camera_path_length_m": float(np.linalg.norm(
            np.diff(centres, axis=0), axis=1).sum()) if len(Ts) > 1 else 0.0,
        "board": board.describe(),
        "world_frame": ("board frame: origin = board centre, x = columns "
                        "(right as printed), y = towards row 0 (up as "
                        "printed), z = normal out of the printed side"),
        "board_orientation": orientation,
        "gravity_world": grav,
        "up_world": None if grav is None else [-g for g in grav],
        "scale_anchor": {
            "source": "board spec lengths (metres)",
            "marker_length_m": board.marker_length,
            "square_length_m": getattr(board, "square_length", None),
            "note": "absolute scale is only as good as the printed size; "
                    "measure a printed square/marker and put the measured "
                    "value in the board spec",
        },
    }
    if grav is not None:
        g_cam = [(np.linalg.inv(T)[:3, :3] @ np.asarray(grav)).tolist()
                 for T in Ts[:1]]
        summary["gravity_cam_first_used_frame"] = g_cam[0]
    spath = ctx.output_path("summary", "summary.json")
    io.write_json(spath, summary)
    ctx.set_output("summary", spath)
    if ov_dir is not None:
        ctx.set_output("overlay", ov_dir)
    ctx.metadata.update({k: summary[k] for k in (
        "frames_total", "frames_used", "mean_reproj_px", "max_reproj_px")})


# ------------------------------------------------------------ scale_align
def _traj_indices(data: Dict[str, Any], n: int) -> List[int]:
    for key in ("frame_index", "frame_indices"):
        if isinstance(data.get(key), list) and len(data[key]) == n:
            return [int(i) for i in data[key]]
    names = data.get("frames")
    if isinstance(names, list) and len(names) == n:
        try:
            return [int(Path(str(s)).stem) for s in names]
        except ValueError:
            pass
    return list(range(n))


def _load_traj(path: Path, what: str):
    data = io.read_json(path)
    Ts = np.asarray(data.get("T_world_cam"), dtype=np.float64)
    if Ts.ndim != 3 or Ts.shape[1:] != (4, 4):
        raise NodeError(f"{what}: T_world_cam must be a list of 4x4")
    return data, Ts, _traj_indices(data, len(Ts))


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2,
                                              -1, 1))))


def _umeyama(X: np.ndarray, Y: np.ndarray):
    """Similarity s,R,t minimising |Y - (s R X + t)|^2 (Umeyama 1991)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    var_x = (Xc ** 2).sum(1).mean()
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / var_x)
    t = my - s * R @ mx
    return s, R, t


def _rotation_from_orientations(Rs_src: np.ndarray, Rs_ref: np.ndarray):
    """Chordal mean of R_ref_i R_src_i^T (rotation src world -> ref world)."""
    M = sum(Rr @ Rs.T for Rs, Rr in zip(Rs_src, Rs_ref))
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return U @ D @ Vt


def _scale_translation(X: np.ndarray, Y: np.ndarray, R: np.ndarray):
    """Given R, least squares s,t for Y = s R X + t."""
    mx, my = X.mean(0), Y.mean(0)
    RX = (R @ (X - mx).T).T
    s = float((RX * (Y - my)).sum() / (RX ** 2).sum())
    return s, my - s * R @ mx


def _transform_ply(src: Path, dst: Path, S: np.ndarray, s: float) -> int:
    from plyfile import PlyData

    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    names = v.dtype.names
    if not all(k in names for k in ("x", "y", "z")):
        raise NodeError("pointcloud ply has no x/y/z vertex properties")
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    xyz = (S[:3, :3] @ xyz.T).T + S[:3, 3]
    for i, k in enumerate("xyz"):
        v[k] = xyz[:, i].astype(v.dtype[k])
    if all(k in names for k in ("nx", "ny", "nz")):
        R = S[:3, :3] / s
        nrm = (R @ np.stack([v["nx"], v["ny"], v["nz"]], 1).T).T
        for i, k in enumerate(("nx", "ny", "nz")):
            v[k] = nrm[:, i].astype(v.dtype[k])
    ply["vertex"].data = v
    ply.write(str(dst))
    return len(v)


def scale_align(ctx: Context) -> None:
    src_data, T_src, i_src = _load_traj(ctx.input("trajectory"),
                                        "trajectory")
    _, T_ref, i_ref = _load_traj(ctx.input("reference"), "reference")
    ref_pos = {fi: k for k, fi in enumerate(i_ref)}
    pairs = [(k, ref_pos[fi]) for k, fi in enumerate(i_src) if fi in ref_pos]
    min_frames = max(2, int(ctx.param("min_frames", 3)))
    if len(pairs) < min_frames:
        raise NodeError(
            f"only {len(pairs)} frames shared between trajectory and "
            f"reference (need >= {min_frames})",
            hint="frames are matched by frame_index (or %06d names in "
            "'frames'); the board must be visible in enough frames that the "
            "source trajectory also covers")
    ks, kr = zip(*pairs)
    X = T_src[list(ks), :3, 3]
    Y = T_ref[list(kr), :3, 3]
    Rs_src, Rs_ref = T_src[list(ks), :3, :3], T_ref[list(kr), :3, :3]
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    spread = float(np.sqrt((sv ** 2).sum() / len(X)))
    if spread <= 1e-9:
        raise NodeError("source camera centres do not move; scale is "
                        "unobservable", hint="need camera translation "
                        "between the shared frames")
    planarity = [float(sv[1] / sv[0]), float(sv[2] / sv[0])]
    collinear = sv[1] / sv[0] < float(ctx.param("collinear_ratio", 0.05))
    method = ctx.param("method", "auto")
    if method == "umeyama" and (collinear or len(pairs) < 3):
        raise NodeError(
            "camera centres are (nearly) collinear: Umeyama rotation about "
            f"the motion line is undetermined (singular value ratio "
            f"{planarity[0]:.4f})",
            hint="use method=auto or method=orientation (rotation from "
            "camera orientations), or move the camera along a curve")
    if method == "orientation" or (method == "auto" and (
            collinear or len(pairs) < 3)):
        R = _rotation_from_orientations(Rs_src, Rs_ref)
        s, t = _scale_translation(X, Y, R)
        used = "orientation"
    else:
        s, R, t = _umeyama(X, Y)
        used = "umeyama"
    if not s > 0:
        raise NodeError(f"estimated scale {s:.4g} is not positive",
                        hint="trajectories do not correspond (check frame "
                        "indices / that both are camera-to-world)")
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    rot_err = [_rot_angle_deg((R @ Rsrc).T @ Rref)
               for Rsrc, Rref in zip(Rs_src, Rs_ref)]

    aligned = []
    for T in T_src:
        A = np.eye(4)
        A[:3, :3] = R @ T[:3, :3]
        A[:3, 3] = s * R @ T[:3, 3] + t
        aligned.append(A.tolist())
    out_traj = {k: v for k, v in src_data.items() if k != "T_world_cam"}
    out_traj["T_world_cam"] = aligned
    tpath = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(tpath, out_traj)
    ctx.set_output("trajectory", tpath)

    info = {
        "scale": s, "R": R.tolist(), "t": t.tolist(),
        "T_ref_src": S.tolist(),
        "convention": "p_ref = scale * R @ p_src + t (T_ref_src is the 4x4 "
                      "similarity with scale*R in its upper-left block); "
                      "aligned T_world_cam = [R @ R_src, scale * R @ t_src "
                      "+ t]; source depth/points in metres = scale * value",
        "method": used, "shared_frames": [int(i_src[k]) for k in ks],
        "n_shared": len(pairs),
        "rmse_m": float(np.sqrt((res ** 2).mean())),
        "median_err_m": float(np.median(res)),
        "max_err_m": float(res.max()),
        "rot_err_deg_mean": float(np.mean(rot_err)),
        "rot_err_deg_max": float(np.max(rot_err)),
        "src_spread": spread, "src_singular_ratios": planarity,
        "per_frame": [{"frame": int(i_src[k]), "err_m": float(e),
                       "rot_err_deg": float(r)}
                      for k, e, r in zip(ks, res, rot_err)],
    }
    apath = ctx.output_path("alignment", "alignment.json")
    io.write_json(apath, info)
    ctx.set_output("alignment", apath)

    if ctx.has_input("pointcloud"):
        ppath = ctx.output_path("pointcloud", "pointcloud.ply")
        n = _transform_ply(ctx.input("pointcloud"), ppath, S, s)
        ctx.log(f"transformed {n} points")
        ctx.set_output("pointcloud", ppath)
    if ctx.has_input("depth"):
        ddir = ctx.output_path("depth")
        for f in io.list_frames(ctx.input("depth"), (".npy",)):
            io.write_depth(ddir / f.name, np.load(f).astype(np.float32) * s)
        ctx.set_output("depth", ddir)
    ctx.metadata.update({"scale": s, "rmse_m": info["rmse_m"],
                         "n_shared": len(pairs), "method": used})


if __name__ == "__main__":
    raise SystemExit(main({"detect_board": detect_board,
                           "scale_align": scale_align}))
