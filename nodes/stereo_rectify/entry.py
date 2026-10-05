"""stereo_rectify node: calibrated (fisheye or pinhole) stereo -> rectified
pinhole pair for stereo matchers (FoundationStereo).

Deterministic OpenCV algorithm node (CPU, no weights). Conventions:

* calibration extrinsics follow cv2.stereoCalibrate: X_right = R @ X_left + T
  (metres), i.e. ``T_right_left``;
* rectified frames: R1 rotates LEFT camera coordinates into the rectified
  left frame (X_rect = R1 @ X_left); depth produced on the rectified left
  image is z in that frame; ``T_left_rect`` (= [R1^T | 0]) maps it back to
  the original left camera (OpenCV axes);
* output K is the rectified left pinhole (dist = 0) at the output size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


# ------------------------------------------------------------------ calib
def _mat(v, shape, where):
    a = np.asarray(v, dtype=np.float64)
    if a.shape != shape:
        raise NodeError(f"calib {where} must have shape {shape}, got {a.shape}")
    return a


def read_calib(path: Path) -> Dict:
    try:
        c = json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        raise NodeError(f"cannot read calib json {path}: {exc}") from exc
    model = c.get("model")
    if model not in ("fisheye", "pinhole"):
        raise NodeError("calib.model must be 'fisheye' (OpenCV cv2.fisheye, "
                        "Kannala-Brandt k1..k4) or 'pinhole' (OpenCV "
                        "k1,k2,p1,p2[,k3..])", hint="see `verdi describe stereo_rectify`")
    try:
        W, H = int(c["width"]), int(c["height"])
        K1 = _mat(c["left"]["K"], (3, 3), "left.K")
        K2 = _mat(c["right"]["K"], (3, 3), "right.K")
        D1 = np.asarray(c["left"].get("D", []), dtype=np.float64).ravel()
        D2 = np.asarray(c["right"].get("D", []), dtype=np.float64).ravel()
    except KeyError as exc:
        raise NodeError(f"calib is missing {exc}") from None
    if "T_right_left" in c:
        Trl = _mat(c["T_right_left"], (4, 4), "T_right_left")
        R, T = Trl[:3, :3], Trl[:3, 3]
    else:
        if "R" not in c or "T" not in c:
            raise NodeError("calib needs R (3x3) + T (3) or T_right_left (4x4)")
        R = _mat(c["R"], (3, 3), "R")
        T = np.asarray(c["T"], dtype=np.float64).ravel()
        if T.shape != (3,):
            raise NodeError("calib T must have 3 values (metres)")
    if abs(np.linalg.det(R) - 1) > 1e-3 or np.abs(R @ R.T - np.eye(3)).max() > 1e-3:
        raise NodeError("calib R is not a rotation matrix")
    if model == "fisheye":
        for name, D in (("left", D1), ("right", D2)):
            if D.size != 4:
                raise NodeError(f"fisheye calib {name}.D must have 4 values "
                                f"(k1..k4), got {D.size}")
    units = c.get("units", "m")
    if units == "mm":
        T = T / 1000.0
    elif units != "m":
        raise NodeError("calib.units must be 'm' or 'mm'")
    baseline = float(np.linalg.norm(T))
    if not 0.005 < baseline < 2.0:
        raise NodeError(f"baseline |T| = {baseline:.4f} m looks wrong",
                        hint="T is metres (or set units='mm'); X_right = R X_left + T")
    if T[0] > 0:
        raise NodeError(
            "T[0] > 0: the 'right' camera is to the LEFT of the 'left' camera "
            "(X_right = R X_left + T means T ~ [-baseline, 0, 0] for a "
            "standard rig)", hint="swap left/right or check the T convention")
    return {"model": model, "W": W, "H": H, "K1": K1, "D1": D1, "K2": K2,
            "D2": D2, "R": R, "T": T, "baseline": baseline, "raw": c}


def rectify_maps(cal: Dict, out_size: Tuple[int, int], balance: float,
                 fov_scale: float, alpha: float):
    import cv2

    size = (cal["W"], cal["H"])
    flags = cv2.CALIB_ZERO_DISPARITY
    if cal["model"] == "fisheye":
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            cal["K1"], cal["D1"], cal["K2"], cal["D2"], size, cal["R"],
            cal["T"].reshape(3, 1), flags, newImageSize=out_size,
            balance=balance, fov_scale=fov_scale)
        m1 = cv2.fisheye.initUndistortRectifyMap(
            cal["K1"], cal["D1"], R1, P1, out_size, cv2.CV_32FC1)
        m2 = cv2.fisheye.initUndistortRectifyMap(
            cal["K2"], cal["D2"], R2, P2, out_size, cv2.CV_32FC1)
    else:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cal["K1"], cal["D1"], cal["K2"], cal["D2"], size, cal["R"],
            cal["T"].reshape(3, 1), flags=flags, alpha=alpha,
            newImageSize=out_size)
        m1 = cv2.initUndistortRectifyMap(
            cal["K1"], cal["D1"], R1, P1, out_size, cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(
            cal["K2"], cal["D2"], R2, P2, out_size, cv2.CV_32FC1)
    return R1, R2, P1, P2, Q, m1, m2


def _valid(mx, my, W, H):
    return (mx >= 0) & (my >= 0) & (mx <= W - 1) & (my <= H - 1)


# ----------------------------------------------------------------- frames
def _frame_pairs(ctx: Context, W: int, H: int) -> List[Tuple[str, str, Path]]:
    """[(name, kind, path)], kind in sbs | pair; loads lazily."""
    if ctx.has_input("stereo"):
        if ctx.has_input("left") or ctx.has_input("right"):
            raise NodeError("pass EITHER stereo (side-by-side) OR left + right")
        files = io.list_frames(ctx.input("stereo"))
        return [("sbs", f, None) for f in files]
    if not (ctx.has_input("left") and ctx.has_input("right")):
        raise NodeError("need input stereo (SBS frames/video) or left + right",
                        hint="-i stereo=sbs.mp4 or -i left=dirL -i right=dirR")
    fl = io.list_frames(ctx.input("left"))
    fr = io.list_frames(ctx.input("right"))
    if len(fl) != len(fr):
        raise NodeError(f"left has {len(fl)} frames, right has {len(fr)}")
    return [("pair", a, b) for a, b in zip(fl, fr)]


def _load_pair(kind, a, b, W, H, left_first: bool):
    if kind == "sbs":
        img = io.read_image(a)
        if img.shape[1] != 2 * W or img.shape[0] != H:
            raise NodeError(f"SBS frame {a.name} is {img.shape[1]}x{img.shape[0]}, "
                            f"expected {2 * W}x{H} for a {W}x{H} calibration")
        L, R = img[:, :W], img[:, W:]
        return (L, R) if left_first else (R, L)
    L, R = io.read_image(a), io.read_image(b)
    for name, im in (("left", L), ("right", R)):
        if im.shape[:2] != (H, W):
            raise NodeError(f"{name} frame {a.name} is {im.shape[1]}x{im.shape[0]}, "
                            f"calibration is {W}x{H}")
    return L, R


def epipolar_check(L: np.ndarray, R: np.ndarray, max_feat: int = 4000) -> Dict:
    """SIFT matches on a rectified pair -> vertical residual |y_l - y_r|.

    A correct rectification has |dy| ~ 0 (< 0.5 px) and dx = x_l - x_r >= 0.
    """
    import cv2

    gl = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_feat)
    k1, d1 = sift.detectAndCompute(gl, None)
    k2, d2 = sift.detectAndCompute(gr, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return {"matches": 0, "note": "too few features"}
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < 0.7 * n.distance]
    if len(good) < 8:
        return {"matches": len(good), "note": "too few matches"}
    pl = np.array([k1[m.queryIdx].pt for m in good])
    pr = np.array([k2[m.trainIdx].pt for m in good])
    dy = pl[:, 1] - pr[:, 1]
    dx = pl[:, 0] - pr[:, 0]
    # matches far off the epipolar line are mismatches, not rectification error
    inl = np.abs(dy - np.median(dy)) < 3.0
    dyi = dy[inl]
    return {"matches": int(len(good)), "inliers": int(inl.sum()),
            "median_dy_px": float(np.median(dyi)),
            "median_abs_dy_px": float(np.median(np.abs(dyi))),
            "p90_abs_dy_px": float(np.percentile(np.abs(dyi), 90)),
            "negative_dx_ratio": float((dx[inl] < -0.5).mean())}


# ------------------------------------------------------------------- task
def rectify(ctx: Context) -> None:
    import cv2

    cal = read_calib(ctx.input("calib"))
    W, H = cal["W"], cal["H"]
    ow = int(ctx.param("out_width", 0)) or W
    oh = int(ctx.param("out_height", 0)) or H
    balance = float(ctx.param("balance", 0.0))
    fov_scale = float(ctx.param("fov_scale", 1.0))
    alpha = float(ctx.param("alpha", 0.0))
    if not 0 <= balance <= 1:
        raise NodeError("balance must be in [0, 1]")
    R1, R2, P1, P2, Q, m1, m2 = rectify_maps(cal, (ow, oh), balance,
                                            fov_scale, alpha)
    K = P1[:3, :3].copy()
    if np.abs(P1[:3, :3] - P2[:3, :3]).max() > 1e-6:
        raise NodeError("rectified K of left and right differ (unexpected)")
    # P2[0,3] = -fx * B  (CALIB_ZERO_DISPARITY, horizontal rig)
    baseline_rect = float(-P2[0, 3] / P2[0, 0])
    if abs(P2[1, 3]) > 1e-6 * abs(P2[0, 3]):
        raise NodeError("rectification is vertical (cameras stacked); "
                        "only horizontal rigs are supported")
    doffs = float(P2[0, 2] - P1[0, 2])

    pairs = _frame_pairs(ctx, W, H)
    if not pairs:
        raise NodeError("no frames found")
    left_first = bool(ctx.param("sbs_left_first", True))
    interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
              "lanczos": cv2.INTER_LANCZOS4}[str(ctx.param("interpolation", "linear"))]
    out_l = ctx.output_path("left")
    out_r = ctx.output_path("right")
    frames = []
    first = None
    for i, (kind, a, b) in enumerate(pairs):
        L, R = _load_pair(kind, a, b, W, H, left_first)
        rl = cv2.remap(L, m1[0], m1[1], interp, borderMode=cv2.BORDER_CONSTANT)
        rr = cv2.remap(R, m2[0], m2[1], interp, borderMode=cv2.BORDER_CONSTANT)
        io.write_image(out_l / io.frame_name(i), rl)
        io.write_image(out_r / io.frame_name(i), rr)
        frames.append({"index": i, "source": str(a) if b is None else
                       [str(a), str(b)]})
        if first is None:
            first = (rl, rr)
    ctx.set_output("left", out_l)
    ctx.set_output("right", out_r)

    cam = ctx.output_path("camera", "camera.json")
    io.write_camera(cam, K, ow, oh)
    ctx.set_output("camera", cam)

    valid_l = _valid(m1[0], m1[1], W, H)
    valid_r = _valid(m2[0], m2[1], W, H)
    vpath = ctx.output_path("valid", "valid_left.png")
    io.write_mask(vpath, valid_l.astype(np.uint16))
    ctx.set_output("valid", vpath)

    check = epipolar_check(*first) if ctx.param("epipolar_check", True) else None
    T_left_rect = np.eye(4)
    T_left_rect[:3, :3] = R1.T
    T_right_rect = np.eye(4)
    T_right_rect[:3, :3] = R2.T
    rect = {
        "convention": {
            "calib_extrinsics": "X_right = R @ X_left + T (cv2.stereoCalibrate), metres",
            "rectified_left": "X_rect = R1 @ X_left; depth on the rectified left image is z in this frame",
            "T_left_rect": "4x4 rectified-left frame -> original left camera frame (OpenCV axes)",
            "disparity": "d = x_left - x_right (px, rectified images); z = fx * baseline_m / (d + doffs)",
        },
        "model": cal["model"],
        "input_size": [W, H], "output_size": [ow, oh],
        "K_left_input": cal["K1"].tolist(), "D_left_input": cal["D1"].tolist(),
        "K_right_input": cal["K2"].tolist(), "D_right_input": cal["D2"].tolist(),
        "R": cal["R"].tolist(), "T": cal["T"].tolist(),
        "baseline_m": cal["baseline"], "baseline_rect_m": baseline_rect,
        "K_rect": K.tolist(), "doffs_px": doffs,
        "R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(),
        "P2": P2.tolist(), "Q": Q.tolist(),
        "T_left_rect": T_left_rect.tolist(), "T_right_rect": T_right_rect.tolist(),
        "params": {"balance": balance, "fov_scale": fov_scale, "alpha": alpha,
                   "sbs_left_first": left_first},
        "valid_ratio_left": float(valid_l.mean()),
        "valid_ratio_right": float(valid_r.mean()),
        "epipolar_check_frame0": check,
        "frames": frames,
    }
    rpath = ctx.output_path("rectification", "rectification.json")
    io.write_json(rpath, rect)
    ctx.set_output("rectification", rpath)
    ctx.metadata.update({
        "frames": len(frames), "baseline_m": cal["baseline"],
        "fx_rect": float(K[0, 0]), "doffs_px": doffs,
        "epipolar_median_abs_dy_px": None if not check else check.get("median_abs_dy_px")})
    if check and check.get("median_abs_dy_px") is not None:
        lim = float(ctx.param("max_epipolar_dy_px", 1.0))
        if lim > 0 and check["median_abs_dy_px"] > lim:
            raise NodeError(
                f"rectified pair has median |dy| = {check['median_abs_dy_px']:.2f} px "
                f"> {lim} px on frame 0: calibration does not match the images",
                hint="check left/right order, the R/T convention and the "
                     "intrinsics; set max_epipolar_dy_px=0 to skip this check")


if __name__ == "__main__":
    raise SystemExit(main({"rectify": rectify}))
