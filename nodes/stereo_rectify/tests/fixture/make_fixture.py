"""Synthetic fisheye stereo rig with analytic ground truth.

Renders (ray casting, 3x supersampling) a textured box room with a table
seen by a horizontal fisheye stereo rig (OpenCV cv2.fisheye / Kannala-Brandt
model, baseline 60.5 mm like the real2sim head camera, small relative
rotation between the eyes), as side-by-side frames:

  fixture/sbs/%06d.png        2 SBS frames (left | right), each eye 640x360
  fixture/calib.json          calibration (X_right = R X_left + T, metres)
  expected/rect_left/, expected/rect_right/   the SAME scene rendered
                              directly by the ideal rectified pinhole
                              cameras (cv2.fisheye.stereoRectify, balance 0)
  expected/rect_camera.json   rectified left K
  expected/rect_depth/%06d.npy  analytic z-depth of the rectified left image
                              (metres; 0 where the point is not seen by the
                              rectified right image)

Used by stereo_rectify (remap correctness vs direct rendering) and by
foundation_stereo (metric depth vs analytic truth). Run with any python
that has numpy + opencv + pillow:  python make_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W, H = 640, 360
SS = 3                      # supersampling factor
K1 = np.array([[205.0, 0, 321.3], [0, 205.6, 178.9], [0, 0, 1]])
D1 = np.array([0.031, -0.012, 0.004, -0.0011])
K2 = np.array([[204.2, 0, 318.4], [0, 204.9, 181.2], [0, 0, 1]])
D2 = np.array([0.028, -0.009, 0.002, -0.0007])
BASELINE = 0.0605


def rotvec(v):
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64))[0]


# right-from-left extrinsics: right camera centre at +x of the left camera
R_rl = rotvec([0.004, -0.009, 0.003])
C_right_in_left = np.array([BASELINE, 0.0008, -0.0005])
C_right_in_left *= BASELINE / np.linalg.norm(C_right_in_left)
T_rl = -R_rl @ C_right_in_left

# ------------------------------------------------------------------ scene
rng = np.random.default_rng(7)


def make_texture(n=1024):
    tex = np.zeros((n, n, 3), np.float32)
    for octave, amp in ((8, 0.35), (32, 0.25), (128, 0.25), (512, 0.15)):
        small = rng.random((octave, octave, 3)).astype(np.float32)
        tex += amp * cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return tex


TEX = [make_texture() for _ in range(3)]
PX_PER_M = 260.0

# room: x in [-1.4, 1.4], y in [-1.2, 0.65] (floor y = 0.65), z in [-1.0, 2.2]
ROOM_MIN = np.array([-1.4, -1.2, -1.0])
ROOM_MAX = np.array([1.4, 0.65, 2.2])
# table block
TAB_MIN = np.array([-0.45, 0.22, 0.75])
TAB_MAX = np.array([0.40, 0.65, 1.30])


def sample_tex(tid, u, v):
    """Bilinear, wrapped texture lookup at surface coords (metres)."""
    tex = TEX[tid]
    n = tex.shape[0]
    x = (u * PX_PER_M) % n
    y = (v * PX_PER_M) % n
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    ax = (x - x0)[:, None]
    ay = (y - y0)[:, None]
    x0 %= n
    y0 %= n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    return ((1 - ax) * (1 - ay) * tex[y0, x0] + ax * (1 - ay) * tex[y0, x1]
            + (1 - ax) * ay * tex[y1, x0] + ax * ay * tex[y1, x1])


def cast(o, d):
    """o (3,), d (N,3) unit rays in WORLD (= left cam of frame 0) frame.
    Returns t (N,), rgb (N,3)."""
    N = d.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
    # room from inside: exit distance of slab
    t1 = (ROOM_MIN - o) * inv
    t2 = (ROOM_MAX - o) * inv
    tfar = np.maximum(t1, t2)
    t_room = np.nanmin(np.where(np.isfinite(tfar), tfar, np.inf), axis=1)
    axis_room = np.nanargmin(np.where(np.isfinite(tfar), tfar, np.inf), axis=1)
    # table block from outside
    s1 = (TAB_MIN - o) * inv
    s2 = (TAB_MAX - o) * inv
    tnear = np.nanmax(np.minimum(s1, s2), axis=1)
    tex_ = np.nanmin(np.maximum(s1, s2), axis=1)
    hit_tab = (tnear <= tex_) & (tnear > 1e-6) & (tnear < t_room)
    axis_tab = np.nanargmax(np.minimum(s1, s2), axis=1)
    t = np.where(hit_tab, tnear, t_room)
    axis = np.where(hit_tab, axis_tab, axis_room)
    P = o[None] + t[:, None] * d
    # in-plane coordinates per hit axis
    u = np.where(axis == 0, P[:, 2], P[:, 0])
    v = np.where(axis == 1, P[:, 2], P[:, 1])
    rgb = np.zeros((N, 3), np.float32)
    for tid, sel in ((0, ~hit_tab & (axis != 1)), (1, ~hit_tab & (axis == 1)),
                     (2, hit_tab)):
        if sel.any():
            rgb[sel] = sample_tex(tid, u[sel], v[sel])
    # simple shading per face orientation to give edges contrast
    shade = np.array([0.85, 1.0, 0.7])[axis]
    shade = np.where(hit_tab, shade * 0.9, shade)
    return t, rgb * shade[:, None]


# ----------------------------------------------------------------- camera
def fisheye_rays(K, D, w, h, ss):
    us = (np.arange(w * ss) + 0.5) / ss - 0.5
    vs = (np.arange(h * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(us, vs)
    # K has no skew
    xd = (uu - K[0, 2]) / K[0, 0]
    yd = (vv - K[1, 2]) / K[1, 1]
    td = np.sqrt(xd ** 2 + yd ** 2)
    th = td.copy()
    k1, k2, k3, k4 = D
    for _ in range(30):  # Newton: theta*(1+k1 t^2+...) = theta_d
        t2 = th * th
        f = th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4) - td
        df = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
        th = th - f / df
    s = np.where(td > 1e-12, np.sin(th) / np.maximum(td, 1e-12), 1.0)
    d = np.stack([xd * s, yd * s, np.cos(th)], -1)
    return d.reshape(-1, 3)


def pinhole_rays(K, w, h, ss):
    us = (np.arange(w * ss) + 0.5) / ss - 0.5
    vs = (np.arange(h * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(us, vs)
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                  np.ones_like(uu)], -1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def render(dirs_cam, R_world_cam, C_world, w, h, ss):
    d = dirs_cam @ R_world_cam.T
    t, rgb = cast(C_world, d)
    img = rgb.reshape(h * ss, w * ss, 3)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), t, d


# rig poses in the world (= left camera of frame 0): T_world_left
RIG = [np.eye(4)]
T = np.eye(4)
T[:3, :3] = rotvec([0.0, 0.035, 0.0])
T[:3, 3] = [0.06, -0.01, 0.04]
RIG.append(T)


def main():
    import shutil

    for d in ("sbs",):
        shutil.rmtree(HERE / d, ignore_errors=True)
        (HERE / d).mkdir(parents=True)
    exp = HERE.parent / "expected"
    for d in ("rect_left", "rect_right", "rect_depth"):
        shutil.rmtree(exp / d, ignore_errors=True)
        (exp / d).mkdir(parents=True)

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, (W, H), R_rl, T_rl.reshape(3, 1),
        cv2.CALIB_ZERO_DISPARITY, newImageSize=(W, H), balance=0.0,
        fov_scale=1.0)
    Kr = P1[:3, :3]
    B = -P2[0, 3] / P2[0, 0]
    print("rectified K", Kr.tolist(), "baseline", B)

    fl = fisheye_rays(K1, D1, W, H, SS)
    fr = fisheye_rays(K2, D2, W, H, SS)
    pr = pinhole_rays(Kr, W, H, SS)
    pr1 = pinhole_rays(Kr, W, H, 1)
    R_left_right = R_rl.T            # X_left = R_rl^T (X_right - T)
    C_right = -R_rl.T @ T_rl          # right centre in left frame
    for i, Twl in enumerate(RIG):
        Rw, Cw = Twl[:3, :3], Twl[:3, 3]
        L, _, _ = render(fl, Rw, Cw, W, H, SS)
        Rr = Rw @ R_left_right
        Cr = Cw + Rw @ C_right
        Rimg, _, _ = render(fr, Rr, Cr, W, H, SS)
        Image.fromarray(np.concatenate([L, Rimg], 1)).save(HERE / "sbs" / f"{i:06d}.png")
        # ideal rectified pinhole renders
        R_world_rectL = Rw @ R1.T
        R_world_rectR = Rr @ R2.T
        rl, _, _ = render(pr, R_world_rectL, Cw, W, H, SS)
        rr, _, _ = render(pr, R_world_rectR, Cr, W, H, SS)
        Image.fromarray(rl).save(exp / "rect_left" / f"{i:06d}.png")
        Image.fromarray(rr).save(exp / "rect_right" / f"{i:06d}.png")
        # analytic z-depth of the rectified left image (pixel centres)
        t, _ = cast(Cw, pr1 @ R_world_rectL.T)
        z = (t * pr1[:, 2]).reshape(H, W)
        # visible in the rectified right image? x_r = u - fx B / z  (+ occlusion)
        uu = np.arange(W)[None].repeat(H, 0)
        xr = uu - Kr[0, 0] * B / z
        P_world = Cw[None] + t[:, None] * (pr1 @ R_world_rectL.T)
        v_r = P_world - Cr[None]
        dist_r = np.linalg.norm(v_r, axis=1)
        tr, _ = cast(Cr, v_r / dist_r[:, None])
        occluded = (tr < dist_r - 1e-3).reshape(H, W)
        zz = np.where((xr >= 0) & ~occluded, z, 0).astype(np.float32)
        np.save(exp / "rect_depth" / f"{i:06d}.npy", zz)
        print(i, "depth range", float(z.min()), float(z.max()),
              "gt valid", float((zz > 0).mean()))
    calib = {
        "model": "fisheye", "width": W, "height": H, "units": "m",
        "left": {"K": K1.tolist(), "D": D1.tolist()},
        "right": {"K": K2.tolist(), "D": D2.tolist()},
        "R": R_rl.tolist(), "T": T_rl.tolist(),
        "note": "synthetic rig; X_right = R @ X_left + T",
    }
    (HERE / "calib.json").write_text(json.dumps(calib, indent=2))
    # negative test: wrong relative rotation (1.7 deg about x) -> large |dy|
    bad = dict(calib, R=(rotvec([0.03, 0.0, 0.0]) @ R_rl).tolist(),
               note="WRONG on purpose: R perturbed by 1.7 deg about x")
    (HERE / "calib_swapped_rotation.json").write_text(json.dumps(bad, indent=2))
    (exp / "rect_camera.json").write_text(json.dumps(
        {"K": Kr.tolist(), "width": W, "height": H, "dist": [0, 0, 0, 0, 0]},
        indent=2))
    (exp / "rect_baseline.txt").write_text(f"{B:.6f}\n")


if __name__ == "__main__":
    main()
