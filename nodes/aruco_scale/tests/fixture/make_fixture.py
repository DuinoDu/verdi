"""Render the synthetic aruco_scale fixtures with known camera poses.

Run with the node venv from nodes/aruco_scale:
    python tests/fixture/make_fixture.py
Writes tests/fixture/{camera.json, charuco_board.json, charuco/,
apriltag_board.json, apriltag/, src_trajectory.json, src_points.ply} and
tests/expected/{charuco_gt_*.json, apriltag_gt.json}.

The board lies in the world plane z = 0 (world = node board frame: origin
board centre, x right, y up as printed, z out of the board). Every pixel is
ray-cast through the distorted camera model (4x supersampling) onto that
plane, so the images follow the exact OpenCV camera model incl. distortion.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "expected"
W, H = 640, 480
K = np.array([[520.0, 0, 319.5], [0, 520.0, 239.5], [0, 0, 1]])
DIST = np.array([-0.08, 0.03, 0.0005, -0.0003, 0.0])
SS = 4            # supersampling
PPM = 4000        # board texture pixels per metre
MARGIN = 0.015    # white paper margin around the pattern (m)
A = cv2.aruco


def look_at(eye, target, up_hint):
    """T_world_cam (OpenCV axes) for a camera at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up_hint)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def roll(T, deg):
    a = np.radians(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                   [0, 0, 1]])
    T = T.copy()
    T[:3, :3] = T[:3, :3] @ Rz
    return T


def board_texture(board, bw, bh):
    """Pattern image incl. white margin; returns (img, origin_px)."""
    pw, ph = int(round(bw * PPM)), int(round(bh * PPM))
    pat = board.generateImage((pw, ph), marginSize=0, borderBits=1)
    m = int(round(MARGIN * PPM))
    img = cv2.copyMakeBorder(pat, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    img = (img.astype(np.float32) / 255.0) * (235 - 25) + 25
    return img, m


def background(seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(256, 256)).astype(np.float32)
    tex = cv2.GaussianBlur(noise, (0, 0), 6)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    stripes = 0.5 + 0.5 * np.sin(np.linspace(0, 40, 256))[None, :]
    tex = 0.6 * tex + 0.4 * stripes.astype(np.float32)
    return (90 + 90 * tex).astype(np.float32)  # covers [-1.5, 1.5]^2 m


def render(T_wc, tex, origin_px, bw, bh, bg, seed):
    hw, hh = W * SS, H * SS
    Kh = K.copy()
    Kh[0, 0] *= SS
    Kh[1, 1] *= SS
    Kh[0, 2] = (K[0, 2] + 0.5) * SS - 0.5
    Kh[1, 2] = (K[1, 2] + 0.5) * SS - 0.5
    u, v = np.meshgrid(np.arange(hw, dtype=np.float32),
                       np.arange(hh, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    nrm = cv2.undistortPointsIter(pts, Kh, DIST, None, None,
                                  (cv2.TERM_CRITERIA_COUNT
                                   | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    d = np.concatenate([nrm.reshape(-1, 2), np.ones((hw * hh, 1))], 1)
    dw = d @ T_wc[:3, :3].T
    C = T_wc[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -C[2] / dw[:, 2]
    hit = lam > 0
    P = C[None, :] + lam[:, None] * dw
    xw, yw = P[:, 0], P[:, 1]
    # board texture coordinates (OpenCV board frame: y down)
    tu = (xw + bw / 2) * PPM - 0.5 + origin_px
    tv = (bh / 2 - yw) * PPM - 0.5 + origin_px
    th, tw = tex.shape
    on_paper = hit & (tu >= -0.5) & (tv >= -0.5) & (tu <= tw - 0.5) & (
        tv <= th - 0.5)
    img = np.full(hw * hh, 140.0, np.float32)
    paper = cv2.remap(tex, tu.reshape(hh, hw).astype(np.float32),
                      tv.reshape(hh, hw).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gb = bg.shape[0]
    bu = ((xw + 1.5) / 3.0 * gb - 0.5).reshape(hh, hw).astype(np.float32)
    bv = ((yw + 1.5) / 3.0 * gb - 0.5).reshape(hh, hw).astype(np.float32)
    back = cv2.remap(bg, bu, bv, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
    img[hit] = back.ravel()[hit]
    img[on_paper] = paper.ravel()[on_paper]
    img = img.reshape(hh, hw)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    # mild vignetting + colour tint + sensor noise
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    img = img * vig
    rng = np.random.default_rng(seed)
    rgb = np.stack([img * 1.02, img, img * 0.95], 2)
    rgb += rng.normal(0, 2.0, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def write_traj(path, Ts, idx):
    path.write_text(json.dumps({
        "T_world_cam": [np.asarray(T).tolist() for T in Ts],
        "frame_index": list(idx), "frames": [f"{i:06d}.jpg" for i in idx],
        "timestamps": [float(i) for i in idx]}, indent=1))


def make_seq(name, board, bw, bh, spec, poses, visible, seed):
    tex, org = board_texture(board, bw, bh)
    bg = background(seed)
    out = HERE / name
    out.mkdir(exist_ok=True)
    for i, T in enumerate(poses):
        rgb = render(T, tex, org, bw, bh, bg, seed + i)
        cv2.imwrite(str(out / f"{i:06d}.jpg"), rgb[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    (HERE / f"{name}_board.json").write_text(json.dumps(spec, indent=1))
    vis = [i for i in range(len(poses)) if visible[i]]
    write_traj(EXP / f"{name}_gt.json", [poses[i] for i in vis], vis)
    write_traj(EXP / f"{name}_gt_all.json", poses, range(len(poses)))
    return vis


def main():
    EXP.mkdir(exist_ok=True)
    (HERE / "camera.json").write_text(json.dumps({
        "K": K.tolist(), "width": W, "height": H, "dist": DIST.tolist()}))

    # ChArUco 7x5 squares, 40 mm squares, 30 mm markers, lying on a table
    d = A.getPredefinedDictionary(A.DICT_5X5_100)
    cb = A.CharucoBoard((7, 5), 0.04, 0.03, d)
    spec = {"type": "charuco", "dictionary": "DICT_5X5_100", "rows": 5,
            "cols": 7, "square_length_m": 0.04, "marker_length_m": 0.03,
            "first_id": 0}
    poses, visible = [], []
    n = 10
    for i in range(n):
        az = np.radians(-50 + 100 * i / (n - 1))
        el = np.radians(55 + 12 * np.sin(i))
        r = 0.45 + 0.2 * i / (n - 1)
        eye = r * np.array([np.sin(az) * np.cos(el),
                            -np.cos(az) * np.cos(el), np.sin(el)])
        target = np.array([0.01 * np.cos(i), 0.01 * np.sin(i), 0.0])
        if i == 6:  # camera looks away: board not visible in this frame
            target = eye + np.array([1.0, 0.6, 0.2])
        T = roll(look_at(eye, target, np.array([0, 0, 1.0])),
                 8 * np.sin(1.3 * i))
        poses.append(T)
        visible.append(i != 6)
    make_seq("charuco", cb, 0.28, 0.20, spec, poses, visible, 100)

    # AprilTag 36h11 grid 4x3 (Kalibr-like spacing 0.3), ids 10..21, on a
    # wall: board normal horizontal, row 0 at the top
    d = A.getPredefinedDictionary(A.DICT_APRILTAG_36h11)
    ids = np.arange(10, 22, dtype=np.int32)
    gb = A.GridBoard((4, 3), 0.05, 0.015, d, ids)
    spec = {"type": "apriltag_grid", "dictionary": "DICT_APRILTAG_36h11",
            "rows": 3, "cols": 4, "marker_length_m": 0.05,
            "separation_m": 0.015, "first_id": 10}
    bw, bh = 4 * 0.05 + 3 * 0.015, 3 * 0.05 + 2 * 0.015
    poses = []
    for i in range(5):
        eye = np.array([-0.25 + 0.12 * i, -0.08 + 0.04 * i,
                        0.55 + 0.05 * i])
        T = roll(look_at(eye, np.array([0.0, 0.0, 0.0]),
                         np.array([0, 1.0, 0])), -5 + 2.5 * i)
        poses.append(T)
    make_seq("apriltag", gb, bw, bh, spec, poses, [True] * 5, 200)

    # "up-to-scale" source trajectory (vggt-like) for scale_align: world =
    # first camera, unit = metres / 0.4, small pose noise; all 10 frames
    gt = json.loads((EXP / "charuco_gt_all.json").read_text())
    Tgt = np.asarray(gt["T_world_cam"])
    T0inv = np.linalg.inv(Tgt[0])
    rng = np.random.default_rng(7)
    src = []
    for T in Tgt:
        S = T0inv @ T
        S[:3, 3] = S[:3, 3] / 0.4 + rng.normal(0, 0.001 / 0.4, 3)
        a = rng.normal(0, np.radians(0.2), 3)
        S[:3, :3] = cv2.Rodrigues(a)[0] @ S[:3, :3]
        src.append(S)
    write_traj(HERE / "src_trajectory.json", src, range(len(src)))
    # 5 points (board corners + centre) in the source frame
    pw = np.array([[-0.14, -0.10, 0], [0.14, -0.10, 0], [0.14, 0.10, 0],
                   [-0.14, 0.10, 0], [0, 0, 0]])
    ps = ((T0inv[:3, :3] @ pw.T).T + T0inv[:3, 3]) / 0.4
    lines = ["ply", "format ascii 1.0", f"element vertex {len(ps)}",
             "property float x", "property float y", "property float z",
             "end_header"] + [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in ps]
    (HERE / "src_points.ply").write_text("\n".join(lines) + "\n")
    for nm in ("charuco_gt", "apriltag_gt"):
        g = json.loads((EXP / f"{nm}.json").read_text())
        for k in (0, -1):
            print(nm, g["frame_index"][k],
                  np.round(np.asarray(g["T_world_cam"][k])[:3, 3], 4))
    print("src frame 6 gt centre", np.round(Tgt[6][:3, 3], 4))


if __name__ == "__main__":
    main()
