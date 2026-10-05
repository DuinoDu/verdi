"""MASt3R-SLAM node: dense monocular SLAM (single process, no GUI).

The loop below is upstream ``main.py`` + ``run_backend`` + ``relocalization``
run synchronously in one process (upstream's ``single_thread`` semantics):
tracking of every frame against the last keyframe, a backend step (graph
construction with retrieval loop-closure edges + global Gauss-Newton) after
every new keyframe, relocalisation when tracking is lost.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

# checkpoints store argparse Namespaces (torch>=2.6 defaults to weights_only)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

IMG_SIZE = 512


def _setup_paths(repo: Path) -> None:
    for p in (repo / "thirdparty" / "lietorch",
              repo / "thirdparty" / "mast3r" / "asmk",
              repo / "thirdparty" / "mast3r",
              repo):
        sys.path.insert(0, str(p))


class _Value:
    def __init__(self, _typecode, value):
        self.value = value


class _LocalManager:
    """Stand-in for multiprocessing.Manager (everything runs in-process)."""

    def RLock(self):  # noqa: N802
        return threading.RLock()

    def Value(self, typecode, value):  # noqa: N802
        return _Value(typecode, value)

    def list(self):
        return []


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def _one_file(d: Path, pattern: str) -> Path:
    files = sorted(d.glob(pattern))
    if not files:
        raise NodeError(f"no {pattern} in {d}", hint="run `verdi setup mast3r_slam`",
                        kind="setup")
    return files[0]


def slam(ctx: Context) -> None:
    import cv2

    _setup_paths(ctx.repo)
    import torch
    import lietorch
    from mast3r_slam.config import config, load_config
    from mast3r_slam.frame import SharedKeyframes, create_frame
    from mast3r_slam.global_opt import FactorGraph
    from mast3r_slam.geometry import constrain_points_to_ray
    from mast3r_slam.lietorch_utils import as_SE3
    from mast3r_slam.mast3r_utils import (load_mast3r, load_retriever,
                                          mast3r_inference_mono, resize_img)
    from mast3r_slam.tracker import FrameTracker

    if ctx.device != "cuda":
        raise NodeError("MASt3R-SLAM needs a CUDA GPU (custom CUDA kernels)")
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # ---- frames
    all_frames = io.list_frames(ctx.input("frames"))
    step = max(1, int(ctx.param("subsample", 1)))
    sel = list(range(0, len(all_frames), step))
    if len(sel) < 2:
        raise NodeError(f"need at least 2 frames, got {len(sel)}")
    first = cv2.imread(str(all_frames[0]))
    if first is None:
        raise NodeError(f"cannot read {all_frames[0]}")
    H1, W1 = first.shape[:2]

    # ---- config (upstream config/base.yaml, single-threaded)
    load_config(str(ctx.repo / "config" / "base.yaml"))
    use_calib = ctx.has_input("camera")
    config["use_calib"] = use_calib
    config["single_thread"] = True

    _, (scale_w, scale_h, half_w, half_h) = resize_img(
        np.zeros((H1, W1, 3)), IMG_SIZE, return_transformation=True)
    probe = resize_img(np.zeros((H1, W1, 3)), IMG_SIZE)
    h, w = probe["img"][0].shape[1:]

    # ---- calibration (same as upstream dataloader.Intrinsics.from_calib)
    K_frame = None
    remap = None
    K_opt = None
    if use_calib:
        cam = io.read_camera(ctx.input("camera"))
        if (int(cam["width"]), int(cam["height"])) != (W1, H1):
            raise NodeError(
                f"camera is {cam['width']}x{cam['height']} but frames are {W1}x{H1}",
                hint="scale K to the frame resolution")
        K = np.asarray(cam["K"], dtype=np.float64)
        dist = np.asarray(cam["dist"], dtype=np.float64)
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, dist, (W1, H1), 0, (W1, H1),
            centerPrincipalPoint=config["dataset"]["center_principle_point"])
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_opt,
                                                 (W1, H1), cv2.CV_32FC1)
        remap = (mapx, mapy)
        K_frame = K_opt.copy()
        K_frame[0, 0] = K_opt[0, 0] / scale_w
        K_frame[1, 1] = K_opt[1, 1] / scale_h
        K_frame[0, 2] = K_opt[0, 2] / scale_w - half_w
        K_frame[1, 2] = K_opt[1, 2] / scale_h - half_h

    def load(idx: int) -> np.ndarray:
        bgr = cv2.imread(str(all_frames[idx]))
        if bgr is None:
            raise NodeError(f"cannot read {all_frames[idx]}")
        if bgr.shape[:2] != (H1, W1):
            raise NodeError(f"{all_frames[idx].name} is {bgr.shape[1]}x{bgr.shape[0]}, "
                            f"expected {W1}x{H1}", hint="all frames must share one resolution")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if remap is not None:
            img = cv2.remap(img, remap[0], remap[1], cv2.INTER_LINEAR)
        return img.astype(np.float32) / 255.0

    # ---- model, keyframe store, tracker, backend
    model = load_mast3r(str(_one_file(ctx.weight("mast3r", prefetch=True), "*.pth")),
                        device=dev)
    model.eval()
    ckdir = Path(tempfile.mkdtemp(prefix="mast3r_slam_ckpt_"))
    retr_src = _one_file(ctx.weight("retrieval"), "*.pth")
    cb_src = _one_file(ctx.weight("codebook"), "*.pkl")
    (ckdir / retr_src.name).symlink_to(retr_src)
    # upstream derives the codebook path from the retrieval checkpoint name
    cb_name = "_".join(retr_src.name.split("_")[:-1]) + "_codebook.pkl"
    (ckdir / cb_name).symlink_to(cb_src)

    max_kf = int(ctx.param("max_keyframes", 512))
    share = torch.Tensor.share_memory_
    torch.Tensor.share_memory_ = lambda self: self  # single process: no shm
    try:
        keyframes = SharedKeyframes(_LocalManager(), h, w, buffer=max_kf,
                                    device=dev)
    finally:
        torch.Tensor.share_memory_ = share
    K_t = None
    if use_calib:
        K_t = torch.from_numpy(K_frame).to(dev, dtype=torch.float32)
        keyframes.set_intrinsics(K_t)
    tracker = FrameTracker(model, keyframes, dev)
    factor_graph = FactorGraph(model, keyframes, K_t, dev)
    retrieval = load_retriever(model, str(ckdir / retr_src.name), device=dev)
    t_loaded = time.time()

    def solve():
        if use_calib:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

    loop_edges = []

    def append_kf(frame) -> int:
        if len(keyframes) >= max_kf:
            raise NodeError(
                f"more than max_keyframes={max_kf} keyframes after "
                f"{frame.frame_id} frames",
                hint="raise max_keyframes (~9 MB GPU each at 512x384) or "
                "set subsample")
        keyframes.append(frame)
        return len(keyframes) - 1

    def backend(idx: int) -> None:
        # upstream run_backend: previous keyframe + retrieval candidates
        kf_idx = [idx - 1] if idx > 0 else []
        frame = keyframes[idx]
        retrieval_inds = retrieval.update(
            frame, add_after_query=True, k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"])
        kf_idx += retrieval_inds
        lc = set(retrieval_inds)
        lc.discard(idx - 1)
        loop_edges.extend([int(j), idx] for j in sorted(lc))
        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        if kf_idx:
            factor_graph.add_factors(kf_idx, [idx] * len(kf_idx),
                                     config["local_opt"]["min_match_frac"])
        solve()

    def relocalize(frame) -> bool:
        # upstream main.relocalization
        kf_idx = retrieval.update(frame, add_after_query=False,
                                  k=config["retrieval"]["k"],
                                  min_thresh=config["retrieval"]["min_thresh"])
        if not kf_idx:
            return False
        n_kf = append_kf(frame) + 1
        kf_idx = list(kf_idx)
        if factor_graph.add_factors([n_kf - 1] * len(kf_idx), kf_idx,
                                    config["reloc"]["min_match_frac"],
                                    is_reloc=config["reloc"]["strict"]):
            retrieval.update(frame, add_after_query=True,
                             k=config["retrieval"]["k"],
                             min_thresh=config["retrieval"]["min_thresh"])
            keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            solve()
            return True
        keyframes.pop_last()
        return False

    # ---- main loop (upstream main.py, single-threaded)
    INIT, TRACKING, RELOC = 0, 1, 2
    mode = INIT
    records = {}          # input index -> ("kf", k) | ("rel", k, Sim3 data)
    n_reloc = 0
    T_prev = lietorch.Sim3.Identity(1, device=dev)
    for n, idx in enumerate(sel):
        frame = create_frame(idx, load(idx), T_prev, img_size=IMG_SIZE,
                             device=dev)
        if mode == INIT:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            k = append_kf(frame)
            backend(k)
            records[idx] = ("kf", k)
            mode = TRACKING
        elif mode == TRACKING:
            add_kf, _, try_reloc = tracker.track(frame)
            if try_reloc:
                mode = RELOC
            elif add_kf:
                k = append_kf(frame)
                backend(k)
                records[idx] = ("kf", k)
            else:
                kref = len(keyframes) - 1
                rel = keyframes[kref].T_WC.inv() * frame.T_WC
                records[idx] = ("rel", kref, rel.data.detach().clone())
        else:  # RELOC
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            if relocalize(frame):
                n_reloc += 1
                records[idx] = ("kf", len(keyframes) - 1)
                mode = TRACKING
        T_prev = frame.T_WC
        if n % 100 == 0:
            ctx.log(f"frame {n + 1}/{len(sel)}  keyframes {len(keyframes)}  "
                    f"{(n + 1) / max(time.time() - t_loaded, 1e-6):.1f} fps")
    t_slam = time.time() - t_loaded
    n_kf = len(keyframes)

    # ---- poses
    def mat(T) -> np.ndarray:
        return as_SE3(T).matrix()[0].double().numpy()

    kf_T = [keyframes[k].T_WC for k in range(n_kf)]
    kf_src = [int(keyframes.dataset_idx[k]) for k in range(n_kf)]
    order, Ts = [], []
    for idx in sel:
        r = records.get(idx)
        if r is None:
            continue
        if r[0] == "kf":
            T = kf_T[r[1]]
        else:
            T = kf_T[r[1]] * lietorch.Sim3(r[2])
        order.append(idx)
        Ts.append(mat(T))
    lost = [idx for idx in sel if idx not in records]

    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {
        "T_world_cam": [T.tolist() for T in Ts],
        "frame_index": order, "timestamps": order,
        "frames": [all_frames[i].name for i in order],
        "scale": "arbitrary (monocular SLAM), not metres",
        "scale_status": "relative",
        "pose_source": "MASt3R-SLAM visual tracking + backend optimisation",
        "world": "camera frame of the first keyframe"})
    kfj = ctx.output_path("keyframes", "keyframes.json")
    io.write_json(kfj, {
        "T_world_cam": [mat(T).tolist() for T in kf_T],
        "frame_index": kf_src, "timestamps": kf_src,
        "frames": [all_frames[i].name for i in kf_src],
        "scale": "arbitrary (monocular SLAM), not metres",
        "scale_status": "relative",
        "pose_source": "MASt3R-SLAM visual tracking + backend optimisation",
        "world": "camera frame of the first keyframe"})

    # ---- fused point cloud (upstream evaluate.save_reconstruction) + depth
    c_thr = float(ctx.param("conf_threshold", 1.5))
    save_depth = bool(ctx.param("save_depth", False))
    depth_dir = ctx.output_path("depth") if save_depth else None
    if save_depth:
        uu, vv = np.meshgrid(np.arange(W1, dtype=np.float32),
                             np.arange(H1, dtype=np.float32))
        mx = (uu + 0.5) / scale_w - 0.5 - half_w
        my = (vv + 0.5) / scale_h - 0.5 - half_h
    pts, cols = [], []
    for k in range(n_kf):
        kf = keyframes[k]
        X = kf.X_canon
        if use_calib:
            X = constrain_points_to_ray(kf.img_shape.flatten()[:2], X[None],
                                        K_t).squeeze(0)
        pW = kf.T_WC.act(X).cpu().numpy().reshape(-1, 3)
        conf = kf.get_average_conf().cpu().numpy().reshape(-1)
        valid = conf > c_thr
        col = (kf.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        pts.append(pW[valid].astype(np.float32))
        cols.append(col[valid])
        if save_depth:
            z = X[:, 2].cpu().numpy().reshape(h, w).astype(np.float32)
            ok = (valid.reshape(h, w) & (z > 0)).astype(np.float32)
            zf = cv2.remap(z * ok, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            okf = cv2.remap(ok, mx, my, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            zf[okf < 0.999] = 0
            io.write_depth(depth_dir / io.frame_name(k, ".npy"), zf)
    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    if len(pts) == 0:
        raise NodeError("no confident points", hint="lower conf_threshold")
    max_points = int(ctx.param("max_points", 2_000_000))
    if len(pts) > max_points:
        s = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[s], cols[s]
    ply = ctx.output_path("pointcloud", "pointcloud.ply")
    _write_ply(ply, pts, cols)

    ctx.set_output("trajectory", traj)
    ctx.set_output("keyframes", kfj)
    ctx.set_output("pointcloud", ply)
    if save_depth:
        ctx.set_output("depth", depth_dir)
    peak = torch.cuda.max_memory_allocated() / 2**30
    ctx.metadata.update({
        "mode": "calibrated" if use_calib else "uncalibrated",
        "network_resolution": [int(w), int(h)],
        "frames_processed": len(sel),
        "frames_with_pose": len(order),
        "lost_frames": lost,
        "keyframes": n_kf,
        "loop_closure_edges": loop_edges,
        "relocalisations": n_reloc,
        "slam_seconds": round(t_slam, 2),
        "slam_fps": round(len(sel) / max(t_slam, 1e-6), 2),
        "total_seconds": round(time.time() - t_start, 2),
        "peak_vram_allocated_gb": round(peak, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "units": "arbitrary (monocular SLAM); not metres",
    })
    if use_calib:
        ctx.metadata["camera_undistorted"] = {
            "K": K_opt.tolist(), "width": W1, "height": H1,
            "note": "frames were undistorted to this pinhole before SLAM"}


if __name__ == "__main__":
    raise SystemExit(main({"slam": slam}))
