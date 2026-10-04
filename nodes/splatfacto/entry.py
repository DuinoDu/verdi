"""nerfstudio splatfacto node: train (posed images -> 3DGS ply).

Mirrors SimFoundry auto_bg 5_train_bg_splat.py: build a nerfstudio
transforms.json from known poses/intrinsics (no COLMAP), seed points from
depth, `ns-train <method> ... nerfstudio-data --auto-scale-poses False
--center-method none --orientation-method none --scale-factor 1.0`, then
`ns-export gaussian-splat`.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

# OpenCV camera axes (x right, y down, z forward) -> nerfstudio / OpenGL
# camera axes (x right, y up, z backward): flip the camera y and z axes.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

# torch>=2.6 defaults torch.load(weights_only=True); nerfstudio 1.1.5
# checkpoints hold non-tensor objects (same shim as SimFoundry's ns-export).
TORCH_LOAD_SHIM = (
    "import torch; _orig = torch.load; "
    "torch.load = lambda *a, **kw: _orig(*a, **{'weights_only': False, **kw})"
)


def _jit_path() -> str:
    """PATH with the venv bin (ninja) and $CUDA_HOME/bin (nvcc) up front."""
    parts = [str(Path(sys.executable).parent)]
    if os.environ.get("CUDA_HOME"):
        parts.append(os.path.join(os.environ["CUDA_HOME"], "bin"))
    return os.pathsep.join(parts + [os.environ.get("PATH", "")])


ALEXNET_FILE = "alexnet-owt-7be5be79.pth"


def _torch_home(ctx: Context, work: Path) -> str:
    """TORCH_HOME holding the pinned torchvision alexnet weights.

    splatfacto builds torchmetrics LPIPS (alexnet backbone) at model init,
    which would otherwise download them from download.pytorch.org.
    """
    th = work / "torch_home"
    ck = th / "hub" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    link = ck / ALEXNET_FILE
    if not link.exists():
        os.symlink(ctx.weight("alexnet") / ALEXNET_FILE, link)
    return str(th)


def _env() -> dict:
    env = dict(os.environ)
    env["PATH"] = _jit_path()  # ninja (+ nvcc) for the gsplat JIT
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"
    for k in ("NERFSTUDIO_MCMC", "NERFSTUDIO_MCMC_CAP_MAX",
              "NERFSTUDIO_MCMC_NOISE_LR", "NERFSTUDIO_DEPTH_LOSS",
              "NERFSTUDIO_DEPTH_LOSS_MULT", "NERFSTUDIO_DEPTH_DIR",
              "NERFSTUDIO_DEPTH_LOSS_MIN"):
        env.pop(k, None)
    return env


def _run(cmd, cwd: Path, env: dict, what: str) -> None:
    Context.log("$", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise NodeError(f"{what} failed (exit {proc.returncode})",
                        hint="see log.txt of the run for the nerfstudio "
                        "traceback")


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape == (h, w):
        return arr
    import cv2

    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


def _load_inputs(ctx: Context):
    frames = io.list_frames(ctx.input("frames"))
    Ts = np.asarray(io.read_json(ctx.input("trajectory"))["T_world_cam"],
                    dtype=np.float64)
    if len(Ts) != len(frames):
        raise NodeError(f"trajectory has {len(Ts)} poses but frames has "
                        f"{len(frames)} images",
                        hint="one T_world_cam per frame, in frame order")
    cam = io.read_camera(ctx.input("camera"))
    h, w = io.read_image(frames[0]).shape[:2]
    K = cam["K"].copy()
    sx, sy = w / float(cam["width"]), h / float(cam["height"])
    if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
        ctx.log(f"rescaling K from {cam['width']}x{cam['height']} to {w}x{h}")
        K[0] *= sx
        K[1] *= sy
    if np.any(np.abs(cam["dist"]) > 1e-9):
        ctx.log("warning: camera distortion is ignored (pinhole assumed)")
    masks = None
    if ctx.has_input("masks"):
        mfiles = io.list_frames(ctx.input("masks"), (".png",))
        if len(mfiles) != len(frames):
            raise NodeError(f"masks has {len(mfiles)} frames, frames has "
                            f"{len(frames)}")
        masks = mfiles
    depths = None
    if ctx.has_input("depth"):
        dfiles = sorted(Path(ctx.input("depth")).glob("*.npy"))
        if len(dfiles) != len(frames):
            raise NodeError(f"depth has {len(dfiles)} frames, frames has "
                            f"{len(frames)}")
        depths = dfiles
    return frames, Ts, K, (h, w), masks, depths


def _keep_mask(masks, i: int, h: int, w: int) -> np.ndarray:
    if masks is None:
        return np.ones((h, w), bool)
    from PIL import Image

    m = np.asarray(Image.open(masks[i]))
    return _resize_nearest(m, h, w) == 0


def _seed_from_depth(frames, Ts, K, hw, masks, depths, max_points, seed):
    h, w = hw
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xyz, rgb = [], []
    for i, f in enumerate(frames):
        d = _resize_nearest(np.load(depths[i]).astype(np.float32), h, w)
        keep = (d > 0) & np.isfinite(d) & _keep_mask(masks, i, h, w)
        if not keep.any():
            continue
        z = d[keep].astype(np.float64)
        x = (u[keep] - K[0, 2]) * z / K[0, 0]
        y = (v[keep] - K[1, 2]) * z / K[1, 1]
        pc = np.stack([x, y, z], 1)
        xyz.append(pc @ Ts[i][:3, :3].T + Ts[i][:3, 3])
        rgb.append(io.read_image(f)[keep])
    if not xyz:
        raise NodeError("depth has no valid pixels for the seed point cloud")
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    if len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points,
                                                 replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def _read_points(path: Path):
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    names = v.data.dtype.names
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, np.uint8)
    return xyz, rgb


def _write_seed_ply(path: Path, xyz, rgb) -> None:
    from plyfile import PlyData, PlyElement

    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"),
                                    ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def _build_dataset(ctx, data_dir: Path, frames, Ts, K, hw, masks, depths,
                   seed_ply: bool):
    h, w = hw
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    if masks is not None:
        (data_dir / "masks").mkdir(parents=True, exist_ok=True)
    entries = []
    for i, f in enumerate(frames):
        name = f"frame_{i:05d}{f.suffix.lower()}"
        dst = data_dir / "images" / name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
        e = {"file_path": f"images/{name}"}
        if masks is not None:
            from PIL import Image

            # nerfstudio masks: 0 = ignore, 255 = use
            keep = _keep_mask(masks, i, h, w)
            mname = f"frame_{i:05d}.png"
            Image.fromarray(np.where(keep, 255, 0).astype(np.uint8)).save(
                data_dir / "masks" / mname)
            e["mask_path"] = f"masks/{mname}"
        e.update({
            "transform_matrix": (Ts[i] @ CV_TO_GL).tolist(),
            "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]), "w": w, "h": h,
        })
        entries.append(e)
    meta = {"camera_model": "OPENCV", "frames": entries}
    if seed_ply:
        meta["ply_file_path"] = "points3d.ply"
    io.write_json(data_dir / "transforms.json", meta)
    if depths is not None:
        ddir = data_dir / "depths"
        ddir.mkdir(exist_ok=True)
        for i, p in enumerate(depths):
            d = _resize_nearest(np.load(p).astype(np.float32), h, w).copy()
            d[~np.isfinite(d)] = 0
            if masks is not None:
                d[~_keep_mask(masks, i, h, w)] = 0
            # the depth-loss patch globs frame_*.npy in sorted order
            np.save(ddir / f"frame_{i:05d}.npy", d)


def _render_views(ctx: Context, config_path: Path, frames, out_dir: Path):
    import torch

    _orig = torch.load  # same weights_only shim as TORCH_LOAD_SHIM
    torch.load = lambda *a, **kw: _orig(*a, **{"weights_only": False, **kw})
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
    model = pipeline.model
    model.eval()
    cameras = pipeline.datamanager.train_dataset.cameras.to(model.device)
    if len(cameras) != len(frames):
        raise NodeError(f"nerfstudio loaded {len(cameras)} cameras for "
                        f"{len(frames)} frames")
    psnrs = []
    for i in range(len(cameras)):
        with torch.no_grad():
            out = model.get_outputs_for_camera(cameras[i:i + 1])
        rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(
            np.uint8)
        gt = io.read_image(frames[i])
        if rgb.shape != gt.shape:
            raise NodeError(f"render shape {rgb.shape} != frame {gt.shape}")
        io.write_image(out_dir / io.frame_name(i), rgb)
        mse = np.mean((rgb.astype(np.float64) - gt) ** 2)
        psnrs.append(float(10 * math.log10(255.0 ** 2 / max(mse, 1e-10))))
    return psnrs


def train(ctx: Context) -> None:
    # gsplat JIT-loads its CUDA extension in this process too (rendering);
    # torch needs ninja (venv bin) on PATH even to load the cached build.
    os.environ["PATH"] = _jit_path()
    frames, Ts, K, hw, masks, depths = _load_inputs(ctx)
    work = ctx.output_path("_work")
    data_dir = work / "ns_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seed = int(ctx.param("seed", 42))
    if ctx.has_input("points"):
        xyz, rgb = _read_points(ctx.input("points"))
    elif depths is not None:
        xyz, rgb = _seed_from_depth(frames, Ts, K, hw, masks, depths,
                                    int(ctx.param("seed_max_points")), seed)
    else:
        xyz = None
        ctx.log("no depth/points: splatfacto random initialisation")
    if xyz is not None:
        _write_seed_ply(data_dir / "points3d.ply", xyz, rgb)
        ctx.log(f"seed point cloud: {len(xyz)} points")
    _build_dataset(ctx, data_dir, frames, Ts, K, hw, masks, depths,
                   seed_ply=xyz is not None)

    os.environ["TORCH_HOME"] = _torch_home(ctx, work)  # also for _render_views
    env = _env()
    use_depth = depths is not None and bool(ctx.param("use_depth_loss"))
    if use_depth:
        env["NERFSTUDIO_DEPTH_LOSS"] = "1"
        env["NERFSTUDIO_DEPTH_LOSS_MULT"] = str(ctx.param("depth_loss_mult"))
        env["NERFSTUDIO_DEPTH_DIR"] = str((data_dir / "depths").resolve())
    method = ctx.param("method")
    iters = int(ctx.param("max_num_iterations"))
    py = sys.executable
    train_cmd = [
        py, "-c", TORCH_LOAD_SHIM + "; import sys; sys.argv[0] = 'ns-train'; "
        "from nerfstudio.scripts.train import entrypoint; entrypoint()",
        method,
        "--max-num-iterations", iters,
        "--data", data_dir.resolve(),
        "--output-dir", (work / "outputs").resolve(),
        "--experiment-name", "splat", "--timestamp", "run",
        "--vis", "tensorboard",
        "--machine.seed", seed,
        "--steps-per-save", max(iters, 1),
        f"--pipeline.model.camera-optimizer.mode={ctx.param('camera_optimizer')}",
        "nerfstudio-data",
        "--auto-scale-poses", "False",
        "--center-method", "none",
        "--orientation-method", "none",
        "--scale-factor", "1.0",
        "--downscale-factor", "1",
        "--eval-mode", "all",
        "--load-3D-points", "True" if xyz is not None else "False",
    ]
    _run(train_cmd, work, env, "ns-train")
    configs = sorted((work / "outputs").glob("**/config.yml"))
    if not configs:
        raise NodeError("ns-train produced no config.yml")
    config_path = configs[-1]

    export_dir = work / "export"
    export_cmd = [
        py, "-c", TORCH_LOAD_SHIM + "; import sys; sys.argv = ['ns-export', "
        f"'gaussian-splat', '--load-config', {str(config_path)!r}, "
        f"'--output-dir', {str(export_dir)!r}]; "
        "from nerfstudio.scripts.exporter import entrypoint; entrypoint()",
    ]
    _run(export_cmd, work, env, "ns-export")
    ply = export_dir / "splat.ply"
    if not ply.exists():
        raise NodeError("ns-export did not write splat.ply")
    out_ply = ctx.output_path("gaussians", "gaussians.ply")
    shutil.move(str(ply), out_ply)
    ctx.set_output("gaussians", out_ply)

    renders = ctx.output_path("renders")
    psnrs = _render_views(ctx, config_path, frames, renders)
    ctx.set_output("renders", renders)

    from verdi.types.registry import validate

    n = validate("gaussians", out_ply)["gaussians"]
    metrics = {"psnr_mean": float(np.mean(psnrs)),
               "psnr_min": float(np.min(psnrs)), "psnr": psnrs,
               "num_gaussians": int(n), "iterations": iters,
               "method": method, "depth_loss": use_depth,
               "seed_points": 0 if xyz is None else int(len(xyz))}
    mpath = ctx.output_path("metrics", "metrics.json")
    io.write_json(mpath, metrics)
    ctx.set_output("metrics", mpath)
    ctx.metadata.update({"config": str(config_path)})
    shutil.rmtree(work / "ns_data" / "images", ignore_errors=True)
    shutil.rmtree(work / "torch_home", ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main({"train": train}))
