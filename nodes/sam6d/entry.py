"""SAM-6D node: estimate (render templates -> ISM -> PEM), no Docker.

The three official scripts are run as subprocesses of the node venv
(exactly like SAM-6D/demo.sh); entry.py only converts the unified types
to the upstream file conventions (mm mesh, uint16 mm depth png,
camera.json {cam_K, depth_scale}) and back (T_cam_obj in metres).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

N_TEMPLATES = 42


def _run(cmd, cwd: Path, what: str, env=None) -> None:
    Context.log(f"[sam6d] {what}: {' '.join(str(c) for c in cmd)}")
    env = dict(env or os.environ)
    env["PWD"] = str(cwd)  # blenderproc resolves relative paths with $PWD
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if proc.returncode != 0:
        raise NodeError(f"SAM-6D {what} failed (exit {proc.returncode})",
                        hint="see log.txt of the run for the upstream error")


def _mesh_to_mm(src: Path, work: Path) -> Path:
    """Scale the metric mesh to millimetres (SAM-6D convention)."""
    import trimesh

    mesh = trimesh.load(src, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise NodeError(f"could not load a triangle mesh from {src}")
    ext = mesh.bounds[1] - mesh.bounds[0]
    if float(ext.max()) > 20.0:
        raise NodeError(
            f"mesh extent {ext.max():.1f} looks like millimetres, expected "
            "metres", hint="scale the mesh to metres (mesh type convention)")
    mesh.apply_scale(1000.0)
    if src.suffix.lower() == ".ply":
        out = work / "mesh_mm.ply"
        mesh.export(out)
    else:
        out = work / "mesh_mm" / "mesh_mm.obj"
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out, include_texture=True)
    return out


def _rle(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE (column-major), as written by the ISM."""
    flat = np.asarray(mask, bool).flatten(order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(np.concatenate([[0], flat, [0]])))
    runs = np.diff(np.concatenate([[0], change, [flat.size]]))
    counts = runs.tolist()
    if counts and counts[-1] == 0:
        counts = counts[:-1]
    return {"size": list(mask.shape), "counts": [int(c) for c in counts]}


def _decode(seg: dict) -> np.ndarray:
    import pycocotools.mask as cocomask

    h, w = seg["size"]
    try:
        rle = cocomask.frPyObjects(seg, h, w)
    except Exception:  # noqa: BLE001 - compressed RLE
        rle = seg
    return cocomask.decode(rle).astype(bool)


def _dets_from_mask(mask_path: Path, shape) -> list:
    from PIL import Image

    ids = np.asarray(Image.open(mask_path))
    if ids.shape != tuple(shape):
        raise NodeError(f"mask size {ids.shape} != image size {shape}")
    dets = []
    for i in sorted(int(v) for v in np.unique(ids) if v != 0):
        m = ids == i
        ys, xs = np.nonzero(m)
        dets.append({
            "scene_id": 0, "image_id": 0, "category_id": 1, "score": 1.0,
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1),
                     int(ys.max() - ys.min() + 1)],
            "segmentation": _rle(m), "time": 0.0})
    if not dets:
        raise NodeError("input mask is empty (no instance ids > 0)")
    return dets


def estimate(ctx: Context) -> None:
    from PIL import Image

    root = ctx.repo / "SAM-6D"
    blender = Path(os.environ.get("SAM6D_BLENDER", ""))
    if not (blender / "blender").exists():
        raise NodeError(f"Blender not found at {blender}",
                        hint="run `verdi setup sam6d`", kind="setup")
    py = Path(sys.executable)
    work = ctx.output_path("_work")

    # ---- inputs -> upstream conventions
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    depth = np.load(ctx.input("depth")).astype(np.float32)
    if depth.shape != (h, w):
        raise NodeError(f"depth {depth.shape} and image {(h, w)} sizes differ",
                        hint="depth must be aligned to the RGB image")
    depth[~np.isfinite(depth)] = 0
    if float(depth.max(initial=0)) > 65.0:
        raise NodeError("depth max > 65 m; depth must be metres")
    cam = io.read_camera(ctx.input("camera"))
    rgb_path = io.write_image(work / "rgb.png", rgb)
    depth_mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
    depth_path = work / "depth.png"
    Image.fromarray(depth_mm).save(depth_path)
    cam_path = io.write_json(work / "camera.json", {
        "cam_K": np.asarray(cam["K"], float).reshape(-1).tolist(),
        "depth_scale": 1.0})
    mesh_path = _mesh_to_mm(ctx.input("mesh"), work)
    label = Path(ctx.input("mesh")).stem

    # ---- 1. render templates (official Render/render_custom_templates.py)
    _run([py.parent / "blenderproc", "run",
          root / "Render" / "render_custom_templates.py",
          "--output_dir", work, "--cad_path", mesh_path,
          "--custom-blender-path", blender],
         root / "Render", "template rendering")
    tdir = work / "templates"
    missing = [n for i in range(N_TEMPLATES)
               for n in (f"rgb_{i}.png", f"mask_{i}.png", f"xyz_{i}.npy")
               if not (tdir / n).exists()]
    if missing:
        raise NodeError(f"template rendering incomplete, missing {missing[:3]}")

    # ---- 2. instance segmentation (or user mask)
    res_dir = work / "sam6d_results"
    res_dir.mkdir(exist_ok=True)
    seg_path = res_dir / "detection_ism.json"
    if ctx.has_input("mask"):
        io.write_json(seg_path, _dets_from_mask(ctx.input("mask"), (h, w)))
        ctx.metadata["segmentation"] = "input mask"
    else:
        _run([py, "run_inference_custom.py", "--segmentor_model", "sam",
              "--output_dir", work, "--cad_path", mesh_path,
              "--rgb_path", rgb_path, "--depth_path", depth_path,
              "--cam_path", cam_path, "--stability_score_thresh",
              float(ctx.param("stability_score_thresh", 0.97))],
             root / "Instance_Segmentation_Model", "instance segmentation")
        ctx.metadata["segmentation"] = "ISM (SAM vit_h + DINOv2)"
    if not seg_path.exists() or not json.loads(seg_path.read_text()):
        raise NodeError("SAM-6D found no candidate instance of the object",
                        hint="lower stability_score_thresh or pass a mask")
    thresh = float(ctx.param("det_score_thresh", 0.2))
    n_cand = sum(d["score"] > thresh
                 for d in json.loads(seg_path.read_text()))
    ctx.metadata["candidates"] = n_cand
    if n_cand == 0:
        raise NodeError(f"no ISM candidate has score > det_score_thresh "
                        f"{thresh}", hint="lower det_score_thresh")

    # ---- 3. pose estimation
    _run([py, "run_inference_custom.py", "--output_dir", work,
          "--cad_path", mesh_path, "--rgb_path", rgb_path,
          "--depth_path", depth_path, "--cam_path", cam_path,
          "--seg_path", seg_path, "--det_score_thresh", thresh],
         root / "Pose_Estimation_Model", "pose estimation")
    pem = json.loads((res_dir / "detection_pem.json").read_text())
    if not pem:
        raise NodeError("pose estimation returned no pose")

    # ---- outputs (mm -> m)
    pem = sorted(pem, key=lambda d: -float(d["score"]))
    pem = pem[:max(1, int(ctx.param("top_k", 1)))]
    ids = np.zeros((h, w), np.uint8 if len(pem) < 255 else np.uint16)
    poses, labels = [], {}
    for i in range(len(pem), 0, -1):  # best pose painted last (on top)
        ids[_decode(pem[i - 1]["segmentation"])] = i
    for i, d in enumerate(pem, start=1):
        T = np.eye(4)
        T[:3, :3] = np.asarray(d["R"], float)
        T[:3, 3] = np.asarray(d["t"], float) / 1000.0
        x, y, bw, bh = d["bbox"]
        poses.append({"T_cam_obj": T, "label": label,
                      "score": float(d["score"]), "mask_id": i,
                      "bbox_xyxy": [float(x), float(y), float(x + bw),
                                    float(y + bh)]})
        labels[i] = {"label": label, "score": float(d["score"])}
    out = ctx.output_path("poses", "poses.json")
    io.write_pose_set(out, poses)
    ctx.set_output("poses", out)
    mpath = ctx.output_path("mask", "mask.png")
    io.write_mask(mpath, ids, labels)
    ctx.set_output("mask", mpath)
    vis = ctx.output_path("vis", "vis.png")
    shutil.copy(res_dir / "vis_pem.png", vis)
    ctx.set_output("vis", vis)
    if not ctx.param("keep_work", False):
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main({"estimate": estimate}))
