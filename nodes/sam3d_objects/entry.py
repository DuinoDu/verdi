"""SAM 3D Objects node: image + object mask(s) -> per-object textured mesh
plus pose / size in the OpenCV camera frame."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

CFGS = ("pipeline", "ss_generator", "slat_generator", "ss_decoder",
        "slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh")
CKPTS = CFGS[1:]
DINOV2_REPO = "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CKPT = "dinov2_vitl14_reg4_pretrain.pth"
# upstream to_glb: v_glb = v_local @ A (z-up -> y-up); column form v_local = A @ v_glb
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
# PyTorch3D camera (x left, y up, z forward) -> OpenCV (x right, y down, z forward)
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0])


def _no_fa3_off_hopper() -> None:
    """xformers 0.0.32 enables FlashAttention-3 (Hopper sm_90a only) for any
    GPU with compute capability >= 9.0; on sm_120 it aborts. Turn it off
    there (see nodes/trellis2)."""
    import torch
    from xformers.ops.fmha import dispatch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0):
        dispatch._set_use_fa3(False)


def _pipeline(ctx: Context):
    """Instantiate upstream InferencePipelinePointMap from pinned local files."""
    import torch
    from omegaconf import OmegaConf

    if ctx.device != "cuda":
        raise NodeError("SAM 3D Objects needs a CUDA GPU")
    _no_fa3_off_hopper()
    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # DINOv2 conditioning: upstream torch.hub.load("facebookresearch/dinov2",
    # source="github"); point it at the pinned local repo + weights instead
    hub = Path(tempfile.mkdtemp(prefix="s3o_hub_"))
    (hub / "checkpoints").mkdir()
    (hub / "checkpoints" / DINOV2_CKPT).symlink_to(
        ctx.weight("dinov2_ckpt", prefetch=True) / DINOV2_CKPT)
    torch.hub.set_dir(str(hub))
    dino_repo = str(ctx.weight("dinov2_repo") / DINOV2_REPO)

    ws = Path(tempfile.mkdtemp(prefix="s3o_ckpt_"))
    for key in CKPTS:
        (ws / f"{key}.ckpt").symlink_to(
            ctx.weight(key, prefetch=True) / f"{key}.ckpt")
    for key in CFGS:
        cfg = OmegaConf.load(ctx.weight(f"{key}_cfg") / f"{key}.yaml")
        _local_dino(cfg, dino_repo)
        OmegaConf.save(cfg, ws / f"{key}.yaml")
    config = OmegaConf.load(ws / "pipeline.yaml")
    config.workspace_dir = str(ws)
    config.rendering_engine = "pytorch3d"  # as upstream notebook/inference.py
    config.compile_model = False
    config.depth_model.model.pretrained_model_name_or_path = str(
        ctx.weight("moge_vitl", prefetch=True) / "model.pt")

    import sam3d_objects  # noqa: F401  (upstream: "do not remove this import")
    from hydra.utils import instantiate

    return instantiate(config)


def _local_dino(node, repo: str) -> None:
    from omegaconf import DictConfig, ListConfig

    if isinstance(node, DictConfig):
        if str(node.get("_target_", "")).endswith("embedder.dino.Dino"):
            node.repo_or_dir = repo
            node.source = "local"
        for v in node.values():
            _local_dino(v, repo)
    elif isinstance(node, ListConfig):
        for v in node:
            _local_dino(v, repo)


def _read_inputs(ctx: Context):
    from PIL import Image

    rgb = np.asarray(Image.open(ctx.input("image")).convert("RGB"))
    ids = np.asarray(Image.open(ctx.input("mask")))
    if ids.ndim == 3:
        ids = ids[..., 0]
    if ids.shape[:2] != rgb.shape[:2]:
        raise NodeError(f"mask size {ids.shape[1]}x{ids.shape[0]} != image "
                        f"size {rgb.shape[1]}x{rgb.shape[0]}")
    mid = int(ctx.param("mask_id", 0))
    obj_ids = [mid] if mid > 0 else sorted(int(i) for i in np.unique(ids) if i)
    obj_ids = [i for i in obj_ids if (ids == i).sum() >= 64]
    if not obj_ids:
        raise NodeError("no object in mask", hint="mask ids must be > 0 "
                        "(and cover at least 64 px); check `mask_id`")
    return rgb, ids, obj_ids


def _depth_pointmap(ctx: Context, hw):
    """HxWx3 point map (PyTorch3D camera frame, metres) from depth + K."""
    if not ctx.has_input("camera"):
        raise NodeError("`depth` needs `camera` (intrinsics)")
    cam = io.read_camera(ctx.input("camera"))
    K = cam["K"]
    depth = np.load(ctx.input("depth")).astype(np.float32)
    if depth.shape != tuple(hw):
        raise NodeError(f"depth size {depth.shape[::-1]} != image size {hw[::-1]}")
    v, u = np.mgrid[0:hw[0], 0:hw[1]].astype(np.float32)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pts = np.stack([-x, -y, depth], -1)  # OpenCV -> PyTorch3D (x, y flipped)
    valid = np.isfinite(depth) & (depth > 0)
    rtol = float(ctx.param("depth_edge_rtol", 0.04))
    if rtol > 0:
        # flying pixels at depth discontinuities (sensor depth mixes object
        # and background along the mask border): 3x3 max/min depth ratio,
        # as MoGe / utils3d depth_map_edge
        import cv2

        d = np.where(valid, depth, np.nan).astype(np.float32)
        k = np.ones((3, 3), np.uint8)
        dmax = cv2.dilate(np.nan_to_num(d, nan=0.0), k)
        dmin = -cv2.dilate(-np.nan_to_num(d, nan=np.inf), k)
        edge = valid & (dmax - dmin > rtol * depth)
        ctx.log(f"[sam3d_objects] depth edges removed: {int(edge.sum())} px")
        valid &= ~edge
    pts[~valid] = np.nan
    return pts, K


def _pose(out):
    """Upstream layout (PyTorch3D, z-up local frame) -> 4x4 similarity from
    the glb frame to the OpenCV camera frame."""
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    q = out["rotation"].reshape(-1, 4)[:1].float()
    t = out["translation"].reshape(-1, 3)[:1].float()
    s = out["scale"].reshape(-1, 3)[:1].float()
    tfm = compose_transform(scale=s, rotation=quaternion_to_matrix(q),
                            translation=t)
    M = tfm.get_matrix()[0].detach().cpu().double().numpy().T  # column form
    F4, A4 = np.eye(4), np.eye(4)
    F4[:3, :3], A4[:3, :3] = P3D_TO_CV, A
    S = F4 @ M @ A4  # p_cv = S @ [v_glb, 1]
    L = S[:3, :3]
    scale = float(np.cbrt(np.linalg.det(L)))
    U, _, Vt = np.linalg.svd(L / scale)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = U @ Vt, S[:3, 3]
    return S, T, scale, s[0].cpu().numpy().tolist()


def _project(meshes_cam, K, hw, ids_order):
    """Instance-id silhouette of camera-frame meshes (far objects first)."""
    import cv2

    out = np.zeros(hw, np.uint16)
    depth_of = {k: float(np.median(m.vertices[:, 2])) for k, m in meshes_cam.items()}
    for k in sorted(ids_order, key=lambda k: -depth_of[k]):
        v = np.asarray(meshes_cam[k].vertices, np.float64)
        f = np.asarray(meshes_cam[k].faces)
        z = v[:, 2]
        uv = (v[:, :2] / np.maximum(z[:, None], 1e-6)) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        tri = f[(z[f] > 1e-6).all(1)]
        polys = np.round(uv[tri] * 8).astype(np.int32)  # 3 bits subpixel
        cv2.fillPoly(out, list(polys), int(k), lineType=cv2.LINE_8, shift=3)
    return out


def reconstruct(ctx: Context) -> None:
    import torch
    import trimesh

    rgb, ids, obj_ids = _read_inputs(ctx)
    hw = rgb.shape[:2]
    pipe = _pipeline(ctx)
    metric = ctx.has_input("depth")
    with pipe.device:
        if metric:
            pts, K = _depth_pointmap(ctx, hw)
            pointmap = torch.from_numpy(pts)
        else:
            # one MoGe point map for all objects (shared, scale-invariant)
            rgba = np.dstack([rgb, (ids > 0).astype(np.uint8) * 255])
            pm = pipe.compute_pointmap(rgba)
            pointmap = pm["pointmap"].permute(1, 2, 0).float().cpu()
            K = pm["intrinsics"].reshape(3, 3).double().cpu().numpy()
            if K[0, 2] < 2:  # normalised intrinsics (MoGe) -> pixels
                K = K * np.array([[hw[1]], [hw[0]], [1.0]])
    s1 = int(ctx.param("stage1_steps", 0)) or None
    s2 = int(ctx.param("stage2_steps", 0)) or None
    mesh_dir = ctx.output_path("meshes")
    poses, meshes_cam = [], {}
    for k in obj_ids:
        ctx.log(f"[sam3d_objects] object {k} ({(ids == k).sum()} px)")
        rgba = np.dstack([rgb, (ids == k).astype(np.uint8) * 255])
        out = pipe.run(
            rgba, None, seed=int(ctx.param("seed", 42)),
            with_mesh_postprocess=bool(ctx.param("mesh_postprocess", True)),
            with_texture_baking=bool(ctx.param("texture_baking", True)),
            with_layout_postprocess=bool(ctx.param("layout_refine", False)),
            use_vertex_color=True,
            stage1_inference_steps=s1, stage2_inference_steps=s2,
            pointmap=pointmap)
        if ctx.param("layout_refine", False) and "iou" not in out:
            # upstream catches every exception of the post-optimisation and
            # silently returns the unrefined layout; do not pass that off
            raise NodeError("layout post-optimisation failed (see log: "
                            "'Error during layout post optimization')",
                            hint="rerun with -p layout_refine=false")
        glb = out.get("glb")
        if glb is None or len(glb.faces) == 0:
            raise NodeError(f"empty mesh for object {k}",
                            hint="check the mask; try another seed")
        name = f"obj_{k}.glb"
        glb.export(str(mesh_dir / name))
        S, T, scale, scale_xyz = _pose(out)
        ext = np.asarray(glb.vertices).max(0) - np.asarray(glb.vertices).min(0)
        poses.append({
            "T_cam_obj": T, "label": f"obj_{k}", "mask_id": int(k),
            "scale": scale, "scale_xyz_upstream": scale_xyz,
            "size": (scale * ext).round(5).tolist(),
            "units": "metres" if metric else "relative",
            "mesh": f"meshes/{name}",
            "upstream": {key: out[key].reshape(-1).tolist()
                         for key in ("rotation", "translation", "scale")},
        })
        if "iou" in out:
            poses[-1]["layout_refine_iou"] = float(out["iou"])
            poses[-1]["layout_refine_iou_before"] = float(out["iou_before_optim"])
            poses[-1]["layout_refine_accepted"] = bool(out["optim_accepted"])
            if float(out["iou_before_optim"]) <= 0:
                raise NodeError(
                    "layout post-optimisation rendered no overlap with the "
                    "mask (IoU 0 before optimisation); refined pose is not "
                    "trustworthy", hint="rerun with -p layout_refine=false")
        mc = glb.copy()
        mc.apply_transform(S)
        meshes_cam[k] = mc
        torch.cuda.empty_cache()

    ctx.set_output("meshes", mesh_dir)
    p = ctx.output_path("poses", "poses.json")
    io.write_pose_set(p, poses)
    ctx.set_output("poses", p)
    scene = trimesh.Scene()
    for k, m in meshes_cam.items():
        scene.add_geometry(m, node_name=f"obj_{k}", geom_name=f"obj_{k}")
    p = ctx.output_path("scene", "scene.glb")
    scene.export(str(p))
    ctx.set_output("scene", p)
    p = ctx.output_path("camera", "camera.json")
    io.write_camera(p, K, hw[1], hw[0])
    ctx.set_output("camera", p)

    sil = _project(meshes_cam, K, hw, obj_ids)
    p = ctx.output_path("render_mask", "render_mask.png")
    io.write_mask(p, sil)
    ctx.set_output("render_mask", p)
    over = rgb.astype(np.float32).copy()
    import cv2

    for i, k in enumerate(obj_ids):
        col = np.array(cv2.applyColorMap(np.uint8([[(37 * i + 60) % 256]]),
                                         cv2.COLORMAP_JET)[0, 0, ::-1], np.float32)
        m = sil == k
        over[m] = 0.45 * over[m] + 0.55 * col
        cnt, _ = cv2.findContours((ids == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
        over_u8 = np.ascontiguousarray(over.astype(np.uint8))
        cv2.drawContours(over_u8, cnt, -1, (255, 255, 255), 1)
        over = over_u8.astype(np.float32)
    p = ctx.output_path("overlay", "overlay.png")
    io.write_image(p, over.astype(np.uint8))
    ctx.set_output("overlay", p)

    ious = {}
    for k in obj_ids:
        a, b = sil == k, ids == k
        ious[f"obj_{k}"] = round(float((a & b).sum() / max((a | b).sum(), 1)), 4)
    ctx.metadata.update({
        "objects": len(obj_ids), "units": "metres" if metric else "relative",
        "fx": round(float(K[0, 0]), 2), "projection_iou": ious,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    })


if __name__ == "__main__":
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    raise SystemExit(main({"reconstruct": reconstruct}))
