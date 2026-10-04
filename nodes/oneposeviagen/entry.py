"""OnePoseViaGen 3D generation node: generate (Hi3DGen_Color) / generate_amodal (Amodal3R)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main

DINOV2_REPO = "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CKPT = "dinov2_vitl14_reg4_pretrain.pth"


# ------------------------------------------------------------------ setup
def _no_fa3_off_hopper() -> None:
    """xformers 0.0.32 enables its FlashAttention-3 kernels (Hopper sm_90a
    only) for any compute capability >= 9.0, so on Blackwell (sm_120) the
    first attention call aborts with "CUDA error ... invalid argument".
    Turn FA3 off there; xformers then dispatches to FA2 / cutlass."""
    import torch
    from xformers.ops.fmha import dispatch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0):
        dispatch._set_use_fa3(False)


def _prepare(ctx: Context) -> None:
    """sys.path like upstream app.py + a torch.hub dir with pinned DINOv2."""
    if ctx.device != "cuda":
        raise NodeError("OnePoseViaGen generation needs a CUDA GPU")
    root = ctx.repo / "oneposeviagen"
    for p in (root / "trellis", root / "Amodal3R"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import torch

    _no_fa3_off_hopper()
    hub = Path(tempfile.mkdtemp(prefix="opv_hub_"))
    (hub / "checkpoints").mkdir()
    # upstream: torch.hub.load(<hub_dir>/facebookresearch_dinov2_main, source='local')
    (hub / "facebookresearch_dinov2_main").symlink_to(
        ctx.weight("dinov2_repo") / DINOV2_REPO)
    (hub / "checkpoints" / DINOV2_CKPT).symlink_to(
        ctx.weight("dinov2_ckpt", prefetch=True) / DINOV2_CKPT)
    torch.hub.set_dir(str(hub))


def _read_mask(path: Path, mask_id: int = 0) -> np.ndarray:
    from PIL import Image

    ids = np.asarray(Image.open(path))
    if ids.ndim == 3:
        ids = ids[..., 0]
    return ids == mask_id if mask_id > 0 else ids > 0


def _image_and_mask(ctx: Context):
    """RGB uint8 HxWx3 and boolean object mask."""
    from PIL import Image

    img = Image.open(ctx.input("image"))
    if ctx.has_input("mask"):
        rgb = np.asarray(img.convert("RGB"))
        fg = _read_mask(ctx.input("mask"), int(ctx.param("mask_id", 0)))
    else:
        rgba = np.asarray(img.convert("RGBA"))
        rgb, fg = rgba[..., :3], rgba[..., 3] > 127
        if fg.all():
            raise NodeError("no mask given and the image has no transparent "
                            "background", hint="pass `-i mask=...` (e.g. from "
                            "sam2 / langsam) or an RGBA cut-out")
    if fg.shape != rgb.shape[:2]:
        raise NodeError(f"mask size {fg.shape[::-1]} != image size "
                        f"{rgb.shape[1::-1]}")
    if fg.sum() < 64:
        raise NodeError("object mask is empty", hint="check mask / mask_id")
    return rgb, fg


def _sampler(ctx: Context):
    ss = {"steps": int(ctx.param("ss_steps", 12)),
          "cfg_strength": float(ctx.param("ss_guidance", 7.5))}
    slat = {"steps": int(ctx.param("slat_steps", 25)),
            "cfg_strength": float(ctx.param("slat_guidance", 15.0))}
    return ss, slat


def _save(ctx: Context, mesh, outputs) -> None:
    out = ctx.output_path("mesh", "mesh.glb")
    mesh.export(str(out))
    ctx.set_output("mesh", out)
    v = np.asarray(mesh.vertices)
    import torch

    ctx.metadata.update({
        "vertices": int(len(v)), "faces": int(len(mesh.faces)),
        "bbox_min": v.min(0).round(4).tolist(),
        "bbox_max": v.max(0).round(4).tolist(),
        "raw_mesh_faces": int(outputs["mesh"][0].faces.shape[0]),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    })


# ------------------------------------------------------------------ tasks
def generate(ctx: Context) -> None:
    import torch
    from PIL import Image

    rgb, fg = _image_and_mask(ctx)
    _prepare(ctx)
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipe = TrellisImageTo3DPipeline.from_pretrained(
        str(ctx.weight("onepose", prefetch=True) / "Hi3DGen_Color"))
    pipe.cuda()
    # RGBA input takes upstream preprocess_image's alpha branch (1.2x crop of
    # the alpha box, pad to square, 518 px, premultiplied) - no BiRefNet
    rgba = Image.fromarray(np.dstack([rgb, fg.astype(np.uint8) * 255]), "RGBA")
    ss, slat = _sampler(ctx)
    with torch.no_grad():
        cond_img = pipe.preprocess_image(rgba)
        cond_img.save(ctx.output_dir / "input_crop.png")
        outputs = pipe.run(cond_img, seed=int(ctx.param("seed", 42)),
                           formats=["mesh", "gaussian"], preprocess_image=False,
                           sparse_structure_sampler_params=ss,
                           slat_sampler_params=slat)
    if outputs["mesh"][0].faces.shape[0] == 0:
        raise NodeError("Hi3DGen produced an empty mesh",
                        hint="check the mask; try another seed")
    mesh = postprocessing_utils.to_trimesh(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=float(ctx.param("simplify", 0.95)),
        texture_size=int(ctx.param("texture_size", 1024)), verbose=False)
    _save(ctx, mesh, outputs)


def _occluders_from_depth(depth: np.ndarray, obj: np.ndarray,
                          area_threshold: int = 100, max_iter: int = 50):
    """Port of OnePoseViaGen app.py generate_final_mask (occluder part)."""
    import cv2

    valid = depth > 0
    if not (obj & valid).any():
        raise NodeError("depth has no valid pixels on the object")
    ref = np.percentile(depth[obj & valid], 90)
    occ = (depth < ref) & valid & ~obj
    kernel = np.ones((3, 3), np.uint8)
    occ = cv2.erode(occ.astype(np.uint8), kernel, iterations=1)
    x, y, w, h = cv2.boundingRect(obj.astype(np.uint8))
    box = np.zeros_like(occ)
    box[y:y + h, x:x + w] = occ[y:y + h, x:x + w]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(box, connectivity=8)
    filt = np.zeros_like(box)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            filt[labels == i] = 1
    dil = filt.copy()
    for _ in range(max_iter):
        if ((dil > 0) & obj).any():
            break
        dil = cv2.dilate(dil, kernel, iterations=1)
    gap = (dil > 0) & ~filt.astype(bool) & ~obj
    filt[gap] = 1
    return filt > 0


def _square_crop(rgb, final, obj, occ):
    """Square crop (1.2x box of object + nearby occluders), padded."""
    ys, xs = np.nonzero(obj)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # occluders count only near the object (box enlarged 1.5x)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * 0.75 + 1, (y1 - y0) * 0.75 + 1
    near = np.zeros_like(occ)
    near[max(0, int(cy - hh)):int(cy + hh) + 1,
         max(0, int(cx - hw)):int(cx + hw) + 1] = True
    oy, ox = np.nonzero(occ & near)
    if len(ox):
        x0, x1 = min(x0, ox.min()), max(x1, ox.max())
        y0, y1 = min(y0, oy.min()), max(y1, oy.max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = int(max(x1 - x0, y1 - y0) * 1.2 / 2) + 1
    X0, Y0 = int(cx) - half, int(cy) - half
    size = 2 * half
    H, W = obj.shape
    img_c = np.zeros((size, size, 3), np.uint8)
    fin_c = np.full((size, size), 255, np.uint8)
    sx0, sy0 = max(0, X0), max(0, Y0)
    sx1, sy1 = min(W, X0 + size), min(H, Y0 + size)
    img_c[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = rgb[sy0:sy1, sx0:sx1]
    fin_c[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = final[sy0:sy1, sx0:sx1]
    return img_c, fin_c


def generate_amodal(ctx: Context) -> None:
    import torch
    from PIL import Image

    rgb, obj = _image_and_mask(ctx)
    if ctx.has_input("occluder"):
        occ = _read_mask(ctx.input("occluder"))
        if occ.shape != obj.shape:
            raise NodeError("occluder mask size differs from the image")
        occ &= ~obj
    elif ctx.has_input("depth"):
        depth = np.load(ctx.input("depth")).astype(np.float32)
        if depth.shape != obj.shape:
            raise NodeError("depth size differs from the image")
        occ = _occluders_from_depth(depth, obj)
    else:
        raise NodeError("generate_amodal needs `occluder` or `depth`",
                        hint="for fully visible objects use task `generate`")
    # OnePoseViaGen condition mask: 255 background, 188 object, 0 occluder
    final = np.full(obj.shape, 255, np.uint8)
    final[obj] = 188
    final[occ] = 0
    img_c, fin_c = _square_crop(rgb, final, obj, occ)
    ctx.metadata["occluder_pixels"] = int(occ.sum())

    _prepare(ctx)
    from amodal3r.pipelines import Amodal3RImageTo3DPipeline
    from amodal3r.utils import postprocessing_utils

    pipe = Amodal3RImageTo3DPipeline.from_pretrained(
        str(ctx.weight("onepose", prefetch=True) / "Amodal3R"))
    pipe.cuda()
    ss, slat = _sampler(ctx)
    fin_img = Image.fromarray(fin_c, "L")
    fpath = ctx.output_path("final_mask", "final_mask.png")
    fin_img.resize((518, 518), Image.NEAREST).save(fpath)
    ctx.set_output("final_mask", fpath)
    Image.fromarray(img_c).resize((518, 518), Image.LANCZOS).save(
        ctx.output_dir / "input_crop.png")
    with torch.no_grad():
        outputs = pipe.run_multi_image(
            [Image.fromarray(img_c)], [fin_img],
            seed=int(ctx.param("seed", 42)), formats=["mesh", "gaussian"],
            sparse_structure_sampler_params=ss, slat_sampler_params=slat)
    if outputs["mesh"][0].faces.shape[0] == 0:
        raise NodeError("Amodal3R produced an empty mesh",
                        hint="check masks; try another seed")
    mesh = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=float(ctx.param("simplify", 0.95)),
        texture_size=int(ctx.param("texture_size", 1024)), verbose=False)
    _save(ctx, mesh, outputs)


if __name__ == "__main__":
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    raise SystemExit(main({"generate": generate,
                           "generate_amodal": generate_amodal}))
