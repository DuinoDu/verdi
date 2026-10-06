"""DINOv2 ViT-L/14 + 4 registers: frozen image features (CLS / patch / registers)
and region-pooled patch features, with the exact patch <-> input-pixel map.

Conventions:
* Continuous pixel coordinates, origin at the top-left CORNER of the top-left
  pixel (pixel (u, v) has its centre at (u + 0.5, v + 0.5)).
* Every frame is resized (no crop, no padding) to H x W, both multiples of 14:
  u_net = sx * u_in, v_net = sy * v_in with sx = W / w, sy = H / h (sx and sy may
  differ slightly because both sides are rounded to the patch stride).
* Patch (i, j) (row i, column j) covers u_in in [14 j / sx, 14 (j + 1) / sx),
  v_in in [14 i / sy, 14 (i + 1) / sy); its centre is ((14 j + 7) / sx,
  (14 i + 7) / sy). The whole input image is valid (no padded patches).
* Features are the final-LayerNorm tokens of forward_features
  (x_norm_clstoken / x_norm_regtokens / x_norm_patchtokens); not L2-normalised
  unless l2_normalize = true. This embedding space is NOT comparable with CLIP.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

STRIDE = 14
DIM = 1024
N_REG = 4
HUB_MODEL = "dinov2_vitl14_reg"  # hubconf name; 4 registers, checkpoint *_reg4_pretrain.pth
DINOV2_REPO = "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CKPT = "dinov2_vitl14_reg4_pretrain.pth"
CKPT_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"  # = manifest
MEAN = (0.485, 0.456, 0.406)  # ImageNet, as upstream dinov2/data/transforms.py
STD = (0.229, 0.224, 0.225)
EMBEDDING_SPACE = ("dinov2_vitl14_reg4 final-LayerNorm tokens (x_norm_*), checkpoint "
                   "dinov2_vitl14_reg4_pretrain.pth; not comparable with CLIP or other "
                   "DINOv2 variants")


# ------------------------------------------------------------------ model
def _load(ctx: Context):
    import torch

    repo = ctx.weight("dinov2_repo") / DINOV2_REPO
    ckpt = ctx.weight("dinov2_ckpt", prefetch=True) / DINOV2_CKPT
    if not (repo / "hubconf.py").exists() or not ckpt.exists():
        raise NodeError(f"pinned DINOv2 code / checkpoint missing ({repo}, {ckpt})",
                        kind="setup", hint="verdi setup dinov2_features")
    t0 = time.time()
    model = torch.hub.load(str(repo), HUB_MODEL, source="local", pretrained=False)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)  # raises on any key / shape mismatch
    dev = "cuda" if ctx.device == "cuda" else "cpu"
    # strict float32 on CUDA: torch enables TF32 for cuDNN convolutions by default,
    # which made the 14x14 patch embedding deviate from the CPU result (SC-02 GPU
    # regression 2026-10-06: patch-token cosine down to 0.9937 vs CPU)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.to(dev).eval().float()
    ctx.log(f"[dinov2] loaded {HUB_MODEL} on {dev} in {time.time() - t0:.1f}s")
    return model, dev


# ----------------------------------------------------------------- inputs
def _frames(ctx: Context):
    files = io.list_frames(ctx.input("frames"))
    if not files:
        raise NodeError("frames contains no images")
    imgs = [io.read_image(f) for f in files]
    h, w = imgs[0].shape[:2]
    for f, im in zip(files, imgs):
        if im.shape[:2] != (h, w):
            raise NodeError(f"frame {f.name} is {im.shape[1]}x{im.shape[0]} but frame 0 is "
                            f"{w}x{h}; one call = one resolution (one patch grid)",
                            hint="split the call per resolution")
        if float(im.std()) == 0.0:
            ctx.log(f"[dinov2] warning: frame {f.name} is constant (features are still computed)")
    return files, imgs, h, w


def _grid(h: int, w: int, max_side: int) -> Dict[str, Any]:
    if max_side < STRIDE or max_side % STRIDE:
        raise NodeError(f"max_side must be a positive multiple of {STRIDE}, got {max_side}")
    s = max_side / max(h, w)
    gw = max(1, int(round(w * s / STRIDE)))
    gh = max(1, int(round(h * s / STRIDE)))
    W, H = gw * STRIDE, gh * STRIDE
    return {"stride": STRIDE, "grid_hw": [gh, gw], "network_hw": [H, W], "input_hw": [h, w],
            "sx": W / w, "sy": H / h,
            "convention": "continuous pixel coords, origin at the top-left image corner; "
                          "u_net = sx * u_in, v_net = sy * v_in (resize only: no crop, no pad)",
            "patch_bounds_input": "patch (i, j): u_in in [14 j / sx, 14 (j+1) / sx), "
                                  "v_in in [14 i / sy, 14 (i+1) / sy)",
            "patch_center_input": "((14 j + 7) / sx, (14 i + 7) / sy)",
            "valid_region": "all patches (the whole input image is mapped; no padding)",
            "resize": "PIL bicubic (antialiased) on uint8 RGB"}


def read_regions(path: Path, n_frames: int, h: int, w: int) -> List[Dict[str, Any]]:
    """regions json -> list of {region_id, frame_index, object_id, bbox, mask(bool HxW|None)}."""
    data = io.read_json(path)
    regs = data.get("regions") if isinstance(data, dict) else None
    if not isinstance(regs, list) or not regs:
        raise NodeError("regions json needs a non-empty list 'regions'")
    out, seen = [], set()
    for k, r in enumerate(regs):
        rid = str(r.get("region_id", k))
        if rid in seen:
            raise NodeError(f"duplicate region_id {rid!r}")
        seen.add(rid)
        fi = r.get("frame_index")
        if not isinstance(fi, int) or not 0 <= fi < n_frames:
            raise NodeError(f"region {rid!r}: frame_index {fi!r} not in [0, {n_frames})")
        mask = None
        if r.get("mask"):
            mp = (Path(path).parent / r["mask"]).resolve()
            if not mp.exists():
                raise NodeError(f"region {rid!r}: mask {mp} not found")
            from PIL import Image

            mask = np.array(Image.open(mp)) > 0
            if mask.ndim == 3:
                mask = mask.any(-1)
            if mask.shape != (h, w):
                raise NodeError(f"region {rid!r}: mask is {mask.shape[1]}x{mask.shape[0]}, "
                                f"frames are {w}x{h}")
            if not mask.any():
                raise NodeError(f"region {rid!r}: mask is empty")
        bbox = r.get("bbox_xyxy")
        if bbox is None:
            if mask is None:
                raise NodeError(f"region {rid!r}: needs bbox_xyxy and/or mask")
            ys, xs = np.nonzero(mask)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        bbox = [float(b) for b in bbox]
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise NodeError(f"region {rid!r}: bbox_xyxy {bbox} is empty or outside the "
                            f"{w}x{h} frame (continuous coords, x1/y1 exclusive)")
        out.append({"region_id": rid, "frame_index": fi, "object_id": r.get("object_id"),
                    "bbox_xyxy": bbox, "mask": mask, "mask_file": r.get("mask")})
    return out


def _region_raster(r: Dict[str, Any], h: int, w: int) -> np.ndarray:
    """Fractional per-pixel area of the region (bbox ∩ mask)."""
    x0, y0, x1, y1 = r["bbox_xyxy"]
    xs = np.clip(np.minimum(np.arange(w) + 1, x1) - np.maximum(np.arange(w), x0), 0, 1)
    ys = np.clip(np.minimum(np.arange(h) + 1, y1) - np.maximum(np.arange(h), y0), 0, 1)
    R = ys[:, None] * xs[None, :]
    if r["mask"] is not None:
        R = R * r["mask"]
    return R.astype(np.float64)


def patch_coverage(R: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Exact area fraction of every patch cell covered by raster R (box integral)."""
    h, w = R.shape
    S = np.zeros((h + 1, w + 1))
    S[1:, 1:] = R.cumsum(0).cumsum(1)

    def integ(ys, xs):  # integral of R over [0, y) x [0, x) for continuous y, x
        y0 = np.floor(ys).astype(int).clip(0, h)
        x0 = np.floor(xs).astype(int).clip(0, w)
        fy, fx = ys - y0, xs - x0
        y1, x1 = (y0 + 1).clip(0, h), (x0 + 1).clip(0, w)
        A = S[y0][:, x0]
        B = S[y1][:, x0]
        C = S[y0][:, x1]
        D = S[y1][:, x1]
        return (A * (1 - fy)[:, None] * (1 - fx)[None] + B * fy[:, None] * (1 - fx)[None]
                + C * (1 - fy)[:, None] * fx[None] + D * fy[:, None] * fx[None])

    ye = np.linspace(0, h, gh + 1)
    xe = np.linspace(0, w, gw + 1)
    I = integ(ye, xe)
    cell = I[1:, 1:] - I[:-1, 1:] - I[1:, :-1] + I[:-1, :-1]
    area = (h / gh) * (w / gw)
    return np.clip(cell / area, 0.0, 1.0)


# ------------------------------------------------------------------- task
def extract(ctx: Context) -> None:
    import torch
    from PIL import Image

    files, imgs, h, w = _frames(ctx)
    max_side = int(ctx.param("max_side", 518))
    grid = _grid(h, w, max_side)
    gh, gw = grid["grid_hw"]
    H, W = grid["network_hw"]
    l2 = bool(ctx.param("l2_normalize", False))
    bs = max(1, int(ctx.param("batch_size", 4)))
    patch_dtype = str(ctx.param("patch_dtype", "float32"))
    min_cov = float(ctx.param("min_coverage", 0.5))
    if not 0.0 < min_cov <= 1.0:
        raise NodeError(f"min_coverage must be in (0, 1], got {min_cov}")
    regions = read_regions(ctx.input("regions"), len(files), h, w) \
        if ctx.has_input("regions") else []

    model, dev = _load(ctx)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    F = len(files)
    cls = np.zeros((F, DIM), np.float32)
    reg = np.zeros((F, N_REG, DIM), np.float32)
    patch = np.zeros((F, gh, gw, DIM), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, F, bs):
            batch = [np.asarray(Image.fromarray(im).resize((W, H), Image.BICUBIC))
                     for im in imgs[s:s + bs]]
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - mean) / std).to(dev)
            o = model.forward_features(x)
            n = x.shape[0]
            pt = o["x_norm_patchtokens"]
            if pt.shape[1] != gh * gw or o["x_norm_regtokens"].shape[1] != N_REG:
                raise NodeError(f"unexpected token layout {tuple(pt.shape)} for grid {gh}x{gw}")
            cls[s:s + n] = o["x_norm_clstoken"].float().cpu().numpy()
            reg[s:s + n] = o["x_norm_regtokens"].float().cpu().numpy()
            patch[s:s + n] = pt.float().cpu().numpy().reshape(n, gh, gw, DIM)
    infer = time.time() - t0
    if not (np.isfinite(cls).all() and np.isfinite(patch).all()):
        raise NodeError("non-finite features (model / input problem)")

    def norm(a):
        return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)

    if l2:
        cls, reg, patch = norm(cls), norm(reg), norm(patch)

    # region pooling: coverage-weighted mean of the (possibly normalised) patch tokens
    reg_feats = np.zeros((len(regions), DIM), np.float32)
    reg_info = []
    for k, r in enumerate(regions):
        cov = patch_coverage(_region_raster(r, h, w), gh, gw)
        sel = cov >= min_cov
        rec = {"index": k, "region_id": r["region_id"], "frame_index": r["frame_index"],
               "frame": files[r["frame_index"]].name, "object_id": r["object_id"],
               "bbox_xyxy": r["bbox_xyxy"], "mask_file": r["mask_file"],
               "region_area_px": float(_region_raster(r, h, w).sum()),
               "n_patches": int(sel.sum()), "min_coverage": min_cov}
        if sel.any():
            wts = cov[sel]
            v = (patch[r["frame_index"]][sel] * wts[:, None]).sum(0) / wts.sum()
            reg_feats[k] = norm(v) if l2 else v
            rec["valid"] = True
            rec["patch_ij"] = [[int(i), int(j)] for i, j in zip(*np.nonzero(sel))]
        else:
            rec["valid"] = False
            rec["reason"] = (f"no patch covered >= {min_cov:.2f} by the region (region smaller "
                             f"than a patch cell); feature row is zeros")
        reg_info.append(rec)

    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    centers = np.stack([(STRIDE * jj + 7) / grid["sx"], (STRIDE * ii + 7) / grid["sy"]], -1)

    np.save(ctx.output_path("cls", "cls.npy"), cls)
    np.save(ctx.output_path("registers", "registers.npy"), reg)
    pp = ctx.output_path("patch", "patch.npy")
    np.save(pp, patch.astype(np.float16 if patch_dtype == "float16" else np.float32))
    np.save(ctx.output_path("patch_centers", "patch_centers.npy"), centers.astype(np.float64))
    for name, path in (("cls", "cls.npy"), ("registers", "registers.npy"),
                       ("patch", "patch.npy"), ("patch_centers", "patch_centers.npy")):
        ctx.set_output(name, ctx.output_dir / path)
    if regions:
        np.save(ctx.output_path("region_features", "region_features.npy"), reg_feats)
        ctx.set_output("region_features", ctx.output_dir / "region_features.npy")

    info = {
        "model": HUB_MODEL, "checkpoint": DINOV2_CKPT, "checkpoint_sha256": CKPT_SHA256,
        "code": DINOV2_REPO, "embedding_space": EMBEDDING_SPACE,
        "layer": "final LayerNorm output of forward_features (x_norm_clstoken, "
                 "x_norm_regtokens, x_norm_patchtokens); registers kept separate, never "
                 "mixed into patch / region features",
        "normalize": {"input": {"mean": MEAN, "std": STD, "scale": "uint8 / 255"},
                      "l2_normalize": l2},
        "dtype": "float32 compute (TF32 disabled for cuDNN and matmul)", "patch_dtype": patch_dtype,
        "device": dev,
        "shapes": {"cls": list(cls.shape), "registers": list(reg.shape),
                   "patch": list(patch.shape), "patch_centers": list(centers.shape),
                   "region_features": [len(regions), DIM]},
        "pixel_map": grid,
        "frames": [{"frame_index": i, "frame": f.name} for i, f in enumerate(files)],
        "regions": reg_info,
        "region_pooling": "coverage-weighted mean of patch tokens whose cell is covered >= "
                          "min_coverage by (bbox ∩ mask) (exact area fractions); invalid "
                          "regions are reported (valid=false), not dropped",
        "params": {"max_side": max_side, "batch_size": bs, "l2_normalize": l2,
                   "patch_dtype": patch_dtype, "min_coverage": min_cov},
        "infer_sec": round(infer, 3),
        "note": "frozen pretrained features; no task-level quality evaluation is implied",
    }
    io.write_json(ctx.output_path("info", "info.json"), info)
    ctx.set_output("info", ctx.output_dir / "info.json")


if __name__ == "__main__":
    raise SystemExit(main({"extract": extract}))
