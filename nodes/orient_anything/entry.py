"""Orient Anything node: estimate_orientation."""
from __future__ import annotations

import os
import sys

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

CKPT = "ronormsigma1/dino_weight.pt"   # upstream app.py default checkpoint
OUT_DIM = 360 + 180 + 360 + 2           # azimuth, polar, rotation bins + conf

# screen frame S: x right, y up, z towards the viewer -> OpenCV camera frame
S_TO_CV = np.diag([1.0, -1.0, -1.0])
# object axes (front, left, up) expressed in S at zero angles
OBJ_IN_S = np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]])


def rotation_cam_obj(azimuth_deg: float, polar_deg: float,
                     rotation_deg: float) -> np.ndarray:
    """R_cam_obj (OpenCV) consistent with upstream get_proj2D_XYZ."""
    phi, theta = np.radians(azimuth_deg), np.radians(polar_deg)
    gamma = np.radians(-rotation_deg)  # upstream draws with -rotation
    c, s = np.cos, np.sin
    ry = np.array([[c(phi), 0, -s(phi)], [0, 1, 0], [s(phi), 0, c(phi)]])
    rx = np.array([[1, 0, 0], [0, c(theta), -s(theta)],
                   [0, s(theta), c(theta)]])
    rr = np.array([[c(gamma), s(gamma), 0], [-s(gamma), c(gamma), 0],
                   [0, 0, 1]])
    return S_TO_CV @ rr @ rx @ ry @ OBJ_IN_S


def _load(ctx: Context):
    import torch

    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)  # utils.py loads ./assets/axis.obj at import time
    try:
        import paths

        paths.DINO_LARGE = str(ctx.weight("dinov2_large"))
        from inference import get_3angle, get_3angle_infer_aug
        from transformers import AutoImageProcessor
        from utils import background_preprocess
        from vision_tower import DINOv2_MLP
    finally:
        os.chdir(cwd)
    dino = DINOv2_MLP(dino_mode="large", in_dim=1024, out_dim=OUT_DIM,
                      evaluate=True, mask_dino=False, frozen_back=False)
    dino.load_state_dict(torch.load(ctx.weight("orient") / CKPT,
                                    map_location="cpu"))
    dino = dino.to(ctx.device).eval()
    pre = AutoImageProcessor.from_pretrained(paths.DINO_LARGE)
    return dino, pre, get_3angle, get_3angle_infer_aug, background_preprocess


def _crops(ctx: Context, image):
    if not ctx.has_input("boxes"):
        return [(image, None, None)]
    boxes = io.read_json(ctx.input("boxes")).get("boxes", [])
    if not boxes:
        raise NodeError("bbox_set has no boxes", kind="request")
    pad = float(ctx.param("pad", 0.1))
    w, h = image.size
    out = []
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b["xyxy"]]
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        box = [max(0, x0 - px), max(0, y0 - py), min(w, x1 + px),
               min(h, y1 + py)]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            raise NodeError(f"box {b['xyxy']} is empty inside the image",
                            kind="request")
        out.append((image.crop(tuple(int(round(v)) for v in box)),
                    b["xyxy"], b.get("label")))
    return out


def estimate_orientation(ctx: Context) -> None:
    import random

    import torch
    from PIL import Image

    image = Image.open(ctx.input("image")).convert("RGB")
    dino, pre, get_3angle, get_3angle_aug, bg = _load(ctx)
    rm_bkg = bool(ctx.param("remove_background", False))
    tta = bool(ctx.param("test_time_aug", False))
    objects = []
    for crop, box, label in _crops(ctx, image):
        random.seed(0)
        torch.manual_seed(0)
        if tta:  # upstream: always removes background for the aug half
            angles = get_3angle_aug(crop, bg(crop, True), dino, pre,
                                    ctx.device)
        else:
            angles = get_3angle(bg(crop, rm_bkg), dino, pre, ctx.device)
        az, pol, rot, conf = (float(a) for a in angles)
        if not all(np.isfinite([az, pol, rot, conf])):
            raise NodeError("Orient Anything produced non-finite angles")
        obj = {"azimuth_deg": round(az % 360.0, 3),
               "polar_deg": round(pol, 3),
               "rotation_deg": round(rot, 3),
               "confidence": round(conf, 4),
               "R_cam_obj": np.round(rotation_cam_obj(az, pol, rot),
                                     6).tolist()}
        if box is not None:
            obj["box"] = box
        if label is not None:
            obj["label"] = label
        objects.append(obj)
        ctx.log(f"object {len(objects)}: az {az:.1f} polar {pol:.1f} "
                f"rot {rot:.1f} conf {conf:.3f}")
    out = io.write_json(ctx.output_path("orientation", "orientation.json"),
                        {"objects": objects})
    ctx.set_output("orientation", out)


if __name__ == "__main__":
    raise SystemExit(main({"estimate_orientation": estimate_orientation}))
