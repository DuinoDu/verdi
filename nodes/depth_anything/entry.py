"""Depth Anything V2 + Video-Depth-Anything node."""
from __future__ import annotations

import sys

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

VDA_VITL = {"encoder": "vitl", "features": 256,
            "out_channels": [256, 512, 1024, 1024]}


# ------------------------------------------------------------------ images
def _hf_predict(ctx: Context, key: str) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    path = str(ctx.weight(key))
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForDepthEstimation.from_pretrained(path).to(
        ctx.device).eval()
    image = Image.open(ctx.input("image")).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(ctx.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)])[0]
    pred = post["predicted_depth"].float().cpu().numpy()
    if not np.isfinite(pred).all():
        raise NodeError("model produced non-finite values")
    ctx.metadata["model"] = model.config.depth_estimation_type \
        if hasattr(model.config, "depth_estimation_type") else "relative"
    return pred.astype(np.float32)


def estimate_depth(ctx: Context) -> None:
    key = "metric_" + ctx.param("scene", "indoor")
    depth = _hf_predict(ctx, key)
    out = io.write_depth(ctx.output_path("depth", "depth.npy"), depth)
    ctx.set_output("depth", out)
    ctx.metadata["checkpoint"] = key


def estimate_relative_depth(ctx: Context) -> None:
    disp = _hf_predict(ctx, "relative")
    out = ctx.output_path("disparity", "disparity.npy")
    np.save(out, np.clip(disp, 0, None).astype(np.float32))
    ctx.set_output("disparity", out)


# ------------------------------------------------------------------- video
def _video_predict(ctx: Context, metric: bool):
    import cv2
    import torch

    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from video_depth_anything.video_depth import VideoDepthAnything

    files = io.list_frames(ctx.input("frames"))
    if not files:
        raise NodeError("no frames found", kind="request")
    frames = [io.read_image(f) for f in files]
    h, w = frames[0].shape[:2]
    if any(f.shape[:2] != (h, w) for f in frames):
        raise NodeError("frames differ in size", kind="request")
    max_res = int(ctx.param("max_res", 1280))
    if max(h, w) > max_res:  # same rule as upstream read_video_frames
        s = max_res / max(h, w)
        rh, rw = round(h * s), round(w * s)
        rh -= rh % 2
        rw -= rw % 2
        frames = [cv2.resize(f, (rw, rh), interpolation=cv2.INTER_AREA)
                  for f in frames]

    key = "video_metric" if metric else "video_relative"
    fname = ("metric_" if metric else "") + "video_depth_anything_vitl.pth"
    model = VideoDepthAnything(**VDA_VITL, metric=metric)
    model.load_state_dict(torch.load(ctx.weight(key) / fname,
                                     map_location="cpu"), strict=True)
    model = model.to(ctx.device).eval()
    fps = 30.0
    depths, _ = model.infer_video_depth(
        np.stack(frames), fps, input_size=518, device=ctx.device,
        fp32=bool(ctx.param("fp32", False)) or ctx.device == "cpu")
    out = []
    for d in depths[:len(files)]:
        d = np.asarray(d, np.float32)
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(d).all():
            raise NodeError("model produced non-finite depth",
                            hint="retry with -p fp32=true")
        out.append(d)
    if len(out) != len(files):
        raise NodeError(f"got {len(out)} depth maps for {len(files)} frames")
    ctx.metadata["checkpoint"] = key
    return out


def estimate_depth_seq(ctx: Context) -> None:
    out_dir = ctx.output_path("depth")
    for i, d in enumerate(_video_predict(ctx, metric=True)):
        io.write_depth(out_dir / io.frame_name(i, ".npy"), d)
    ctx.set_output("depth", out_dir)


def estimate_relative_depth_seq(ctx: Context) -> None:
    out_dir = ctx.output_path("disparity")
    for i, d in enumerate(_video_predict(ctx, metric=False)):
        np.save(out_dir / io.frame_name(i, ".npy"), np.clip(d, 0, None))
    ctx.set_output("disparity", out_dir)


if __name__ == "__main__":
    raise SystemExit(main({
        "estimate_depth": estimate_depth,
        "estimate_relative_depth": estimate_relative_depth,
        "estimate_depth_seq": estimate_depth_seq,
        "estimate_relative_depth_seq": estimate_relative_depth_seq,
    }))
