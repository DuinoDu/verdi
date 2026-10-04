"""Apple Depth Pro node: estimate_depth / estimate_depth_seq."""
from __future__ import annotations

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


def _load(ctx: Context):
    import torch
    from transformers import (DepthProForDepthEstimation,
                              DepthProImageProcessorFast)

    path = str(ctx.weight("depthpro"))
    dtype = torch.float16 if (ctx.device == "cuda" and
                              ctx.param("precision", "fp16") == "fp16") \
        else torch.float32
    processor = DepthProImageProcessorFast.from_pretrained(path)
    model = DepthProForDepthEstimation.from_pretrained(
        path, torch_dtype=dtype).to(ctx.device).eval()
    return model, processor, dtype


def _infer(model, processor, dtype, device, rgb: np.ndarray):
    """Return (depth_m HxW float32 at estimated focal, focal_px)."""
    import torch
    from PIL import Image

    h, w = rgb.shape[:2]
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype)
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)
    outputs.predicted_depth = outputs.predicted_depth.float()
    if outputs.field_of_view is None:
        raise NodeError("Depth Pro returned no field of view",
                        hint="checkpoint without FOV head; use the pinned "
                        "apple/DepthPro-hf weights")
    outputs.field_of_view = outputs.field_of_view.float()
    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(h, w)])[0]
    depth = post["predicted_depth"].float().cpu().numpy()
    focal = float(post["focal_length"].float().cpu().item())
    if not np.isfinite(depth).all() or not np.isfinite(focal) or focal <= 0:
        raise NodeError("Depth Pro produced non-finite depth/focal",
                        hint="retry with -p precision=fp32")
    return depth.astype(np.float32), focal


def _known_fx(ctx: Context, w: int, h: int):
    if not ctx.has_input("camera"):
        return None, None
    cam = io.read_camera(ctx.input("camera"))
    if int(cam["width"]) != w or int(cam["height"]) != h:
        raise NodeError(
            f"camera is {cam['width']}x{cam['height']} but image is {w}x{h}",
            hint="scale K to the image resolution first", kind="request")
    return float(cam["K"][0, 0]), cam


def _center_K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])


def estimate_depth(ctx: Context) -> None:
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    model, processor, dtype = _load(ctx)
    depth, f_est = _infer(model, processor, dtype, ctx.device, rgb)
    fx, cam = _known_fx(ctx, w, h)
    if fx is not None:
        # depth is proportional to the focal length (depth = f / (W*invd))
        depth = depth * (fx / f_est)
        K, dist = cam["K"], cam["dist"]
    else:
        K, dist = _center_K(f_est, w, h), None
    ctx.metadata.update({"estimated_focal_px": f_est,
                         "used_focal_px": fx or f_est,
                         "estimated_hfov_deg": float(np.degrees(
                             2 * np.arctan(0.5 * w / f_est)))})
    out = io.write_depth(ctx.output_path("depth", "depth.npy"), depth)
    ctx.set_output("depth", out)
    cam_out = io.write_camera(ctx.output_path("camera", "camera.json"), K,
                              w, h, None if dist is None else list(dist))
    ctx.set_output("camera", cam_out)


def estimate_depth_seq(ctx: Context) -> None:
    frames = io.list_frames(ctx.input("frames"))
    if not frames:
        raise NodeError("no frames found", kind="request")
    model, processor, dtype = _load(ctx)
    depths, focals = [], []
    size = None
    for i, f in enumerate(frames):
        rgb = io.read_image(f)
        if size is None:
            size = rgb.shape[:2]
        elif rgb.shape[:2] != size:
            raise NodeError(f"frame {f.name} has size {rgb.shape[:2]} != "
                            f"{size}", kind="request")
        d, fe = _infer(model, processor, dtype, ctx.device, rgb)
        depths.append(d)
        focals.append(fe)
        ctx.log(f"frame {i}: focal {fe:.1f}px")
    h, w = size
    fx, cam = _known_fx(ctx, w, h)
    f_med = float(np.median(focals))
    mode = ctx.param("focal", "median")
    if fx is not None:
        targets = [fx] * len(depths)
        K, dist = cam["K"], list(cam["dist"])
    elif mode == "median":
        targets = [f_med] * len(depths)
        K, dist = _center_K(f_med, w, h), None
    else:
        targets = focals
        K, dist = _center_K(f_med, w, h), None
    out_dir = ctx.output_path("depth")
    for i, (d, fe, ft) in enumerate(zip(depths, focals, targets)):
        io.write_depth(out_dir / io.frame_name(i, ".npy"), d * (ft / fe))
    ctx.set_output("depth", out_dir)
    ctx.set_output("camera", io.write_camera(
        ctx.output_path("camera", "camera.json"), K, w, h, dist))
    ctx.metadata.update({"estimated_focal_px": focals,
                         "median_focal_px": f_med,
                         "used_focal_px": targets})


if __name__ == "__main__":
    raise SystemExit(main({"estimate_depth": estimate_depth,
                           "estimate_depth_seq": estimate_depth_seq}))
