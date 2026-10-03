"""SAM 2.1 node: segment_image / segment_video."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml",
                  "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def _ckpt(ctx: Context):
    name = ctx.param("checkpoint", "large")
    cfg, fname = CONFIGS[name]
    return cfg, str(ctx.weight(name) / fname)


def _prompts_by_obj(ctx: Context, frame_aware: bool):
    """{(frame, obj_id): {"points": [...], "labels": [...], "box": ...}}"""
    data = io.read_json(ctx.input("prompts"))
    grouped = defaultdict(lambda: {"points": [], "labels": [], "box": None})
    for p in data.get("points", []):
        key = (int(p.get("frame", 0)) if frame_aware else 0,
               int(p.get("obj_id", 1)))
        grouped[key]["points"].append(p["xy"])
        grouped[key]["labels"].append(1 if p.get("positive", True) else 0)
    for b in data.get("boxes", []):
        key = (int(b.get("frame", 0)) if frame_aware else 0,
               int(b.get("obj_id", 1)))
        grouped[key]["box"] = b["xyxy"]
    if not grouped:
        raise NodeError("prompt set has no points or boxes",
                        hint="SAM2 needs point or box prompts; use "
                        "groundingdino/langsam for text prompts")
    return grouped


def segment_image(ctx: Context) -> None:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg, ckpt = _ckpt(ctx)
    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=ctx.device))
    image = io.read_image(ctx.input("image"))
    ids = np.zeros(image.shape[:2], np.uint16)
    labels = {}
    with torch.inference_mode(), torch.autocast(
            ctx.device, dtype=torch.bfloat16, enabled=ctx.device == "cuda"):
        predictor.set_image(image)
        for (_, obj_id), pr in sorted(_prompts_by_obj(ctx, False).items()):
            masks, scores, _ = predictor.predict(
                point_coords=np.array(pr["points"], np.float32)
                if pr["points"] else None,
                point_labels=np.array(pr["labels"], np.int32)
                if pr["points"] else None,
                box=np.array(pr["box"], np.float32)
                if pr["box"] is not None else None,
                multimask_output=bool(ctx.param("multimask")),
            )
            best = int(np.argmax(scores))
            ids[masks[best] > 0] = obj_id
            labels[obj_id] = {"score": float(scores[best])}
    out = ctx.output_path("mask", "mask.png")
    io.write_mask(out, ids, labels)
    ctx.set_output("mask", out)


def segment_video(ctx: Context) -> None:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    cfg, ckpt = _ckpt(ctx)
    frames_dir = ctx.input("frames")
    frames = io.list_frames(frames_dir)
    # SAM2 video loader wants a dir of <int>.jpg; convert when needed
    if any(f.suffix.lower() not in (".jpg", ".jpeg") for f in frames):
        jpg_dir = ctx.output_path("_jpg_frames")
        from PIL import Image

        for i, f in enumerate(frames):
            Image.open(f).convert("RGB").save(jpg_dir / f"{i:06d}.jpg",
                                              quality=95)
        frames_dir = jpg_dir
    predictor = build_sam2_video_predictor(cfg, ckpt, device=ctx.device)
    h, w = io.read_image(frames[0]).shape[:2]
    per_frame = [np.zeros((h, w), np.uint16) for _ in frames]
    with torch.inference_mode(), torch.autocast(
            ctx.device, dtype=torch.bfloat16, enabled=ctx.device == "cuda"):
        state = predictor.init_state(video_path=str(frames_dir))
        for (frame, obj_id), pr in sorted(_prompts_by_obj(ctx, True).items()):
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=frame, obj_id=obj_id,
                points=np.array(pr["points"], np.float32)
                if pr["points"] else None,
                labels=np.array(pr["labels"], np.int32)
                if pr["points"] else None,
                box=np.array(pr["box"], np.float32)
                if pr["box"] is not None else None,
            )
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            for k, obj_id in enumerate(obj_ids):
                m = (logits[k] > 0.0).cpu().numpy().squeeze()
                per_frame[fidx][m] = obj_id
    out_dir = ctx.output_path("masks")
    for i, ids in enumerate(per_frame):
        io.write_mask(out_dir / io.frame_name(i), ids)
    ctx.set_output("masks", out_dir)


if __name__ == "__main__":
    raise SystemExit(main({"segment_image": segment_image,
                           "segment_video": segment_video}))
