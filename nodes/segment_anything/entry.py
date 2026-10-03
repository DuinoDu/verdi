"""Segment Anything v1 node (HF transformers SamModel): segment_image / auto_masks."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def _model_dir(ctx: Context):
    return ctx.weight(ctx.param("checkpoint", "huge"))


def _group_prompts(ctx: Context):
    data = io.read_json(ctx.input("prompts"))
    grouped = defaultdict(lambda: {"points": [], "labels": [], "box": None})
    for p in data.get("points", []):
        g = grouped[int(p.get("obj_id", 1))]
        g["points"].append([float(v) for v in p["xy"]])
        g["labels"].append(1 if p.get("positive", True) else 0)
    for b in data.get("boxes", []):
        grouped[int(b.get("obj_id", 1))]["box"] = [float(v) for v in b["xyxy"]]
    if not grouped:
        raise NodeError("prompt set has no points or boxes",
                        hint="SAM needs point or box prompts; use the "
                        "groundingdino or langsam node for text prompts",
                        kind="request")
    if any(k < 1 for k in grouped):
        raise NodeError("obj_id must be >= 1 (0 is background)",
                        kind="request")
    return dict(sorted(grouped.items()))


def segment_image(ctx: Context) -> None:
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    path = _model_dir(ctx)
    processor = SamProcessor.from_pretrained(path)
    model = SamModel.from_pretrained(path).to(ctx.device).eval()
    image = Image.fromarray(io.read_image(ctx.input("image")))
    prompts = _group_prompts(ctx)
    multimask = bool(ctx.param("multimask", False))

    base = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        emb = model.get_image_embeddings(base["pixel_values"].to(ctx.device))
    ids = np.zeros((image.height, image.width), np.uint16)
    labels = {}
    for obj_id, pr in prompts.items():
        kw = {}
        if pr["points"]:
            kw["input_points"] = [[pr["points"]]]
            kw["input_labels"] = [[pr["labels"]]]
        if pr["box"] is not None:
            kw["input_boxes"] = [[pr["box"]]]
        inp = processor(images=image, return_tensors="pt", **kw)
        inp.pop("pixel_values")
        feed = {k: v.to(ctx.device) for k, v in inp.items()
                if k in ("input_points", "input_labels", "input_boxes")}
        with torch.inference_mode():
            out = model(image_embeddings=emb, multimask_output=multimask,
                        **feed)
        masks = processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"],
            inp["reshaped_input_sizes"])[0][0]  # [K,H,W] bool
        scores = out.iou_scores[0, 0].float().cpu().numpy()
        best = int(np.argmax(scores))
        ids[masks[best].numpy()] = obj_id
        labels[obj_id] = {"score": round(float(scores[best]), 4)}
        ctx.log(f"obj {obj_id}: score {scores[best]:.3f} "
                f"area {int(masks[best].sum())}")
    out_path = ctx.output_path("mask", "mask.png")
    io.write_mask(out_path, ids, labels)
    ctx.set_output("mask", out_path)


def auto_masks(ctx: Context) -> None:
    import torch
    from PIL import Image
    from transformers import pipeline

    path = _model_dir(ctx)
    image = Image.fromarray(io.read_image(ctx.input("image")))
    gen = pipeline("mask-generation", model=str(path),
                   device=0 if ctx.device == "cuda" else -1)
    side = int(ctx.param("points_per_side", 32))
    with torch.inference_mode():
        res = gen(image, points_per_batch=64, crops_n_layers=0,
                  pred_iou_thresh=float(ctx.param("pred_iou_thresh", 0.88)),
                  stability_score_thresh=float(
                      ctx.param("stability_score_thresh", 0.95)),
                  crop_n_points_downscale_factor=1,
                  points_per_crop=side)
    masks = [np.asarray(m, bool) for m in res["masks"]]
    scores = [float(s) for s in res["scores"]]
    min_area = int(ctx.param("min_area", 0))
    items = [(m, s) for m, s in zip(masks, scores)
             if m.sum() > max(min_area, 0)]
    if not items:
        raise NodeError("SAM produced no masks above the thresholds",
                        hint="lower pred_iou_thresh / stability_score_thresh")
    items.sort(key=lambda x: -int(x[0].sum()))
    ids = np.zeros((image.height, image.width), np.uint16)
    labels = {}
    for i, (m, s) in enumerate(items, start=1):
        ids[m] = i
        labels[i] = {"score": round(s, 4), "area": int(m.sum())}
    ctx.log(f"{len(items)} masks")
    out_path = ctx.output_path("mask", "masks.png")
    io.write_mask(out_path, ids, labels)
    ctx.set_output("mask", out_path)


if __name__ == "__main__":
    raise SystemExit(main({"segment_image": segment_image,
                           "auto_masks": auto_masks}))
