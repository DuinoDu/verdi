"""LangSAM node: segment / segment_frames / crop_roi / auto_masks."""
from __future__ import annotations

import re

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

SAM_FILES = {"tiny": "sam2.1_hiera_tiny.pt", "small": "sam2.1_hiera_small.pt",
             "base_plus": "sam2.1_hiera_base_plus.pt",
             "large": "sam2.1_hiera_large.pt"}


def _prompt(ctx: Context) -> str:
    phrases = [p.strip().lower()
               for p in re.split(r"[.,;\n]", ctx.param("text") or "")]
    phrases = [p for p in phrases if p]
    if not phrases:
        raise NodeError("text is empty", hint="pass -p text='cat. dog'",
                        kind="request")
    return ". ".join(phrases) + "."


def _sam_ckpt(ctx: Context) -> str:
    key = ctx.param("sam_type", "small")
    return str(ctx.weight(key) / SAM_FILES[key])


def _langsam(ctx: Context):
    from lang_sam import LangSAM

    gd = str(ctx.weight("gdino"))
    key = ctx.param("sam_type", "small")
    return LangSAM(sam_type=f"sam2.1_hiera_{key}", sam_ckpt_path=_sam_ckpt(ctx),
                   gdino_model_ckpt_path=gd, gdino_processor_ckpt_path=gd,
                   device=ctx.device)


def _predict(ctx: Context, model, rgb: np.ndarray, prompt: str):
    """-> (ids HxW, labels {id: {...}}, boxes, masks [N,H,W] by rank)."""
    from PIL import Image

    res = model.predict([Image.fromarray(rgb)], [prompt],
                        box_threshold=float(ctx.param("box_threshold", 0.3)),
                        text_threshold=float(ctx.param("text_threshold", 0.25)))[0]
    h, w = rgb.shape[:2]
    ids = np.zeros((h, w), np.uint16)
    labels_txt = list(res.get("text_labels", res.get("labels")) or [])
    if not labels_txt:
        return ids, {}, [], np.zeros((0, h, w), bool)
    boxes = np.asarray(res["boxes"], float).reshape(-1, 4)
    scores = np.atleast_1d(np.asarray(res["scores"], float))
    masks = np.asarray(res["masks"])
    if masks.ndim == 2:
        masks = masks[None]
    masks = masks.reshape(-1, h, w) > 0.5
    mscores = np.atleast_1d(np.asarray(res["mask_scores"], float))
    if not (len(boxes) == len(scores) == len(masks) == len(labels_txt)):
        raise NodeError(f"inconsistent LangSAM output: {len(boxes)} boxes, "
                        f"{len(masks)} masks, {len(labels_txt)} labels",
                        kind="contract")
    order = np.argsort(-scores)
    out_boxes, labels = [], {}
    for rank, k in enumerate(order, start=1):
        x0, y0, x1, y1 = boxes[k]
        out_boxes.append({
            "xyxy": [round(float(max(0, x0)), 2), round(float(max(0, y0)), 2),
                     round(float(min(w, x1)), 2), round(float(min(h, y1)), 2)],
            "score": round(float(scores[k]), 4),
            "label": str(labels_txt[k]).strip()})
        labels[rank] = {"label": str(labels_txt[k]).strip(),
                        "score": round(float(scores[k]), 4),
                        "mask_score": round(float(mscores[k]), 4)
                        if k < len(mscores) else None}
    # paint large masks first so small (often nested) objects stay visible
    for rank in sorted(labels, key=lambda r: -int(masks[order[r - 1]].sum())):
        ids[masks[order[rank - 1]]] = rank
    return ids, labels, out_boxes, masks[order]


def segment(ctx: Context) -> None:
    prompt = _prompt(ctx)
    model = _langsam(ctx)
    ids, labels, boxes, _ = _predict(ctx, model, io.read_image(ctx.input("image")),
                                  prompt)
    ctx.log(f"{prompt!r}: {len(boxes)} detections")
    mpath = ctx.output_path("mask", "mask.png")
    io.write_mask(mpath, ids, labels)
    bpath = ctx.output_path("boxes", "boxes.json")
    io.write_bbox_set(bpath, boxes)
    ctx.metadata["prompt"] = prompt
    ctx.set_output("mask", mpath)
    ctx.set_output("boxes", bpath)


def segment_frames(ctx: Context) -> None:
    prompt = _prompt(ctx)
    frames = io.list_frames(ctx.input("frames"))
    model = _langsam(ctx)
    out_dir = ctx.output_path("masks")
    all_boxes, all_labels = [], {}
    for i, f in enumerate(frames):
        ids, labels, boxes, _ = _predict(ctx, model, io.read_image(f), prompt)
        io.write_mask(out_dir / io.frame_name(i), ids)
        all_labels[str(i)] = {str(k): v for k, v in labels.items()}
        all_boxes += [{**b, "frame": i} for b in boxes]
        ctx.log(f"frame {i}: {len(boxes)} detections")
    io.write_json(out_dir / "labels.json", all_labels)
    bpath = ctx.output_path("boxes", "boxes.json")
    io.write_bbox_set(bpath, all_boxes)
    ctx.metadata["prompt"] = prompt
    ctx.set_output("masks", out_dir)
    ctx.set_output("boxes", bpath)


def crop_roi(ctx: Context) -> None:
    from PIL import Image

    prompt = _prompt(ctx)
    rgb = io.read_image(ctx.input("image"))
    _, labels, boxes, masks = _predict(ctx, _langsam(ctx), rgb, prompt)
    if not labels:
        raise NodeError(f"no object found for {prompt!r}",
                        hint="lower box_threshold or rephrase text")
    mask = masks[0]
    if not mask.any():
        raise NodeError("top detection has an empty SAM mask",
                        hint="rephrase text or try a larger sam_type")
    ys, xs = np.nonzero(mask)
    m = int(ctx.param("margin", 20))
    x0, y0 = max(int(xs.min()) - m, 0), max(int(ys.min()) - m, 0)
    x1 = min(int(xs.max()) + 1 + m, rgb.shape[1])
    y1 = min(int(ys.max()) + 1 + m, rgb.shape[0])
    crop, cmask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    size = int(ctx.param("size", 1024))
    scale = min(size / crop.shape[0], size / crop.shape[1])
    nw = max(1, int(round(crop.shape[1] * scale)))
    nh = max(1, int(round(crop.shape[0] * scale)))
    img = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
    alpha = Image.fromarray(cmask.astype(np.uint8) * 255).resize(
        (nw, nh), Image.NEAREST)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img = img.convert("RGBA")
    img.putalpha(alpha)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    cpath = ctx.output_path("crop", "crop.png")
    canvas.save(cpath)
    mpath = ctx.output_path("mask", "mask.png")
    io.write_mask(mpath, mask.astype(np.uint8), {1: labels[1]})
    ctx.metadata.update({"prompt": prompt, "crop_xyxy": [x0, y0, x1, y1],
                         "detection": boxes[0]})
    ctx.set_output("crop", cpath)
    ctx.set_output("mask", mpath)


def auto_masks(ctx: Context) -> None:
    from lang_sam.models.sam import SAM

    key = ctx.param("sam_type", "small")
    sam = SAM()
    sam.build_model(f"sam2.1_hiera_{key}", _sam_ckpt(ctx), device=ctx.device)
    rgb = io.read_image(ctx.input("image"))
    res = sam.generate(rgb)
    if not res:
        raise NodeError("SAM 2.1 produced no masks", kind="node")
    res.sort(key=lambda r: -int(r["area"]))
    ids = np.zeros(rgb.shape[:2], np.uint16)
    labels = {}
    for i, r in enumerate(res, start=1):
        ids[np.asarray(r["segmentation"], bool)] = i
        labels[i] = {"score": round(float(r["predicted_iou"]), 4),
                     "stability_score": round(float(r["stability_score"]), 4),
                     "area": int(r["area"])}
    ctx.log(f"{len(res)} masks")
    mpath = ctx.output_path("mask", "masks.png")
    io.write_mask(mpath, ids, labels)
    ctx.set_output("mask", mpath)


if __name__ == "__main__":
    raise SystemExit(main({"segment": segment,
                           "segment_frames": segment_frames,
                           "crop_roi": crop_roi,
                           "auto_masks": auto_masks}))
