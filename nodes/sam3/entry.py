"""SAM 3 node: segment_text / segment_prompts / track_text."""
from __future__ import annotations

import os
import re
import shutil
import time
from collections import defaultdict

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def _phrases(ctx: Context):
    text = ctx.param("text")
    if not text:
        raise NodeError("param text is required",
                        hint="e.g. -p text='truck. tire'")
    phrases = [p.strip() for p in re.split(r"[.,;]", text) if p.strip()]
    if not phrases:
        raise NodeError(f"no phrase in text={text!r}",
                        hint="separate noun phrases with '.' or ','")
    return phrases


def _ckpt(ctx: Context) -> str:
    t = time.time()
    path = ctx.weight("sam3", prefetch=True) / "sam3.pt"
    ctx.log(f"[sam3] checkpoint prefetch {time.time() - t:.1f}s")
    return str(path)


def _bpe() -> str:
    import sam3

    return os.path.join(os.path.dirname(sam3.__file__), "assets",
                        "bpe_simple_vocab_16e6.txt.gz")


def _need_cuda(ctx: Context):
    import torch

    if ctx.device != "cuda" or not torch.cuda.is_available():
        raise NodeError("SAM 3 needs a CUDA GPU",
                        hint="run with a GPU (bf16 autocast + triton kernels)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _image_model(ctx: Context, interactive: bool):
    from sam3 import build_sam3_image_model

    ckpt = _ckpt(ctx)
    t = time.time()
    model = build_sam3_image_model(
        bpe_path=_bpe(), device=ctx.device, checkpoint_path=ckpt,
        load_from_HF=False, enable_inst_interactivity=interactive)
    ctx.log(f"[sam3] model build+load {time.time() - t:.1f}s")
    return model


def _paint(dets, shape):
    """dets: list of (mask bool HxW, info); id i = dets[i-1]; small wins."""
    ids = np.zeros(shape, np.uint16)
    order = sorted(range(len(dets)), key=lambda i: -int(dets[i][0].sum()))
    for i in order:
        ids[dets[i][0]] = i + 1
    return ids


def segment_text(ctx: Context) -> None:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor

    _need_cuda(ctx)
    phrases = _phrases(ctx)
    thr = float(ctx.param("confidence_threshold", 0.5))
    processor = Sam3Processor(_image_model(ctx, False), device=ctx.device,
                              confidence_threshold=thr)
    pil = Image.fromarray(io.read_image(ctx.input("image")))
    dets = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(pil)
        for phrase in phrases:
            processor.reset_all_prompts(state)
            res = processor.set_text_prompt(state=state, prompt=phrase)
            masks = res["masks"].cpu().numpy()[:, 0]
            boxes = res["boxes"].float().cpu().numpy()
            scores = res["scores"].float().cpu().numpy()
            for m, b, s in zip(masks, boxes, scores):
                if m.any():
                    dets.append((m.astype(bool), {
                        "label": phrase, "score": float(s),
                        "xyxy": [float(v) for v in b]}))
    dets.sort(key=lambda d: -d[1]["score"])
    ids = _paint(dets, (pil.height, pil.width))
    labels = {i + 1: {"label": d[1]["label"], "score": d[1]["score"]}
              for i, d in enumerate(dets)}
    out = ctx.output_path("mask", "mask.png")
    io.write_mask(out, ids, labels)
    ctx.set_output("mask", out)
    boxes_out = ctx.output_path("boxes", "boxes.json")
    io.write_bbox_set(boxes_out, [
        {"xyxy": d[1]["xyxy"], "score": d[1]["score"], "label": d[1]["label"],
         "id": i + 1} for i, d in enumerate(dets)])
    ctx.set_output("boxes", boxes_out)
    ctx.metadata["detections"] = len(dets)


def segment_prompts(ctx: Context) -> None:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor

    _need_cuda(ctx)
    data = io.read_json(ctx.input("prompts"))
    grouped = defaultdict(lambda: {"points": [], "labels": [], "box": None})
    for p in data.get("points", []):
        g = grouped[int(p.get("obj_id", 1))]
        g["points"].append(p["xy"])
        g["labels"].append(1 if p.get("positive", True) else 0)
    for b in data.get("boxes", []):
        grouped[int(b.get("obj_id", 1))]["box"] = b["xyxy"]
    if not grouped:
        raise NodeError("prompt set has no points or boxes",
                        hint="use task segment_text for text prompts")
    model = _image_model(ctx, True)
    processor = Sam3Processor(model, device=ctx.device)
    image = io.read_image(ctx.input("image"))
    ids = np.zeros(image.shape[:2], np.uint16)
    labels = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(Image.fromarray(image))
        for obj_id, pr in sorted(grouped.items()):
            masks, scores, _ = model.predict_inst(
                state,
                point_coords=np.array(pr["points"], np.float32)
                if pr["points"] else None,
                point_labels=np.array(pr["labels"], np.int32)
                if pr["points"] else None,
                box=np.array(pr["box"], np.float32)
                if pr["box"] is not None else None,
                multimask_output=bool(ctx.param("multimask")),
            )
            best = int(np.argmax(scores))
            ids[np.asarray(masks[best]) > 0] = obj_id
            labels[obj_id] = {"score": float(scores[best])}
    out = ctx.output_path("mask", "mask.png")
    io.write_mask(out, ids, labels)
    ctx.set_output("mask", out)


def track_text(ctx: Context) -> None:
    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    _need_cuda(ctx)
    phrases = _phrases(ctx)
    thr = float(ctx.param("confidence_threshold", 0.5))
    frames = io.list_frames(ctx.input("frames"))
    if not frames:
        raise NodeError("no frames found", hint="image_seq dir of %06d.jpg")
    # sam3's folder loader sorts by integer file stem: link as %06d<ext>
    link_dir = ctx.output_path("_frames")
    for i, f in enumerate(frames):
        dst = link_dir / io.frame_name(i, f.suffix.lower())
        if not dst.exists():
            os.symlink(f.resolve(), dst)
    h, w = io.read_image(frames[0]).shape[:2]
    per_frame = [np.zeros((h, w), np.uint16) for _ in frames]
    scores = defaultdict(list)
    names = {}
    predictor = Sam3VideoPredictor(checkpoint_path=_ckpt(ctx),
                                   bpe_path=_bpe())
    sid = predictor.handle_request({"type": "start_session",
                                    "resource_path": str(link_dir)})["session_id"]
    offset = 0
    try:
        for phrase in phrases:
            predictor.handle_request({"type": "reset_session",
                                      "session_id": sid})
            predictor.handle_request({
                "type": "add_prompt", "session_id": sid,
                "frame_index": int(ctx.param("prompt_frame", 0)),
                "text": phrase, "output_prob_thresh": thr})
            max_id = -1
            for resp in predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": sid,
                    "output_prob_thresh": thr}):
                fidx = int(resp["frame_index"])
                out = resp["outputs"]
                obj_ids = np.asarray(out["out_obj_ids"]).reshape(-1)
                probs = np.asarray(out.get("out_probs",
                                           np.ones(len(obj_ids)))).reshape(-1)
                masks = np.asarray(out["out_binary_masks"])
                for k, oid in enumerate(obj_ids):
                    m = masks[k].astype(bool)
                    if not m.any():
                        continue
                    nid = offset + int(oid) + 1
                    per_frame[fidx][m] = nid
                    scores[nid].append(float(probs[k]))
                    names[nid] = phrase
                    max_id = max(max_id, int(oid))
            offset += max_id + 1
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})
        torch.cuda.empty_cache()
        shutil.rmtree(link_dir, ignore_errors=True)
    out_dir = ctx.output_path("masks")
    for i, ids in enumerate(per_frame):
        io.write_mask(out_dir / io.frame_name(i), ids)
    io.write_json(out_dir / "labels.json", {
        str(k): {"label": names[k], "score": float(np.mean(v)),
                 "frames": len(v)} for k, v in sorted(scores.items())})
    ctx.set_output("masks", out_dir)
    ctx.metadata["tracks"] = len(names)


if __name__ == "__main__":
    raise SystemExit(main({"segment_text": segment_text,
                           "segment_prompts": segment_prompts,
                           "track_text": track_text}))
