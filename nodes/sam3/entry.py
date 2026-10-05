"""SAM 3 node: segment_text / segment_prompts / track_text / track_prompts."""
from __future__ import annotations

import os
import re
import shutil
import time
from collections import defaultdict

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


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


class _Video:
    """SAM 3 video session over an image_seq + per-frame instance records."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.frames = io.list_frames(ctx.input("frames"))
        if not self.frames:
            raise NodeError("no frames found", hint="image_seq dir of %06d.jpg")
        # sam3's folder loader sorts by integer file stem: link as %06d<ext>
        self.link_dir = ctx.output_path("_frames")
        for i, f in enumerate(self.frames):
            dst = self.link_dir / io.frame_name(i, f.suffix.lower())
            if not dst.exists():
                os.symlink(f.resolve(), dst)
        self.h, self.w = io.read_image(self.frames[0]).shape[:2]
        self.ids = [np.zeros((self.h, self.w), np.uint16) for _ in self.frames]
        # per-frame records: frame -> id -> {score, area_px, bbox_xyxy}
        self.records = [dict() for _ in self.frames]
        self.names: dict = {}
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        self.pred = Sam3VideoPredictor(checkpoint_path=_ckpt(ctx), bpe_path=_bpe())
        self.sid = self.pred.handle_request({
            "type": "start_session", "resource_path": str(self.link_dir)})["session_id"]

    def req(self, **kw):
        kw["session_id"] = self.sid
        return self.pred.handle_request(kw)

    def propagate(self, thr: float, id_map, label_of, direction="both",
                  start_frame=None):
        """Run propagation; id_map(upstream obj id) -> output id or None.

        start_frame is required for tracker-only (point/box) prompts: SAM 3
        then runs its partial (tracker) propagation from that frame."""
        ignored = set()
        req = {"type": "propagate_in_video", "session_id": self.sid,
               "propagation_direction": direction, "output_prob_thresh": thr}
        if start_frame is not None:
            req["start_frame_index"] = int(start_frame)
        for resp in self.pred.handle_stream_request(req):
            fidx = int(resp["frame_index"])
            out = resp["outputs"]
            obj_ids = np.asarray(out["out_obj_ids"]).reshape(-1)
            probs = np.asarray(out.get("out_probs", np.full(len(obj_ids), np.nan))).reshape(-1)
            masks = np.asarray(out["out_binary_masks"])
            for k, oid in enumerate(obj_ids):
                nid = id_map(int(oid))
                if nid is None:
                    ignored.add(int(oid))
                    continue
                m = masks[k].astype(bool)
                if not m.any():
                    continue
                self.ids[fidx][m] = nid
                ys, xs = np.nonzero(m)
                self.records[fidx][nid] = {
                    "score": float(probs[k]), "area_px": int(m.sum()),
                    "bbox_xyxy": [float(xs.min()), float(ys.min()),
                                  float(xs.max() + 1), float(ys.max() + 1)]}
                self.names[nid] = label_of(int(oid))
        return ignored

    def close(self):
        import torch

        try:
            self.req(type="close_session")
        finally:
            torch.cuda.empty_cache()
            shutil.rmtree(self.link_dir, ignore_errors=True)

    def write(self, expected_ids=None, prompt_frames=None):
        """masks (mask_seq) + tracks (json: per-frame visibility/score)."""
        ctx = self.ctx
        out_dir = ctx.output_path("masks")
        for i, ids in enumerate(self.ids):
            io.write_mask(out_dir / io.frame_name(i), ids)
        all_ids = sorted(set(self.names) | set(expected_ids or []))
        inst = {}
        for nid in all_ids:
            seen = [i for i, r in enumerate(self.records) if nid in r]
            sc = [self.records[i][nid]["score"] for i in seen]
            inst[str(nid)] = {
                "label": self.names.get(nid) or (expected_ids or {}).get(nid),
                "score": float(np.nanmean(sc)) if sc else None,
                "frames": len(seen),
                "first_frame": seen[0] if seen else None,
                "last_frame": seen[-1] if seen else None}
        frames_tab = {str(i): {str(k): v for k, v in sorted(r.items())}
                      for i, r in enumerate(self.records)}
        io.write_json(out_dir / "labels.json", {
            **{k: {"label": v["label"], "score": v["score"], "frames": v["frames"]}
               for k, v in inst.items()},
        })
        visible = [[nid in r for nid in all_ids] for r in self.records]
        score = [[(r[nid]["score"] if nid in r else None) for nid in all_ids]
                 for r in self.records]
        tpath = ctx.output_path("tracks", "tracks.json")
        io.write_json(tpath, {
            "convention": {
                "visible": "true = SAM 3 returned a non-empty mask for this id on "
                           "this frame (after its own suppression / unconfirmed / "
                           "removed filtering); false = no mask: occluded, out of "
                           "view or lost. SAM 3 does not tell these apart.",
                "score": "upstream per-object score on that frame: detector "
                         "probability for text-detected objects (track_text); "
                         "for point/box-prompted objects (track_prompts) SAM 3 "
                         "fixes it to 1.0 (user-confirmed), so it is NOT a "
                         "per-frame confidence there; null when not visible",
                "frame": "index into the sorted input frames (names in `frames`)",
                "ids": "instance ids as painted into masks/%06d.png"},
            "frames": [f.name for f in self.frames],
            "prompt_frames": prompt_frames,
            "ids": all_ids, "instances": inst,
            "visible": visible, "score": score,
            "per_frame": frames_tab})
        ctx.set_output("masks", out_dir)
        ctx.set_output("tracks", tpath)
        lost = [k for k, v in inst.items() if v["frames"] == 0]
        ctx.metadata.update({"tracks": len(self.names),
                             "ids_never_visible": lost})


def track_text(ctx: Context) -> None:
    _need_cuda(ctx)
    phrases = _phrases(ctx)
    thr = float(ctx.param("confidence_threshold", 0.5))
    pf = int(ctx.param("prompt_frame", 0))
    v = _Video(ctx)
    if not 0 <= pf < len(v.frames):
        v.close()
        raise NodeError(f"prompt_frame {pf} out of range (0..{len(v.frames) - 1})")
    offset = 0
    per_phrase = {}
    try:
        for phrase in phrases:
            v.req(type="reset_session")
            v.req(type="add_prompt", frame_index=pf, text=phrase,
                  output_prob_thresh=thr)
            seen = []

            def id_map(oid, _off=offset, _seen=seen):
                _seen.append(oid)
                return _off + oid + 1

            v.propagate(thr, id_map, lambda oid, _p=phrase: _p)
            per_phrase[phrase] = sorted({offset + o + 1 for o in seen})
            offset += (max(seen) + 1) if seen else 0
    finally:
        v.close()
    v.write(prompt_frames=[pf])
    ctx.metadata["ids_per_phrase"] = per_phrase
    ctx.metadata["phrases_without_detection"] = [p for p, i in per_phrase.items() if not i]


def track_prompts(ctx: Context) -> None:
    """Instance tracking from point / box prompts (SAM 3 tracker)."""
    _need_cuda(ctx)
    data = io.read_json(ctx.input("prompts"))
    groups = defaultdict(lambda: {"points": [], "labels": [], "box": None,
                                  "frame": None, "label": None})
    for p in data.get("points", []):
        g = groups[int(p.get("obj_id", 1))]
        g["points"].append([float(p["xy"][0]), float(p["xy"][1])])
        g["labels"].append(1 if p.get("positive", True) else 0)
        g.setdefault("_frames", set()).add(int(p.get("frame", 0)))
        g["label"] = p.get("label", g["label"])
    for b in data.get("boxes", []):
        g = groups[int(b.get("obj_id", 1))]
        if g["box"] is not None:
            raise NodeError(f"obj_id {b.get('obj_id', 1)} has more than one box")
        g["box"] = [float(x) for x in b["xyxy"]]
        g.setdefault("_frames", set()).add(int(b.get("frame", 0)))
        g["label"] = b.get("label", g["label"])
    if not groups:
        raise NodeError("prompt set has no points or boxes",
                        hint="use task track_text for text prompts")
    for oid, g in groups.items():
        fr = g.pop("_frames")
        if len(fr) != 1:
            raise NodeError(f"obj_id {oid}: all its prompts must be on ONE frame, got {sorted(fr)}")
        g["frame"] = fr.pop()
        if oid < 1 or oid > 65535:
            raise NodeError(f"obj_id {oid} must be in 1..65535 (mask ids)")
    v = _Video(ctx)
    W, H = v.w, v.h
    try:
        for oid, g in sorted(groups.items()):
            if not 0 <= g["frame"] < len(v.frames):
                raise NodeError(f"obj_id {oid}: frame {g['frame']} out of range")
            pts, lab = list(g["points"]), list(g["labels"])
            if g["box"] is not None:
                x0, y0, x1, y1 = g["box"]
                if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                    raise NodeError(f"obj_id {oid}: box {g['box']} is not xyxy inside {W}x{H}")
                # SAM box prompt = two corner points with labels 2 / 3
                pts = [[x0, y0], [x1, y1]] + pts
                lab = [2, 3] + lab
            rel = [[x / W, y / H] for x, y in pts]
            r = v.req(type="add_prompt", frame_index=g["frame"], points=rel,
                      point_labels=lab, obj_id=oid)
            got = np.asarray(r["outputs"]["out_obj_ids"]).reshape(-1).tolist() \
                if isinstance(r.get("outputs"), dict) else None
            ctx.log(f"[sam3] obj {oid} prompted on frame {g['frame']}: outputs ids {got}")
        thr = float(ctx.param("confidence_threshold", 0.0))
        wanted = set(groups)
        ignored = v.propagate(thr, lambda o: o if o in wanted else None,
                              lambda o: groups[o]["label"],
                              start_frame=min(g["frame"] for g in groups.values()))
    finally:
        v.close()
    v.write(expected_ids={o: g["label"] for o, g in groups.items()},
            prompt_frames={str(o): g["frame"] for o, g in groups.items()})
    if ignored:
        ctx.metadata["ignored_upstream_ids"] = sorted(ignored)


if __name__ == "__main__":
    raise SystemExit(main({"segment_text": segment_text,
                           "segment_prompts": segment_prompts,
                           "track_text": track_text,
                           "track_prompts": track_prompts}))
