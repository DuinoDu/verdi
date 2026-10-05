"""OpenAI CLIP ViT-B/32 (official openai/CLIP code + official ViT-B-32.pt):
L2-normalised text embeddings and image / ROI embeddings in the joint space.

* No probabilities: logit_scale is never applied; the optional similarity
  output is the raw cosine of unit vectors (uncalibrated).
* Text: the official tokenizer (ftfy fix + html unescape + whitespace clean +
  lower-case, BPE), context 77 tokens incl. start / end. Longer texts are an
  error unless on_overflow = "truncate" (then truncated = true is recorded).
* ROI: pad_square (default) pads the bbox crop to a square with the CLIP mean
  colour, so the official Resize(224) + CenterCrop(224) keeps the whole ROI;
  center_crop applies the official preprocess to the raw crop and records the
  part that is cut away.
* Continuous pixel coordinates, origin at the top-left image corner; bbox
  x1 / y1 exclusive. The integer crop is [floor(x0), ceil(x1)) x [floor(y0),
  ceil(y1)) and is recorded.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

CKPT = "ViT-B-32.pt"
CKPT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"  # = manifest
CODE_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"  # = manifest [upstream]
CONTEXT = 77
DIM = 512
INPUT = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
FILL = tuple(int(round(255 * m)) for m in CLIP_MEAN)
EMBEDDING_SPACE = ("openai_clip_vit_b32 joint image-text space (official ViT-B-32.pt, "
                   "sha256 40d36571...); L2-normalised; image and text vectors of THIS "
                   "space are comparable by cosine; not comparable with DINOv2 or other CLIPs")


# ------------------------------------------------------------------ model
def _clip(ctx: Context):
    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import clip  # noqa: E402  (official openai/CLIP package from the pinned checkout)

    return clip


def _load(ctx: Context):
    import torch

    clip = _clip(ctx)
    ckpt = ctx.weight("vitb32", prefetch=True) / CKPT
    if not ckpt.exists():
        raise NodeError(f"CLIP checkpoint missing at {ckpt}", kind="setup",
                        hint="verdi setup clip_features")
    dev = "cuda" if ctx.device == "cuda" else "cpu"
    t0 = time.time()
    model, _ = clip.load(str(ckpt), device=dev, jit=False)
    model = model.float().eval()  # float32 on CPU and CUDA (clip.load keeps fp16 on CUDA)
    ctx.log(f"[clip] loaded ViT-B/32 on {dev} in {time.time() - t0:.1f}s")
    return clip, model, dev


def _norm(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _base_info(dev: str) -> Dict[str, Any]:
    return {"model": "ViT-B/32", "checkpoint": CKPT, "checkpoint_sha256": CKPT_SHA256,
            "code": f"openai/CLIP@{CODE_COMMIT}", "embedding_space": EMBEDDING_SPACE,
            "normalize": "L2 (unit vectors); logit_scale NOT applied; no softmax / probabilities",
            "dtype": "float32", "device": dev}


# ------------------------------------------------------------------- text
def read_texts(path: Path) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    texts = [t.strip() for t in lines]
    if not texts or all(not t for t in texts):
        raise NodeError("texts file is empty (one text per line)")
    empty = [i for i, t in enumerate(texts) if not t]
    if empty:
        raise NodeError(f"texts has empty lines at {empty}; one non-empty text per line "
                        "(indices of the output rows = line numbers)")
    return texts


def encode_texts(ctx: Context, clip, model, dev, texts: List[str], overflow: str):
    import torch
    from clip.simple_tokenizer import SimpleTokenizer

    tok = SimpleTokenizer()
    recs, long = [], []
    for i, t in enumerate(texts):
        n = len(tok.encode(t)) + 2
        recs.append({"index": i, "text": t, "n_tokens": n, "truncated": n > CONTEXT})
        if n > CONTEXT:
            long.append(i)
    if long and overflow == "error":
        raise NodeError(f"texts {long} exceed the CLIP context of {CONTEXT} tokens "
                        f"(n_tokens {[recs[i]['n_tokens'] for i in long]})",
                        hint="shorten them or pass on_overflow=truncate (recorded per text)")
    tokens = clip.tokenize(texts, context_length=CONTEXT, truncate=True).to(dev)
    with torch.inference_mode():
        e = model.encode_text(tokens).float().cpu().numpy()
    if not np.isfinite(e).all():
        raise NodeError("non-finite text embeddings")
    return _norm(e).astype(np.float32), recs


def encode_text(ctx: Context) -> None:
    overflow = str(ctx.param("on_overflow", "error"))
    texts = read_texts(ctx.input("texts"))
    clip, model, dev = _load(ctx)
    emb, recs = encode_texts(ctx, clip, model, dev, texts, overflow)
    np.save(ctx.output_path("text_embeddings", "text_embeddings.npy"), emb)
    ctx.set_output("text_embeddings", ctx.output_dir / "text_embeddings.npy")
    info = {**_base_info(dev), "shape": list(emb.shape), "texts": recs,
            "tokenizer": "official SimpleTokenizer (ftfy fix_text, html unescape, whitespace "
                         "clean, lower-case, BPE); context 77 incl. start/end tokens",
            "params": {"on_overflow": overflow},
            "n_truncated": sum(r["truncated"] for r in recs)}
    io.write_json(ctx.output_path("info", "info.json"), info)
    ctx.set_output("info", ctx.output_dir / "info.json")


# -------------------------------------------------------------------- roi
def read_regions(path: Path, n_frames: int, h: int, w: int) -> List[Dict[str, Any]]:
    """Same schema as dinov2_features (region_id, frame_index, object_id, bbox_xyxy, mask)."""
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


def roi_image(img: np.ndarray, r: Dict[str, Any], mode: str, mask_mode: str):
    """-> (224x224 uint8 RGB, record of exactly what the encoder saw)."""
    from PIL import Image

    h, w = img.shape[:2]
    x0, y0, x1, y1 = r["bbox_xyxy"]
    X0, Y0 = int(math.floor(x0)), int(math.floor(y0))
    X1, Y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    crop = img[Y0:Y1, X0:X1].copy()
    rec: Dict[str, Any] = {"crop_xyxy_int": [X0, Y0, X1, Y1]}
    if mask_mode == "fill":
        if r["mask"] is None:
            raise NodeError(f"region {r['region_id']!r}: mask_mode=fill needs a mask")
        crop[~r["mask"][Y0:Y1, X0:X1]] = FILL
        rec["outside_mask"] = "filled with the CLIP mean colour"
    cw, ch = X1 - X0, Y1 - Y0
    if mode == "pad_square":
        s = max(cw, ch)
        canvas = np.empty((s, s, 3), np.uint8)
        canvas[:] = FILL
        ox, oy = (s - cw) // 2, (s - ch) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop
        rec.update(mode="pad_square", padded_side=s, pad_left_top=[ox, oy],
                   pad_fill_rgb=list(FILL), retained_fraction=1.0,
                   retained_xyxy_input=[X0, Y0, X1, Y1])
        src = canvas
    else:  # center_crop: official preprocess on the raw crop
        s = min(cw, ch)
        ox, oy = (cw - s) / 2.0, (ch - s) / 2.0
        rec.update(mode="center_crop",
                   retained_xyxy_input=[X0 + ox, Y0 + oy, X0 + ox + s, Y0 + oy + s],
                   retained_fraction=round(s / max(cw, ch), 6),
                   cropped_away="left/right" if cw > ch else ("top/bottom" if ch > cw else "none"))
        src = crop
    # official transform: Resize(224, BICUBIC) on the short side + CenterCrop(224)
    im = Image.fromarray(src)
    sw, sh = im.size
    # torchvision Resize(int): short side -> 224, long side int(224 * long / short)
    if sw <= sh:
        nw, nh = INPUT, int(INPUT * sh / sw)
    else:
        nw, nh = int(INPUT * sw / sh), INPUT
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = int(round((nw - INPUT) / 2.0)), int(round((nh - INPUT) / 2.0))
    im = im.crop((l, t, l + INPUT, t + INPUT))
    return np.asarray(im), rec


def encode_regions(ctx: Context) -> None:
    import torch

    files = io.list_frames(ctx.input("frames"))
    if not files:
        raise NodeError("frames contains no images")
    imgs = [io.read_image(f) for f in files]
    mode = str(ctx.param("roi_mode", "pad_square"))
    mask_mode = str(ctx.param("mask_mode", "none"))
    if ctx.has_input("regions"):
        sizes = {im.shape[:2] for im in imgs}
        if len(sizes) != 1:
            raise NodeError("regions need one frame resolution per call (mask / bbox coords)")
        h, w = imgs[0].shape[:2]
        regions = read_regions(ctx.input("regions"), len(files), h, w)
    else:  # whole frames
        if mask_mode != "none":
            raise NodeError("mask_mode=fill needs regions with masks")
        regions = [{"region_id": f"frame_{i}", "frame_index": i, "object_id": None,
                    "bbox_xyxy": [0.0, 0.0, float(im.shape[1]), float(im.shape[0])],
                    "mask": None, "mask_file": None} for i, im in enumerate(imgs)]
    texts = read_texts(ctx.input("texts")) if ctx.has_input("texts") else None

    clip, model, dev = _load(ctx)
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    bs = max(1, int(ctx.param("batch_size", 32)))
    rois, recs = [], []
    for k, r in enumerate(regions):
        a, rec = roi_image(imgs[r["frame_index"]], r, mode, mask_mode)
        rois.append(a)
        recs.append({"index": k, "region_id": r["region_id"], "frame_index": r["frame_index"],
                     "frame": files[r["frame_index"]].name, "object_id": r["object_id"],
                     "bbox_xyxy": r["bbox_xyxy"], "mask_file": r["mask_file"], **rec})
    emb = np.zeros((len(rois), DIM), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(rois), bs):
            x = torch.from_numpy(np.stack(rois[s:s + bs])).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - mean) / std).to(dev)
            emb[s:s + x.shape[0]] = model.encode_image(x).float().cpu().numpy()
    if not np.isfinite(emb).all():
        raise NodeError("non-finite image embeddings")
    emb = _norm(emb).astype(np.float32)
    np.save(ctx.output_path("roi_embeddings", "roi_embeddings.npy"), emb)
    ctx.set_output("roi_embeddings", ctx.output_dir / "roi_embeddings.npy")

    info = {**_base_info(dev), "shape": list(emb.shape), "regions": recs,
            "preprocess": "uint8 RGB ROI -> (pad_square: pad to square with CLIP mean colour) "
                          "-> official Resize(224, bicubic, short side) + CenterCrop(224) -> "
                          "/255 -> Normalize(CLIP mean/std)",
            "params": {"roi_mode": mode, "mask_mode": mask_mode, "batch_size": bs},
            "infer_sec": round(time.time() - t0, 3)}
    if texts is not None:
        temb, trecs = encode_texts(ctx, clip, model, dev, texts,
                                   str(ctx.param("on_overflow", "error")))
        cos = emb @ temb.T
        sim = {"note": "cosine similarity of L2-normalised CLIP embeddings (same space); "
                       "uncalibrated, NOT probabilities; logit_scale not applied",
               "regions": [r["region_id"] for r in recs], "texts": texts,
               "cosine": np.round(cos, 6).tolist(),
               "argmax_text": [texts[int(i)] for i in cos.argmax(1)],
               "margin_top1_top2": [float(np.round(np.sort(c)[-1] - np.sort(c)[-2], 6))
                                    if len(texts) > 1 else None for c in cos]}
        np.save(ctx.output_path("text_embeddings", "text_embeddings.npy"), temb)
        ctx.set_output("text_embeddings", ctx.output_dir / "text_embeddings.npy")
        io.write_json(ctx.output_path("similarity", "similarity.json"), sim)
        ctx.set_output("similarity", ctx.output_dir / "similarity.json")
        info["texts"] = trecs
    io.write_json(ctx.output_path("info", "info.json"), info)
    ctx.set_output("info", ctx.output_dir / "info.json")


if __name__ == "__main__":
    raise SystemExit(main({"encode_text": encode_text, "encode_regions": encode_regions}))
