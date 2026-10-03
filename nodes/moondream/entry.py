"""Moondream2 node: query / caption / detect / point."""
from __future__ import annotations

import importlib
import sys
import types

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def _warm_page_cache(path) -> None:
    """Read the weights sequentially once: safetensors' scattered mmap reads
    are very slow on the network filesystem holding PDEBUG_HOME."""
    buf = bytearray(64 << 20)
    with open(path, "rb", buffering=0) as fh:
        while fh.readinto(buf):
            pass


def _load(ctx: Context):
    import torch
    import tokenizers

    # the remote code calls Tokenizer.from_pretrained("moondream/starmie-v1")
    # (unpinned network fetch); serve the pinned local copy instead
    tok_file = str(ctx.weight("starmie_tokenizer") / "tokenizer.json")
    orig = tokenizers.Tokenizer.from_pretrained

    def _from_pretrained(name, *args, **kwargs):
        if name == "moondream/starmie-v1":
            return tokenizers.Tokenizer.from_file(tok_file)
        return orig(name, *args, **kwargs)

    tokenizers.Tokenizer.from_pretrained = _from_pretrained
    # import the pinned remote code as a package straight from the weights
    # dir (transformers' trust_remote_code copy into HF_HOME/modules misses
    # nested relative imports such as layers.py / rope.py for local dirs)
    wdir = ctx.weight("moondream2")
    pkg = types.ModuleType("moondream2_hf")
    pkg.__path__ = [str(wdir)]
    sys.modules["moondream2_hf"] = pkg
    HfMoondream = importlib.import_module(
        "moondream2_hf.hf_moondream").HfMoondream
    _warm_page_cache(wdir / "model.safetensors")
    cuda = ctx.device == "cuda"
    model = HfMoondream.from_pretrained(
        str(wdir), torch_dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="cuda" if cuda else "cpu")
    model.eval()
    return model


def _image(ctx: Context):
    from PIL import Image

    img = Image.open(ctx.input("image")).convert("RGB")
    return img, img.size


# NOTE: images are encoded with model.encode_image(img) (settings=None)
# before query/caption/detect/point: at this revision encode_image reads
# settings["variant"] unguarded, so passing settings with a PIL image
# raises KeyError('variant').
def _text_settings(ctx: Context):
    return {"temperature": 0.0, "max_tokens": int(ctx.param("max_tokens",
                                                             512))}


def _object(ctx: Context) -> str:
    obj = str(ctx.param("object", "")).strip()
    if not obj:
        raise NodeError("empty object phrase", hint="pass -p object=wheel",
                        kind="request")
    return obj


def query(ctx: Context) -> None:
    question = str(ctx.param("question", "")).strip()
    if not question:
        raise NodeError("empty question", kind="request",
                        hint="pass -p question='...'")
    img, _ = _image(ctx)
    model = _load(ctx)
    res = model.query(model.encode_image(img), question, reasoning=bool(ctx.param("reasoning")),
                      settings=_text_settings(ctx))
    answer = res["answer"].strip()
    if "reasoning" in res:
        ctx.metadata["reasoning"] = res["reasoning"]
    ctx.log(f"[moondream] answer: {answer}")
    out = ctx.output_path("answer", "answer.txt")
    out.write_text(answer, encoding="utf-8")
    ctx.set_output("answer", out)


def caption(ctx: Context) -> None:
    img, _ = _image(ctx)
    model = _load(ctx)
    res = model.caption(model.encode_image(img), length=ctx.param("length", "normal"),
                        settings=_text_settings(ctx))
    text = res["caption"].strip()
    ctx.log(f"[moondream] caption: {text}")
    out = ctx.output_path("caption", "caption.txt")
    out.write_text(text, encoding="utf-8")
    ctx.set_output("caption", out)


def detect(ctx: Context) -> None:
    obj = _object(ctx)
    img, (w, h) = _image(ctx)
    model = _load(ctx)
    res = model.detect(model.encode_image(img), obj, settings={
        "max_objects": int(ctx.param("max_objects", 50))})
    boxes = []
    for o in res["objects"]:
        # normalised [0,1] -> pixel xyxy, clipped to the image
        x0 = min(max(o["x_min"], 0.0), 1.0) * w
        y0 = min(max(o["y_min"], 0.0), 1.0) * h
        x1 = min(max(o["x_max"], 0.0), 1.0) * w
        y1 = min(max(o["y_max"], 0.0), 1.0) * h
        boxes.append({"xyxy": [round(x0, 2), round(y0, 2), round(x1, 2),
                               round(y1, 2)], "label": obj})
    ctx.log(f"[moondream] {len(boxes)} x {obj!r}: {boxes}")
    out = ctx.output_path("boxes", "boxes.json")
    io.write_bbox_set(out, boxes)
    ctx.set_output("boxes", out)


def point(ctx: Context) -> None:
    obj = _object(ctx)
    img, (w, h) = _image(ctx)
    model = _load(ctx)
    res = model.point(model.encode_image(img), obj, settings={
        "max_objects": int(ctx.param("max_objects", 50))})
    points = [{"xy": [round(p["x"] * w, 2), round(p["y"] * h, 2)],
               "positive": True, "obj_id": i + 1, "frame": 0}
              for i, p in enumerate(res["points"])]
    ctx.log(f"[moondream] {len(points)} x {obj!r}: {points}")
    out = ctx.output_path("points", "points.json")
    io.write_json(out, {"points": points, "boxes": [], "text": obj})
    ctx.set_output("points", out)


if __name__ == "__main__":
    raise SystemExit(main({"query": query, "caption": caption,
                           "detect": detect, "point": point}))
