"""Grounding DINO node (HF transformers): detect."""
from __future__ import annotations

import re

from verdi.sdk import Context, NodeError, main
from verdi.types import io


def normalize_text(text: str) -> tuple[str, list[str]]:
    """'Truck, tire' -> ('truck. tire.', ['truck', 'tire'])."""
    phrases = [p.strip().lower() for p in re.split(r"[.,;\n]", text or "")]
    phrases = [p for p in phrases if p]
    if not phrases:
        raise NodeError("text is empty", hint="pass -p text='cat. dog'",
                        kind="request")
    return ". ".join(phrases) + ".", phrases


def detect(ctx: Context) -> None:
    import torch
    from PIL import Image
    from transformers import (AutoModelForZeroShotObjectDetection,
                              AutoProcessor)

    prompt, phrases = normalize_text(ctx.param("text"))
    path = ctx.weight(ctx.param("checkpoint", "base"))
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(path)
    model = model.to(ctx.device).eval()
    image = Image.fromarray(io.read_image(ctx.input("image")))
    inputs = processor(images=image, text=prompt,
                       return_tensors="pt").to(ctx.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        threshold=float(ctx.param("box_threshold", 0.35)),
        text_threshold=float(ctx.param("text_threshold", 0.25)),
        target_sizes=[(image.height, image.width)])[0]
    labels = res.get("text_labels", res.get("labels"))
    boxes = []
    for box, score, label in zip(res["boxes"].tolist(),
                                 res["scores"].tolist(), labels):
        x0, y0, x1, y1 = box
        x0, x1 = max(0.0, x0), min(float(image.width), x1)
        y0, y1 = max(0.0, y0), min(float(image.height), y1)
        boxes.append({"xyxy": [round(v, 2) for v in (x0, y0, x1, y1)],
                      "score": round(float(score), 4),
                      "label": str(label).strip()})
    boxes.sort(key=lambda b: -b["score"])
    ctx.log(f"prompt {prompt!r}: {len(boxes)} boxes")
    out = ctx.output_path("boxes", "boxes.json")
    io.write_bbox_set(out, boxes)
    ctx.metadata["prompt"] = prompt
    ctx.set_output("boxes", out)


if __name__ == "__main__":
    raise SystemExit(main({"detect": detect}))
