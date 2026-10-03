"""Qwen2.5-VL node: chat (-> text) / structured (JSON-Schema validated)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

# size -> (weight key, hf repo, pinned revision)
MODELS = {
    "3b": ("qwen2_5_vl_3b", "Qwen/Qwen2.5-VL-3B-Instruct",
           "66285546d2b821cf421d4f5eb2576359d3770cd3"),
    "7b": ("qwen2_5_vl_7b", "Qwen/Qwen2.5-VL-7B-Instruct",
           "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "32b": ("qwen2_5_vl_32b", "Qwen/Qwen2.5-VL-32B-Instruct",
            "7cfb30d71a1f4f49a57592323337a4a4727301da"),
}
MIN_PIXELS = 4 * 28 * 28


# ------------------------------------------------------------------ model
def _warm_page_cache(path: Path) -> None:
    """Read weight files sequentially once.

    safetensors loads via mmap with scattered page faults, which is very
    slow on the network filesystem holding PDEBUG_HOME (~5 min per 4 GB
    shard); one sequential pass (~1 GB/s) makes the mmap reads hit RAM.
    """
    import time

    t0 = time.time()
    buf = bytearray(64 << 20)
    total = 0
    for f in sorted(path.glob("*.safetensors")):
        with open(f, "rb", buffering=0) as fh:
            while True:
                n = fh.readinto(buf)
                if not n:
                    break
                total += n
    Context.log(f"[qwen] pre-read {total / 2**30:.1f} GiB of weights in "
                f"{time.time() - t0:.0f}s")


def _load(ctx: Context):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    key, repo, rev = MODELS[ctx.param("model", "7b")]
    local = ctx.weights / key
    if local.exists():
        src, kw = str(local), {}
        _warm_page_cache(local)
    else:  # lazy (32b): fetched from the HF mirror into HF_HOME
        if key != "qwen2_5_vl_32b":
            ctx.weight(key)  # raises a setup error
        src, kw = repo, {"revision": rev}
        ctx.log(f"[qwen] {key} not pre-fetched; downloading {repo}@{rev[:8]}")
    cuda = ctx.device == "cuda"
    if not cuda:
        ctx.log("[qwen] WARNING: running on CPU in float32 (very slow)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.bfloat16 if cuda else torch.float32,
        attn_implementation="sdpa", device_map="cuda" if cuda else "cpu",
        **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(src, **kw)
    return model, processor


def _prompt(ctx: Context) -> str:
    text = str(ctx.param("prompt", "") or "")
    if ctx.has_input("prompt_file"):
        if text.strip():
            raise NodeError("both prompt param and prompt_file given",
                            hint="pass exactly one of -p prompt=... or "
                            "-i prompt_file=file.txt", kind="request")
        text = ctx.input("prompt_file").read_text(encoding="utf-8")
    if not text.strip():
        raise NodeError("empty prompt",
                        hint="pass -p prompt='...' or -i prompt_file=file.txt",
                        kind="request")
    return text


def _visual_content(ctx: Context) -> List[Dict[str, Any]]:
    max_pixels = int(ctx.param("max_pixels", 1003520))
    content: List[Dict[str, Any]] = []
    if ctx.has_input("image"):
        content.append({"type": "image",
                        "image": str(ctx.input("image").resolve()),
                        "min_pixels": MIN_PIXELS, "max_pixels": max_pixels})
    if ctx.has_input("frames"):
        frames = io.list_frames(ctx.input("frames"))
        max_frames = int(ctx.param("max_frames", 32))
        if max_frames < 1:
            raise NodeError("max_frames must be >= 1", kind="request")
        if len(frames) > max_frames:
            idx = [round(i * (len(frames) - 1) / (max_frames - 1))
                   if max_frames > 1 else 0 for i in range(max_frames)]
            frames = [frames[i] for i in sorted(set(idx))]
        ctx.metadata["frames_used"] = [f.name for f in frames]
        if ctx.param("frames_mode", "video") == "images":
            for f in frames:
                content.append({"type": "image", "image": str(f.resolve()),
                                "min_pixels": MIN_PIXELS,
                                "max_pixels": max_pixels})
        else:
            # per-frame budget: qwen_vl_utils resizes every frame to it
            content.append({"type": "video",
                            "video": [f"file://{f.resolve()}"
                                      for f in frames],
                            "max_pixels": min(max_pixels, 360 * 420)})
    return content


def _generate(model, processor, messages, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             top_k=None)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


# ------------------------------------------------------------------ tasks
def chat(ctx: Context) -> None:
    prompt = _prompt(ctx)
    messages = []
    if ctx.param("system"):
        messages.append({"role": "system",
                         "content": [{"type": "text",
                                      "text": ctx.param("system")}]})
    messages.append({"role": "user", "content": _visual_content(ctx)
                     + [{"type": "text", "text": prompt}]})
    model, processor = _load(ctx)
    answer = _generate(model, processor, messages,
                       int(ctx.param("max_new_tokens", 512)))
    ctx.log(f"[qwen] answer: {answer}")
    out = ctx.output_path("answer", "answer.txt")
    out.write_text(answer, encoding="utf-8")
    ctx.set_output("answer", out)


_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


def _parse(reply: str, validator) -> Tuple[Optional[Any], str]:
    """Return (value, "") or (None, error message)."""
    body = reply.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1).strip()
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc})"
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        msgs = [f"at {'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
                for e in errors[:5]]
        return None, "schema violation: " + "; ".join(msgs)
    return value, ""


def structured(ctx: Context) -> None:
    import jsonschema

    schema = io.read_json(ctx.input("schema"))
    if not isinstance(schema, dict):
        raise NodeError("schema must be a JSON object (a JSON Schema)",
                        kind="request")
    cls = jsonschema.validators.validator_for(
        schema, default=jsonschema.Draft202012Validator)
    try:
        cls.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise NodeError(f"invalid JSON Schema: {exc.message}",
                        hint="fix the schema file", kind="request") from None
    validator = cls(schema, format_checker=cls.FORMAT_CHECKER)
    prompt = _prompt(ctx)
    system = (
        "You are a precise vision assistant that answers only with JSON. "
        "Your entire reply must be a single JSON value that validates "
        "against this JSON Schema:\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
        + "\nDo not add explanations, comments or markdown. Base every "
        "value on the provided content; do not invent information.")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": _visual_content(ctx)
         + [{"type": "text", "text": prompt}]},
    ]
    model, processor = _load(ctx)
    max_new = int(ctx.param("max_new_tokens", 1024))
    attempts = []
    reply = _generate(model, processor, messages, max_new)
    value, err = _parse(reply, validator)
    attempts.append({"reply": reply, "error": err})
    ctx.log(f"[qwen] reply 1: {reply}\n[qwen] check 1: {err or 'ok'}")
    if err and ctx.param("retry", True):
        messages += [
            {"role": "assistant", "content": [{"type": "text",
                                                "text": reply}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Your reply was rejected: {err}. Reply again with only "
                "the corrected JSON value that validates against the "
                "schema.")}]},
        ]
        reply = _generate(model, processor, messages, max_new)
        value, err = _parse(reply, validator)
        attempts.append({"reply": reply, "error": err})
        ctx.log(f"[qwen] reply 2: {reply}\n[qwen] check 2: {err or 'ok'}")
    ctx.metadata["attempts"] = attempts
    if err:
        raise NodeError(
            f"model reply does not satisfy the schema after "
            f"{len(attempts)} attempt(s): {err}",
            hint="simplify the schema, make the prompt more explicit, raise "
            "max_new_tokens or use model=32b; last reply: " + reply[:500])
    out = ctx.output_path("result", "result.json")
    io.write_json(out, value)
    ctx.set_output("result", out)


if __name__ == "__main__":
    raise SystemExit(main({"chat": chat, "structured": structured}))
