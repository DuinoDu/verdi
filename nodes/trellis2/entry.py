"""TRELLIS.2 node: generate (image -> textured glb)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


def _rgba(ctx: Context):
    """RGBA PIL image whose alpha isolates the object."""
    from PIL import Image

    img = Image.open(ctx.input("image"))
    if ctx.has_input("mask"):
        rgb = np.asarray(img.convert("RGB"))
        ids = np.asarray(Image.open(ctx.input("mask")))
        if ids.ndim == 3:
            ids = ids[..., 0]
        if ids.shape[:2] != rgb.shape[:2]:
            raise NodeError(f"mask size {ids.shape[1]}x{ids.shape[0]} != image "
                            f"size {rgb.shape[1]}x{rgb.shape[0]}")
        mid = int(ctx.param("mask_id", 0))
        fg = ids == mid if mid > 0 else ids > 0
        alpha = (fg * 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    else:
        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info)
        img = img.convert("RGBA")
        alpha = np.asarray(img)[..., 3]
        if not has_alpha or (alpha == 255).all():
            raise NodeError(
                "image has no transparent background and no mask was given",
                hint="pass an RGBA cut-out or `-i mask=...` (e.g. from sam2/"
                "langsam, or hunyuan3d rembg); this node never runs the gated "
                "RMBG-2.0 background remover")
    if (alpha > 0.8 * 255).sum() < 64:
        raise NodeError("object mask/alpha is empty",
                        hint="check the mask id / alpha channel")
    return img


class _NoRembg:
    """Stand-in for the gated BiRefNet/RMBG-2.0 remover (never loaded)."""

    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    cpu = cuda = eval = to

    def __call__(self, *a, **k):
        raise NodeError("background removal requested but not available",
                        hint="pass an RGBA cut-out or an image + mask")


def _no_fa3_off_hopper() -> None:
    """xformers 0.0.32 enables its FlashAttention-3 kernels (Hopper sm_90a
    only) for any compute capability >= 9.0; on Blackwell (sm_120) the
    first attention call aborts ("CUDA error ... invalid argument").
    Turn FA3 off there; xformers then dispatches to FA2 / cutlass."""
    import torch
    from xformers.ops.fmha import dispatch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0):
        dispatch._set_use_fa3(False)


def _pipeline(ctx: Context):
    """Load Trellis2ImageTo3DPipeline from the pinned local weights."""
    _no_fa3_off_hopper()
    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    root = ctx.weight("trellis2_4b", prefetch=True)
    ss_dec = ctx.weight("trellis_image_large", prefetch=True) / "ckpts"
    # DINOv3 files live in separate weight dirs (one url each); give
    # transformers one directory with config.json + model.safetensors
    dino = Path(tempfile.mkdtemp(prefix="dinov3_"))
    for key, fname in (("dinov3", "model.safetensors"),
                       ("dinov3_config", "config.json")):
        (dino / fname).symlink_to(ctx.weight(key, prefetch=True) / fname)
    cfg = json.loads((root / "pipeline.json").read_text())
    args = cfg["args"]
    # point every model at the pinned local copies (upstream resolves the
    # ss decoder and DINOv3 through the HF hub)
    args["models"] = {k: str(root / v) for k, v in args["models"].items()
                      if not v.startswith("microsoft/")} | {
        "sparse_structure_decoder": str(ss_dec / "ss_dec_conv3d_16l8_fp16")}
    args["image_cond_model"]["args"]["model_name"] = str(dino)
    tmp = Path(tempfile.mkdtemp(prefix="trellis2_cfg_"))
    (tmp / "pipeline.json").write_text(json.dumps(cfg))
    rembg.BiRefNet = _NoRembg  # avoid constructing RMBG-2.0
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(tmp))
    pipe.low_vram = bool(ctx.param("low_vram", False))
    if ctx.device != "cuda":
        raise NodeError("TRELLIS.2 needs a CUDA GPU")
    pipe.cuda()
    return pipe


def generate(ctx: Context) -> None:
    import torch

    image = _rgba(ctx)
    pipe = _pipeline(ctx)
    ptype = ctx.param("pipeline_type", "1024_cascade")
    with torch.inference_mode():
        image_in = pipe.preprocess_image(image)  # crop to alpha bbox
        image_in.save(ctx.output_dir / "input_crop.png")
        meshes = pipe.run(
            image_in, seed=int(ctx.param("seed", 42)), preprocess_image=False,
            pipeline_type=ptype,
            max_num_tokens=int(ctx.param("max_num_tokens", 49152)),
            sparse_structure_sampler_params={
                "steps": int(ctx.param("ss_steps", 12)),
                "guidance_strength": float(ctx.param("ss_guidance", 7.5))},
            shape_slat_sampler_params={
                "steps": int(ctx.param("shape_steps", 12)),
                "guidance_strength": float(ctx.param("shape_guidance", 7.5))},
            tex_slat_sampler_params={
                "steps": int(ctx.param("tex_steps", 12)),
                "guidance_strength": float(ctx.param("tex_guidance", 1.0))},
        )
    if not meshes or meshes[0].faces.shape[0] == 0:
        raise NodeError("TRELLIS.2 produced an empty mesh",
                        hint="check the cut-out; try another seed")
    mesh = meshes[0]
    mesh.simplify(16777216)  # nvdiffrast limit (as upstream example.py)
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(ctx.param("num_faces", 300000)),
        texture_size=int(ctx.param("texture_size", 2048)),
        remesh=bool(ctx.param("remesh", True)), remesh_band=1.0,
        remesh_project=float(ctx.param("remesh_project", 0.9)),
        verbose=True)
    out = ctx.output_path("mesh", "mesh.glb")
    glb.export(str(out))
    ctx.set_output("mesh", out)
    v = np.asarray(glb.vertices)
    ctx.metadata.update({
        "pipeline_type": ptype,
        "vertices": int(len(v)), "faces": int(len(glb.faces)),
        "bbox_min": v.min(0).round(4).tolist(),
        "bbox_max": v.max(0).round(4).tolist(),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    })


if __name__ == "__main__":
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    raise SystemExit(main({"generate": generate}))
