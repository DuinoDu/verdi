"""Hunyuan3D-2.1 node: rembg / shape / paint."""
from __future__ import annotations

import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


def _paths(ctx: Context) -> None:
    """Put the upstream packages on sys.path the way demo.py does."""
    repo = ctx.repo
    for p in (repo / "hy3dpaint" / "custom_rasterizer", repo / "hy3dpaint",
              repo / "hy3dshape"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    # basicsr (Real-ESRGAN) imports a module removed in torchvision 0.17
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        import torchvision.transforms.functional as F

        sys.modules["torchvision.transforms.functional_tensor"] = F


def _rgba(ctx: Context, name: str, remove_bg: str):
    """Load an image as RGBA; run rembg when it has no usable alpha."""
    from PIL import Image

    img = Image.open(ctx.input(name))
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA")
    if has_alpha:
        a = np.asarray(img)[..., 3]
        has_alpha = bool((a < 128).any())  # fully opaque alpha is useless
    if remove_bg == "always" or (remove_bg == "auto" and not has_alpha):
        img = _remover(ctx)(img.convert("RGB"))
    elif remove_bg == "never" and not has_alpha:
        raise NodeError("image has no transparent background",
                        hint="pass remove_background=auto, or an RGBA "
                        "image with the object cut out")
    alpha = np.asarray(img)[..., 3]
    if (alpha > 127).mean() < 1e-3:
        raise NodeError("background removal left no foreground object",
                        hint="use an object-centric photo or give an RGBA "
                        "image with the object cut out")
    return img


def _remover(ctx: Context):
    os.environ["U2NET_HOME"] = str(ctx.weight("u2net"))
    _paths(ctx)
    from hy3dshape.rembg import BackgroundRemover

    return BackgroundRemover()


# ------------------------------------------------------------------ rembg
def rembg(ctx: Context) -> None:
    from PIL import Image

    img = Image.open(ctx.input("image")).convert("RGB")
    out = _remover(ctx)(img)
    rgba = np.asarray(out.convert("RGBA"))
    fg = rgba[..., 3] > int(ctx.param("alpha_threshold", 127))
    if not fg.any():
        raise NodeError("no foreground found",
                        hint="u2net expects one salient object")
    img_path = ctx.output_path("image", "image_rgba.png")
    Image.fromarray(rgba, "RGBA").save(img_path)
    mask_path = ctx.output_path("mask", "mask.png")
    io.write_mask(mask_path, fg.astype(np.uint8),
                  {1: {"label": "foreground"}})
    ctx.set_output("image", img_path)
    ctx.set_output("mask", mask_path)
    ctx.metadata["foreground_ratio"] = float(fg.mean())


# ------------------------------------------------------------------ shape
def shape(ctx: Context) -> None:
    import torch

    image = _rgba(ctx, "image", ctx.param("remove_background", "auto"))
    _paths(ctx)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(ctx.weight("hunyuan3d")), subfolder="hunyuan3d-dit-v2-1",
        device=ctx.device)
    gen = torch.Generator(device="cpu").manual_seed(int(ctx.param("seed", 0)))
    meshes = pipe(image=image,
                  num_inference_steps=int(ctx.param("steps", 50)),
                  guidance_scale=float(ctx.param("guidance_scale", 5.0)),
                  octree_resolution=int(ctx.param("octree_resolution", 384)),
                  num_chunks=int(ctx.param("num_chunks", 8000)),
                  generator=gen)
    mesh = meshes[0] if isinstance(meshes, (list, tuple)) and meshes else None
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        raise NodeError("Hunyuan3D-Shape produced no surface",
                        hint="use an object-centric image with a clean "
                        "alpha mask; try more steps or another seed")
    if ctx.param("cleanup", True):
        from hy3dshape.postprocessors import (DegenerateFaceRemover,
                                              FloaterRemover)
        mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
    max_faces = int(ctx.param("max_faces", 0))
    if max_faces and len(mesh.faces) > max_faces:
        from hy3dshape.postprocessors import FaceReducer
        mesh = FaceReducer()(mesh, max_facenum=max_faces)
    out = ctx.output_path("mesh", "mesh.glb")
    mesh.export(str(out))
    ctx.set_output("mesh", out)
    ctx.metadata.update(_mesh_stats(mesh))
    image.save(ctx.output_dir / "input_rgba.png")  # what the model saw


def _mesh_stats(mesh) -> dict:
    lo, hi = mesh.bounds
    return {"vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds_min": [round(float(x), 4) for x in lo],
            "bounds_max": [round(float(x), 4) for x in hi],
            "units": "normalised (fits in [-1,1]^3), not metres; +Y up"}


# ------------------------------------------------------------------ paint
def paint(ctx: Context) -> None:
    import torch  # noqa: F401
    import trimesh

    image = _rgba(ctx, "image", ctx.param("remove_background", "auto"))
    _paths(ctx)
    repo = ctx.repo
    try:
        import custom_rasterizer  # noqa: F401
        import custom_rasterizer_kernel  # noqa: F401
        from DifferentiableRenderer import mesh_inpaint_processor  # noqa
    except ImportError as exc:
        raise NodeError(f"compiled extension missing: {exc}",
                        hint="verdi setup hunyuan3d (builds "
                        "custom_rasterizer + mesh_inpaint_processor)",
                        kind="setup") from exc
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    conf = Hunyuan3DPaintConfig(int(ctx.param("max_num_view", 6)),
                                int(ctx.param("resolution", 512)))
    conf.device = ctx.device
    conf.multiview_cfg_path = str(repo / "hy3dpaint" / "cfgs" /
                                  "hunyuan-paint-pbr.yaml")
    # snapshot root; upstream appends "hunyuan3d-paintpbr-v2-1"
    conf.multiview_pretrained_path = str(ctx.weight("hunyuan3d"))
    conf.dino_ckpt_path = str(ctx.weight("dinov2_giant"))
    conf.realesrgan_ckpt_path = str(ctx.weight("realesrgan") /
                                    "RealESRGAN_x4plus.pth")
    conf.texture_size = int(ctx.param("texture_size", 4096))
    pipe = Hunyuan3DPaintPipeline(conf)

    work = ctx.output_path("_work")
    src = Path(ctx.input("mesh"))
    in_mesh = work / ("input" + src.suffix.lower())
    shutil.copy(src, in_mesh)
    obj = work / "textured.obj"
    pipe(mesh_path=str(in_mesh), image_path=image, output_mesh_path=str(obj),
         use_remesh=bool(ctx.param("remesh", True)), save_glb=False)
    out = ctx.output_path("mesh", "mesh.glb")
    stats = _obj_to_glb(obj, out, trimesh)
    shutil.rmtree(work, ignore_errors=True)  # intermediates (obj/maps) are in the glb
    ctx.set_output("mesh", out)
    ctx.metadata.update(stats)


def _obj_to_glb(obj: Path, glb: Path, trimesh) -> dict:
    """Pack the upstream OBJ + albedo/metallic/roughness maps into a glb.

    Upstream uses Blender (bpy) for this; we build the glTF PBR material
    directly: metallicRoughnessTexture has roughness in G, metallic in B.
    """
    from PIL import Image

    base = obj.with_suffix("")
    albedo = Path(str(base) + ".jpg")
    if not albedo.exists():
        albedo = Path(str(base) + ".png")
    metal = Path(str(base) + "_metallic.jpg")
    rough = Path(str(base) + "_roughness.jpg")
    if not albedo.exists():
        raise NodeError(f"paint produced no albedo texture next to {obj}")
    mesh = trimesh.load(str(obj), force="mesh", process=False)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise NodeError("textured OBJ has no UVs")
    kw = {}
    if metal.exists() and rough.exists():
        m = np.asarray(Image.open(metal).convert("L"))
        r = np.asarray(Image.open(rough).convert("L"))
        mr = np.stack([np.zeros_like(r), r, m], -1)
        kw = dict(metallicRoughnessTexture=Image.fromarray(mr),
                  metallicFactor=1.0, roughnessFactor=1.0)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.open(albedo).convert("RGB"), **kw)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.export(str(glb))
    stats = _mesh_stats(mesh)
    stats["pbr_metallic_roughness"] = bool(kw)
    return stats


if __name__ == "__main__":
    raise SystemExit(main({"rembg": rembg, "shape": shape, "paint": paint}))
