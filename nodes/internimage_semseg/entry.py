"""FlashInternImage (DCNv4) ADE20K semantic segmentation node."""
from __future__ import annotations

import sys
import types

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

CKPT = {"t": "upernet_flash_internimage_t_512_160k_ade20k",
        "s": "upernet_flash_internimage_s_512_160k_ade20k",
        "b": "upernet_flash_internimage_b_512_160k_ade20k",
        "l": "upernet_flash_internimage_l_640_160k_ade20k"}


# DCNv4 @4b848f7 flash_deform_attn_func.py raises NotImplementedError at
# import for GPUs missing from its shared-memory table (no sm_120 entry).
# The value only sizes FlashDeformAttn blocks (unused by UPerNet); sm_120
# has the same 99 KB opt-in shared memory per block as sm_86/sm_89, so it
# gets the same budget (99000). Applied as a source rewrite at import time
# because DCNv4 is an installed uv git dependency, not the [upstream] tree.
_DCNV4_SHM_EXTRA = '"12.0": 99000, '


def _patch_dcnv4_shm_table() -> None:
    import importlib.abc
    import importlib.machinery
    import importlib.util

    target = "DCNv4.functions.flash_deform_attn_func"

    class _Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            src = self.get_data(self.path).decode()
            if "shm_size_dict = {" not in src:
                raise NodeError("DCNv4 shm table patch no longer applies",
                                hint="check the DCNv4 commit in pyproject.toml")
            src = src.replace("shm_size_dict = {",
                              "shm_size_dict = {" + _DCNV4_SHM_EXTRA, 1)
            return compile(src, self.path, "exec", dont_inherit=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_mod=None):
            if fullname != target:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or not spec.origin:
                return None
            return importlib.util.spec_from_file_location(
                fullname, spec.origin, loader=_Loader(fullname, spec.origin))

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def _shim_mmcv_ops() -> None:
    """mmcv 1.7.2 is installed without its CUDA ops (mmcv._ext).

    mmseg / mmseg_custom import a few ops eagerly; the UPerNet path never
    calls them. Provide exact pure-torch versions of point_sample and
    sigmoid_focal_loss, and stubs that raise for the rest.
    """
    try:
        import mmcv.ops  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "mmcv._ext":
            raise
    import torch.nn as nn
    import torch.nn.functional as F

    ops = types.ModuleType("mmcv.ops")

    def point_sample(input, points, align_corners=False, **kw):
        add_dim = points.dim() == 3
        if add_dim:
            points = points.unsqueeze(2)
        out = F.grid_sample(input, points * 2.0 - 1.0,
                            align_corners=align_corners, **kw)
        return out.squeeze(3) if add_dim else out

    def sigmoid_focal_loss(*a, **k):
        raise RuntimeError("sigmoid_focal_loss needs mmcv-full (training)")

    def _stub(name):
        class _Op(nn.Module):
            def __init__(self, *a, **k):
                raise RuntimeError(f"{name} needs mmcv-full CUDA ops; not "
                                   "used by the UPerNet models of this node")
        _Op.__name__ = name
        return _Op

    ops.point_sample = point_sample
    ops.sigmoid_focal_loss = sigmoid_focal_loss
    for name in ("CrissCrossAttention", "PSAMask",
                 "MultiScaleDeformableAttention", "DeformConv2d",
                 "ModulatedDeformConv2d", "RoIAlign", "nms"):
        setattr(ops, name, _stub(name))
    ops.get_onnxruntime_op_path = lambda: ""
    msda = types.ModuleType("mmcv.ops.multi_scale_deform_attn")
    msda.MultiScaleDeformableAttention = ops.MultiScaleDeformableAttention
    cmsda = types.ModuleType("mmseg_custom.models.decode_heads.msda")
    cmsda.CustomMultiScaleDeformableAttention = \
        ops.MultiScaleDeformableAttention
    sys.modules["mmcv.ops"] = ops
    sys.modules["mmcv.ops.multi_scale_deform_attn"] = msda
    sys.modules["mmseg_custom.models.decode_heads.msda"] = cmsda


def _classes():
    from mmseg.core import get_classes

    return [c.strip() for c in get_classes("ade20k")]


def _class_ids(spec: str) -> list[int]:
    names = _classes()
    ids = []
    for n in [s.strip().lower() for s in (spec or "").split(",") if s.strip()]:
        if n not in names:
            raise NodeError(f"unknown ADE20K class {n!r}",
                            hint="valid names: " + ", ".join(names),
                            kind="request")
        ids.append(names.index(n) + 1)
    return ids


class Segmentor:
    def __init__(self, ctx: Context):
        import torch

        sys.path.insert(0, str(ctx.repo / "segmentation"))
        _shim_mmcv_ops()
        _patch_dcnv4_shm_table()
        import mmcv_custom  # noqa: F401  (registers optimizers)
        import mmseg_custom  # noqa: F401  (registers FlashInternImage)
        from mmcv.runner import load_checkpoint
        from mmseg.apis import inference_segmentor, init_segmentor

        key = ctx.param("model", "s")
        name = CKPT[key]
        cfg = ctx.repo / "segmentation/configs/ade20k" / f"{name}.py"
        dev = "cuda:0" if ctx.device == "cuda" else "cpu"
        self.model = init_segmentor(str(cfg), checkpoint=None, device=dev)
        # checkpoints carry mmcv meta (not weights_only safe); pinned source
        load_checkpoint(self.model, str(ctx.weight(key) / f"{name}.pth"),
                        map_location="cpu",
                        revise_keys=[(r"^module\.", "")])
        self.model.eval()
        self._infer = inference_segmentor
        self._torch = torch

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            res = self._infer(self.model, np.ascontiguousarray(rgb[..., ::-1]))
        return np.asarray(res[0]).astype(np.int32) + 1  # ids 1..150


def _filter(ids: np.ndarray, keep: list[int]) -> np.ndarray:
    return np.where(np.isin(ids, keep), ids, 0) if keep else ids


def _labels(ids: np.ndarray) -> dict:
    names = _classes()
    vals, counts = np.unique(ids, return_counts=True)
    return {int(v): {"label": names[v - 1],
                     "area_ratio": round(float(c) / ids.size, 5)}
            for v, c in zip(vals, counts) if v > 0}


def _patch_torch_load():
    """mmcv 1.x calls torch.load without weights_only; torch>=2.6 defaults
    to weights_only=True and rejects the mmcv checkpoint meta."""
    import torch

    orig = torch.load

    def load(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = load


def semseg(ctx: Context) -> None:
    _patch_torch_load()
    keep = _class_ids(ctx.param("classes", ""))
    seg = Segmentor(ctx)
    ids = _filter(seg(io.read_image(ctx.input("image"))), keep)
    labels = _labels(ids)
    ctx.log("classes: " + ", ".join(v["label"] for v in labels.values()))
    out = ctx.output_path("mask", "mask.png")
    io.write_mask(out, ids, labels)
    ctx.set_output("mask", out)


def semseg_frames(ctx: Context) -> None:
    _patch_torch_load()
    keep = _class_ids(ctx.param("classes", ""))
    seg = Segmentor(ctx)
    out_dir = ctx.output_path("masks")
    seen = {}
    for i, f in enumerate(io.list_frames(ctx.input("frames"))):
        ids = _filter(seg(io.read_image(f)), keep)
        io.write_mask(out_dir / io.frame_name(i), ids)
        for k, v in _labels(ids).items():
            seen[k] = {"label": v["label"]}
    io.write_json(out_dir / "labels.json",
                  {str(k): v for k, v in sorted(seen.items())})
    ctx.set_output("masks", out_dir)


def remove_dynamic(ctx: Context) -> None:
    from PIL import Image

    _patch_torch_load()
    remove = _class_ids(ctx.param("classes", "person,car"))
    if not remove:
        raise NodeError("classes is empty", kind="request")
    rgb = io.read_image(ctx.input("image")).copy()
    if ctx.has_input("mask"):
        ids = np.asarray(Image.open(ctx.input("mask"))).astype(np.int32)
        if ids.shape != rgb.shape[:2]:
            raise NodeError(f"mask {ids.shape} and image {rgb.shape[:2]} "
                            "differ in size", kind="request")
    else:
        ids = Segmentor(ctx)(rgb)
    removed = np.zeros(ids.shape, np.int32)
    for cid in remove:
        sel = ids == cid
        if sel.any():
            rgb[sel] = rgb[sel].mean(0).astype(np.uint8)
            removed[sel] = cid
    labels = _labels(removed)
    ctx.log("removed: " + (", ".join(
        f"{v['label']} {v['area_ratio']:.3f}" for v in labels.values())
        or "nothing"))
    ipath = ctx.output_path("image", "image.png")
    io.write_image(ipath, rgb)
    mpath = ctx.output_path("mask", "removed.png")
    io.write_mask(mpath, removed, labels)
    ctx.set_output("image", ipath)
    ctx.set_output("mask", mpath)


if __name__ == "__main__":
    raise SystemExit(main({"semseg": semseg, "semseg_frames": semseg_frames,
                           "remove_dynamic": remove_dynamic}))
