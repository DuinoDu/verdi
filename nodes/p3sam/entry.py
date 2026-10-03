"""P3-SAM (Hunyuan3D-Part) node: automatic 3D part segmentation of a mesh."""
from __future__ import annotations

import random
import sys
import tempfile

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _import_automask(ctx: Context):
    """Import upstream AutoMask with Sonata loaded from the pinned weight."""
    p3 = ctx.repo / "P3-SAM"
    for p in (str(p3), str(ctx.repo / "XPart" / "partgen")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import sonata  # XPart/partgen/models/sonata

    sonata_ckpt = ctx.weight("sonata") / "sonata.pth"
    orig_load = sonata.load

    def _local_load(name="sonata", repo_id=None, download_root=None, **kw):
        # upstream calls sonata.load("sonata", repo_id=..., download_root=~/.cache)
        # which downloads from HF at run time; use the pinned local file instead
        return orig_load(str(sonata_ckpt), **kw)

    sonata.load = _local_load
    from demo.auto_mask import AutoMask

    return AutoMask


def _load_mesh(path):
    import trimesh

    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as exc:  # noqa: BLE001
        raise NodeError(f"cannot load mesh {path}: {exc}",
                        hint="input must be a glb/obj/ply triangle mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise NodeError(f"{path} contains no triangle faces",
                        hint="P3-SAM needs a surface mesh, not a point cloud")
    return mesh


def _tab20(n_parts: int):
    from matplotlib import colormaps

    cmap = colormaps["tab20"].resampled(max(n_parts, 20))
    return [(np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
            for i in range(n_parts)]


def segment(ctx: Context) -> None:
    import torch

    if ctx.device != "cuda":
        raise NodeError("P3-SAM needs a CUDA GPU (spconv + flash-attn)",
                        hint="run with device=cuda")
    AutoMask = _import_automask(ctx)
    mesh = _load_mesh(ctx.input("mesh"))
    n_in = len(mesh.faces)
    seed = int(ctx.param("seed", 42))
    clean = bool(ctx.param("clean_mesh", False))

    ckpt = ctx.weight("p3sam") / "p3sam" / "p3sam.safetensors"
    auto_mask = AutoMask(ckpt_path=str(ckpt))
    _seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        aabb, face_ids, mesh = auto_mask.predict_aabb(
            mesh,
            save_path=tmp,
            point_num=int(ctx.param("point_num", 100000)),
            prompt_num=int(ctx.param("prompt_num", 400)),
            prompt_bs=int(ctx.param("prompt_bs", 8)),
            threshold=float(ctx.param("threshold", 0.85)),
            post_process=bool(ctx.param("post_process", True)),
            seed=seed,
            save_mid_res=False,
            show_info=True,
            clean_mesh_flag=clean,
            is_parallel=False,
        )
    auto_mask.release()
    torch.cuda.empty_cache()

    face_ids = np.asarray(face_ids).astype(np.int64).reshape(-1)
    if face_ids.shape[0] != len(mesh.faces):
        raise NodeError(f"P3-SAM returned {face_ids.shape[0]} labels for "
                        f"{len(mesh.faces)} faces")
    # renumber parts to 0..K-1 ordered by decreasing area; -1 stays unassigned
    areas = mesh.area_faces
    ids = [int(i) for i in np.unique(face_ids) if i >= 0]
    if not ids:
        raise NodeError("P3-SAM found no parts",
                        hint="check the mesh is a closed object surface; "
                        "try a lower threshold")
    ids.sort(key=lambda i: -float(areas[face_ids == i].sum()))
    remap = {old: new for new, old in enumerate(ids)}
    labels = np.full_like(face_ids, -1)
    for old, new in remap.items():
        labels[face_ids == old] = new

    total_area = float(areas.sum())
    parts = []
    for k in range(len(ids)):
        sel = labels == k
        verts = mesh.vertices[np.unique(mesh.faces[sel])]
        parts.append({
            "id": k,
            "num_faces": int(sel.sum()),
            "area_frac": round(float(areas[sel].sum()) / total_area, 6),
            "aabb": [verts.min(0).round(6).tolist(),
                     verts.max(0).round(6).tolist()],
        })

    colors = _tab20(len(ids))
    face_colors = np.zeros((len(mesh.faces), 4), np.uint8)
    face_colors[:, :3] = 128   # unassigned faces: grey
    face_colors[:, 3] = 255
    for k in range(len(ids)):
        face_colors[labels == k, :3] = colors[k]
    import trimesh

    out_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                               faces=np.asarray(mesh.faces), process=False)
    out_mesh.visual.face_colors = face_colors
    mesh_path = ctx.output_path("mesh", "parts.glb")
    out_mesh.export(str(mesh_path))
    ctx.set_output("mesh", mesh_path)

    data = {
        "num_faces": int(len(mesh.faces)),
        "input_faces": int(n_in),
        "num_parts": len(ids),
        "unassigned_faces": int((labels < 0).sum()),
        "parts": parts,
        "part_colors": [c.tolist() for c in colors],
        "face_labels": labels.tolist(),
    }
    json_path = ctx.output_path("parts", "parts.json")
    io.write_json(json_path, data)
    ctx.set_output("parts", json_path)
    ctx.metadata.update({"num_parts": len(ids), "num_faces": len(mesh.faces)})


if __name__ == "__main__":
    raise SystemExit(main({"segment": segment}))
