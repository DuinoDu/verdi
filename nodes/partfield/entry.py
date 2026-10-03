"""PartField node: per-face part features + agglomerative part segmentation."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def _load_mesh(path):
    import trimesh

    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception as exc:  # noqa: BLE001
        raise NodeError(f"cannot load mesh {path}: {exc}",
                        hint="input must be a glb/obj/ply triangle mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise NodeError(f"{path} contains no triangle faces",
                        hint="PartField mesh mode needs a surface mesh")
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                           faces=np.asarray(mesh.faces), process=False)


def _tab20(n: int):
    from matplotlib import colormaps

    cmap = colormaps["tab20"].resampled(max(n, 20))
    return [(np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
            for i in range(n)]


def _extract_features(ctx: Context, mesh_file: Path, work: Path) -> np.ndarray:
    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from partfield.config import default_argument_parser, setup
    from partfield_inference import predict

    feat_dir = work / "feat"
    feat_dir.mkdir()
    ckpt = ctx.weight("partfield") / "model_objaverse.ckpt"
    args = default_argument_parser().parse_args([
        "-c", os.path.join(repo, "configs", "final", "demo.yaml"),
        "--opts",
        "continue_ckpt", str(ckpt),
        "result_name", "pdebug",
        "dataset.data_path", str(mesh_file.parent),
        "feature_output_dir", str(feat_dir),
        "n_point_per_face", str(int(ctx.param("n_point_per_face", 1000))),
    ])
    cfg = setup(args, freeze=False)
    cwd = os.getcwd()
    os.chdir(work)  # upstream writes exp_results/<result_name> relative to cwd
    try:
        predict(cfg)
    finally:
        os.chdir(cwd)
    uid = mesh_file.stem
    for name in (f"part_feat_{uid}_0_batch.npy", f"part_feat_{uid}_0.npy"):
        if (feat_dir / name).exists():
            return np.load(feat_dir / name)
    raise NodeError(f"PartField wrote no features (dir has "
                    f"{sorted(os.listdir(feat_dir))})")


def segment(ctx: Context) -> None:
    if ctx.device != "cuda":
        raise NodeError("PartField needs a CUDA GPU", hint="run with device=cuda")
    import trimesh

    k = int(ctx.param("num_clusters", 10))
    if k < 1:
        raise NodeError("num_clusters must be >= 1")
    mesh = _load_mesh(ctx.input("mesh"))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "data").mkdir()
        mesh_file = work / "data" / "mesh.glb"
        mesh.export(str(mesh_file))
        # reload exactly what upstream reads so features align with faces
        mesh = _load_mesh(mesh_file)
        feats = _extract_features(ctx, mesh_file, work)
    F = len(mesh.faces)
    if feats.shape[0] != F:
        raise NodeError(f"PartField returned {feats.shape[0]} face features "
                        f"for {F} faces")
    if k > F:
        raise NodeError(f"num_clusters={k} > number of faces {F}")

    from run_part_clustering import construct_face_adjacency_matrix_facemst
    from sklearn.cluster import AgglomerativeClustering

    # exactly SimFoundry segment_mesh_partfield.py: L2-normalised features,
    # face adjacency (+MST between components, +kNN), agglomerative clustering
    normed = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
    adj = construct_face_adjacency_matrix_facemst(
        np.asarray(mesh.faces), np.asarray(mesh.vertices), with_knn=True)
    raw = AgglomerativeClustering(n_clusters=k, connectivity=adj).fit(
        normed).labels_.astype(np.int64)

    areas = mesh.area_faces
    ids = sorted({int(i) for i in np.unique(raw)},
                 key=lambda i: -float(areas[raw == i].sum()))
    labels = np.empty_like(raw)
    for new, old in enumerate(ids):
        labels[raw == old] = new
    total = float(areas.sum())
    parts = []
    for p in range(len(ids)):
        sel = labels == p
        verts = mesh.vertices[np.unique(mesh.faces[sel])]
        parts.append({"id": p, "num_faces": int(sel.sum()),
                      "area_frac": round(float(areas[sel].sum()) / total, 6),
                      "aabb": [verts.min(0).round(6).tolist(),
                               verts.max(0).round(6).tolist()]})
    colors = _tab20(len(ids))
    fc = np.zeros((F, 4), np.uint8)
    fc[:, 3] = 255
    for p in range(len(ids)):
        fc[labels == p, :3] = colors[p]
    out = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                          process=False)
    out.visual.face_colors = fc
    mesh_path = ctx.output_path("mesh", "parts.glb")
    out.export(str(mesh_path))
    ctx.set_output("mesh", mesh_path)

    feat_path = ctx.output_path("features", "face_features.npy")
    np.save(feat_path, feats.astype(np.float32))
    ctx.set_output("features", feat_path)

    json_path = ctx.output_path("parts", "parts.json")
    io.write_json(json_path, {
        "num_faces": F, "num_parts": len(ids), "unassigned_faces": 0,
        "parts": parts, "part_colors": [c.tolist() for c in colors],
        "face_labels": labels.tolist()})
    ctx.set_output("parts", json_path)
    ctx.metadata.update({"num_parts": len(ids), "num_faces": F,
                         "feature_dim": int(feats.shape[1])})


if __name__ == "__main__":
    raise SystemExit(main({"segment": segment}))
