"""Type registry: format spec, validation summary and comparisons."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg")


class TypeError_(ValueError):
    """A file does not match its declared type."""


@dataclass
class TypeSpec:
    name: str
    kind: str  # "file" | "dir"
    format: str
    validate: Callable[[Path], Dict[str, Any]]


TYPES: Dict[str, TypeSpec] = {}


def _register(name: str, kind: str, fmt: str):
    def deco(fn):
        TYPES[name] = TypeSpec(name, kind, fmt, fn)
        return fn

    return deco


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise TypeError_(msg)


def _load_json(path: Path) -> Any:
    _need(path.is_file(), f"{path} is not a file")
    with open(path) as f:
        return json.load(f)


def _frames(path: Path, exts) -> List[Path]:
    _need(path.is_dir(), f"{path} is not a directory")
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    _need(bool(files), f"{path} has no {exts} frames")
    return files


def _image_size(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size, im.mode


def _check_T(T: Any, where: str) -> None:
    arr = np.asarray(T, dtype=float)
    _need(arr.shape == (4, 4), f"{where}: expected 4x4, got {arr.shape}")
    _need(np.allclose(arr[3], [0, 0, 0, 1], atol=1e-6),
          f"{where}: last row must be [0,0,0,1]")
    R = arr[:3, :3]
    _need(np.allclose(R @ R.T, np.eye(3), atol=1e-3),
          f"{where}: rotation is not orthonormal")


# ---------------------------------------------------------------- images
@_register("image", "file", "png/jpg, RGB uint8 (RGBA png allowed when the task says so)")
def _v_image(path: Path):
    _need(path.suffix.lower() in IMAGE_EXT, f"{path}: not png/jpg")
    (w, h), mode = _image_size(path)
    return {"width": w, "height": h, "mode": mode}


@_register("image_seq", "dir",
           "dir of %06d.png|jpg (RGB uint8), optional meta.json {fps}")
def _v_image_seq(path: Path):
    files = _frames(path, IMAGE_EXT)
    (w, h), _ = _image_size(files[0])
    meta = path / "meta.json"
    fps = _load_json(meta).get("fps") if meta.exists() else None
    return {"count": len(files), "width": w, "height": h, "fps": fps}


# ----------------------------------------------------------------- depth
# Scale contract: a producer that knows the scale of a depth / depth_seq /
# pointcloud output writes a sidecar (<file>.scale.json, or scale.json inside
# a directory) {"scale_status", "units", "source", "note"}. Metric states are
# accepted by inputs that need metres; anything else is refused there.
METRIC_SCALE_STATES = ("metric", "metric_from_input_poses")
SCALE_STATES = METRIC_SCALE_STATES + ("relative", "input_pose_scale",
                                      "input_depth_scale")


def scale_sidecar(path: Path) -> Path:
    path = Path(path)
    return path / "scale.json" if path.is_dir() else \
        path.with_name(path.name + ".scale.json")


def read_scale(path: Path) -> Dict[str, Any]:
    """Declared scale of a geometry output; 'unspecified' without sidecar."""
    side = scale_sidecar(path)
    if not side.exists():
        return {"scale_status": "unspecified"}
    data = _load_json(side)
    _need(data.get("scale_status") in SCALE_STATES,
          f"{side}: scale_status must be one of {SCALE_STATES}")
    return data


def _scale_summary(path: Path) -> Dict[str, Any]:
    d = read_scale(path)
    out = {"scale_status": d["scale_status"]}
    if d.get("source"):
        out["scale_source"] = d["source"]
    return out


def _depth_stats(arr: np.ndarray, where: str):
    _need(arr.ndim == 2, f"{where}: depth must be HxW, got {arr.shape}")
    valid = arr[np.isfinite(arr) & (arr > 0)]
    return {"height": arr.shape[0], "width": arr.shape[1],
            "valid_ratio": float(valid.size / arr.size),
            "min": float(valid.min()) if valid.size else 0.0,
            "max": float(valid.max()) if valid.size else 0.0,
            "median": float(np.median(valid)) if valid.size else 0.0}


@_register("depth", "file", ".npy float32 HxW z-depth, metres, 0 = invalid; "
           "optional sidecar <file>.scale.json declares a non-metric scale "
           "(summary.scale_status)")
def _v_depth(path: Path):
    _need(path.suffix == ".npy", f"{path}: depth must be .npy")
    return {**_depth_stats(np.load(path), str(path)), **_scale_summary(path)}


@_register("depth_seq", "dir", "dir of %06d.npy float32 HxW z-depth, metres; "
           "optional scale.json declares a non-metric scale "
           "(summary.scale_status)")
def _v_depth_seq(path: Path):
    files = _frames(path, (".npy",))
    first = _depth_stats(np.load(files[0]), str(files[0]))
    return {"count": len(files), **first, **_scale_summary(path)}


@_register("pointmap", "file",
           ".npy float32 HxWx3 per-pixel 3D points in metres, OpenCV "
           "camera frame (non-finite = invalid)")
def _v_pointmap(path: Path):
    _need(path.suffix == ".npy", f"{path}: pointmap must be .npy")
    arr = np.load(path)
    _need(arr.ndim == 3 and arr.shape[2] == 3, f"{path}: need HxWx3")
    z = arr[..., 2]
    valid = np.isfinite(z) & (z > 0)
    return {"height": arr.shape[0], "width": arr.shape[1],
            "valid_ratio": float(valid.mean()),
            "median_z": float(np.median(z[valid])) if valid.any() else 0.0}


@_register("normal", "file",
           ".npy float32 HxWx3 unit normals in OpenCV camera frame")
def _v_normal(path: Path):
    return _normal_stats(np.load(path), str(path))


@_register("disparity", "file",
           ".npy float32 HxW relative inverse depth (larger = closer), "
           "unknown scale and shift; NOT metres")
def _v_disparity(path: Path):
    _need(path.suffix == ".npy", f"{path}: disparity must be .npy")
    arr = np.load(path)
    _need(arr.ndim == 2, f"{path}: need HxW")
    return {"height": arr.shape[0], "width": arr.shape[1],
            "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}


@_register("disparity_seq", "dir",
           "dir of %06d.npy relative inverse depth, one scale/shift per clip")
def _v_disparity_seq(path: Path):
    files = _frames(path, (".npy",))
    arr = np.load(files[0])
    return {"count": len(files), "height": arr.shape[0], "width": arr.shape[1]}


def _normal_stats(arr: np.ndarray, where: str):
    _need(arr.ndim == 3 and arr.shape[2] == 3, f"{where}: need HxWx3")
    norm = np.linalg.norm(arr, axis=2)
    valid = norm > 0.5
    mean = arr[valid].mean(axis=0) if valid.any() else np.zeros(3)
    return {"height": arr.shape[0], "width": arr.shape[1],
            "valid_ratio": float(valid.mean()),
            "mean_x": float(mean[0]), "mean_y": float(mean[1]),
            "mean_z": float(mean[2])}


# ----------------------------------------------------------------- masks
def _mask_stats(path: Path):
    from PIL import Image

    arr = np.asarray(Image.open(path))
    _need(arr.ndim == 2, f"{path}: mask must be single channel")
    ids = [int(i) for i in np.unique(arr) if i != 0]
    return arr, {"height": arr.shape[0], "width": arr.shape[1],
                 "instances": len(ids), "ids": ids,
                 "area_ratio": float((arr != 0).mean())}


@_register("mask", "file",
           "png uint8/uint16 single channel; 0 = background, 1..N = "
           "instance id; optional sidecar <name>.json "
           "{id: {label, score}}")
def _v_mask(path: Path):
    _need(path.suffix.lower() == ".png", f"{path}: mask must be png")
    return _mask_stats(path)[1]


@_register("mask_seq", "dir",
           "dir of %06d.png instance-id masks + optional labels.json "
           "{id: {label, score}} (tracked ids) or {\"frames\": {frame: "
           "{id: {...}}}} (per-frame ids)")
def _v_mask_seq(path: Path):
    files = _frames(path, (".png",))
    ids = set()
    nonempty = 0
    for f in files:
        _, s = _mask_stats(f)
        ids.update(s["ids"])
        nonempty += s["instances"] > 0
    return {"count": len(files), "instances": len(ids),
            "ids": sorted(ids), "nonempty_frames": nonempty}


# ------------------------------------------------------------- json types
@_register("bbox_set", "file",
           'json {"boxes": [{"xyxy": [x0,y0,x1,y1] px, "score", '
           '"label", "frame"?}]}')
def _v_bbox_set(path: Path):
    data = _load_json(path)
    boxes = data.get("boxes")
    _need(isinstance(boxes, list), f"{path}: missing boxes list")
    for i, b in enumerate(boxes):
        xyxy = b.get("xyxy")
        _need(isinstance(xyxy, list) and len(xyxy) == 4,
              f"{path}: boxes[{i}].xyxy must be 4 numbers")
        _need(xyxy[2] >= xyxy[0] and xyxy[3] >= xyxy[1],
              f"{path}: boxes[{i}] is not xyxy")
    return {"count": len(boxes),
            "labels": sorted({str(b.get("label")) for b in boxes})}


@_register("prompt_set", "file",
           'json {"points": [{"xy", "positive", "obj_id", "frame"}], '
           '"boxes": [{"xyxy", "obj_id", "frame"}], "text": str}')
def _v_prompt_set(path: Path):
    data = _load_json(path)
    pts, boxes = data.get("points", []), data.get("boxes", [])
    _need(bool(pts or boxes or data.get("text")),
          f"{path}: empty prompt set")
    return {"points": len(pts), "boxes": len(boxes),
            "text": data.get("text")}


@_register("camera", "file",
           'json {"K": 3x3, "width", "height", "dist": [k1,k2,p1,p2,k3]} '
           "(OpenCV pinhole)")
def _v_camera(path: Path):
    data = _load_json(path)
    K = np.asarray(data.get("K"), dtype=float)
    _need(K.shape == (3, 3), f"{path}: K must be 3x3")
    _need("width" in data and "height" in data, f"{path}: need width/height")
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "width": data["width"], "height": data["height"]}


@_register("camera_seq", "file",
           'json {"cameras": [{"K": 3x3, "width", "height", "dist"}, ...]} '
           "one OpenCV pinhole camera per frame")
def _v_camera_seq(path: Path):
    cams = _load_json(path).get("cameras")
    _need(isinstance(cams, list) and cams, f"{path}: missing cameras list")
    for i, c in enumerate(cams):
        K = np.asarray(c.get("K"), dtype=float)
        _need(K.shape == (3, 3), f"{path}: cameras[{i}].K must be 3x3")
    K0 = np.asarray(cams[0]["K"], dtype=float)
    return {"count": len(cams), "fx": float(K0[0, 0]),
            "fy": float(K0[1, 1])}


@_register("pose_set", "file",
           'json {"poses": [{"T_cam_obj": 4x4 (metres), "label", '
           '"score"}]}')
def _v_pose_set(path: Path):
    poses = _load_json(path).get("poses")
    _need(isinstance(poses, list), f"{path}: missing poses list")
    for i, p in enumerate(poses):
        _check_T(p.get("T_cam_obj"), f"{path}: poses[{i}]")
    out = {"count": len(poses),
           "labels": sorted({str(p.get("label")) for p in poses})}
    if poses:
        t = np.asarray(poses[0]["T_cam_obj"], dtype=float)[:3, 3]
        out.update({"z0": float(t[2]), "dist0": float(np.linalg.norm(t))})
        if poses[0].get("size") is not None:
            out["size0"] = [float(x) for x in poses[0]["size"]]
    return out


@_register("pose_seq", "file",
           'json {"frames": [{"frame": int, "poses": [{"T_cam_obj", '
           '"label"}]}]}')
def _v_pose_seq(path: Path):
    frames = _load_json(path).get("frames")
    _need(isinstance(frames, list), f"{path}: missing frames list")
    for f in frames:
        for p in f.get("poses", []):
            _check_T(p.get("T_cam_obj"), f"{path}: frame {f.get('frame')}")
    return {"count": len(frames)}


@_register("trajectory", "file",
           'json {"T_world_cam": [4x4, ...], "frame_index"?: [int], '
           '"timestamps"?: [...]} (camera-to-world; frame_index = input '
           "frame of each pose when not every frame has one)")
def _v_trajectory(path: Path):
    Ts = _load_json(path).get("T_world_cam")
    _need(isinstance(Ts, list) and Ts, f"{path}: missing T_world_cam")
    for i, T in enumerate(Ts):
        _check_T(T, f"{path}: T_world_cam[{i}]")
    return {"count": len(Ts)}


@_register("tracks", "file",
           ".npz with xy float32 [T,N,2] (px) and visible bool [T,N]")
def _v_tracks(path: Path):
    data = np.load(path)
    xy, vis = data["xy"], data["visible"]
    _need(xy.ndim == 3 and xy.shape[2] == 2, f"{path}: xy must be [T,N,2]")
    _need(vis.shape == xy.shape[:2], f"{path}: visible must be [T,N]")
    return {"frames": xy.shape[0], "points": xy.shape[1],
            "visible_ratio": float(vis.mean())}


# ------------------------------------------------------------- 3D assets
def _ply_vertices(path: Path) -> int:
    with open(path, "rb") as f:
        _need(f.readline().strip() == b"ply", f"{path}: not a ply file")
        for _ in range(200):
            line = f.readline()
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.strip() == b"end_header":
                break
    raise TypeError_(f"{path}: ply without vertex element")


def _glb_stats(path: Path) -> Dict[str, Any]:
    """Vertex/face counts and bounds from the glTF JSON chunk."""
    import struct

    with open(path, "rb") as f:
        _need(f.read(4) == b"glTF", f"{path}: bad glb magic")
        f.read(8)
        length, ctype = struct.unpack("<I4s", f.read(8))
        _need(ctype == b"JSON", f"{path}: first chunk is not JSON")
        gltf = json.loads(f.read(length))
    acc = gltf.get("accessors", [])
    verts = faces = 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None:
                a = acc[pos]
                verts += a.get("count", 0)
                if "min" in a and "max" in a:
                    lo = np.minimum(lo, a["min"])
                    hi = np.maximum(hi, a["max"])
            if prim.get("indices") is not None:
                faces += acc[prim["indices"]].get("count", 0) // 3
    out: Dict[str, Any] = {"vertices": int(verts), "faces": int(faces),
                           "textured": bool(gltf.get("textures"))}
    if np.isfinite(lo).all():
        out["extent"] = [float(x) for x in (hi - lo)]
        out["max_extent"] = float((hi - lo).max())
    return out


@_register("mesh", "file", "glb (preferred) / obj / ply, metres")
def _v_mesh(path: Path):
    ext = path.suffix.lower()
    _need(ext in (".glb", ".obj", ".ply"), f"{path}: mesh must be glb/obj/ply")
    _need(path.stat().st_size > 0, f"{path}: empty file")
    out = {"format": ext[1:], "bytes": path.stat().st_size}
    if ext == ".glb":
        out.update(_glb_stats(path))
    if ext == ".ply":
        out["vertices"] = _ply_vertices(path)
    return out


@_register("pointcloud", "file", "ply, metres, optional rgb; optional sidecar <file>.scale.json declares a non-metric scale")
def _v_pointcloud(path: Path):
    return {"points": _ply_vertices(path), **_scale_summary(path)}


@_register("gaussians", "file", "3DGS ply (INRIA layout), metres")
def _v_gaussians(path: Path):
    return {"gaussians": _ply_vertices(path)}


# --------------------------------------------------------------- generic
@_register("text", "file", "utf-8 text")
def _v_text(path: Path):
    text = path.read_text(encoding="utf-8")
    return {"chars": len(text), "text": text[:2000]}


@_register("json", "file", "json; the task description states its schema")
def _v_json(path: Path):
    data = _load_json(path)
    keys = sorted(data) if isinstance(data, dict) else []
    return {"keys": keys, "length": len(data)
            if isinstance(data, (list, dict)) else None}


@_register("file", "file", "opaque file; format stated by the task")
def _v_file(path: Path):
    _need(path.is_file(), f"{path} is not a file")
    return {"bytes": path.stat().st_size}


@_register("dir", "dir", "opaque directory; layout stated by the task")
def _v_dir(path: Path):
    _need(path.is_dir(), f"{path} is not a directory")
    return {"entries": len(list(path.iterdir()))}


# ------------------------------------------------------------------- API
def validate(type_name: str, path: str | Path) -> Dict[str, Any]:
    """Validate ``path`` against ``type_name``; return a summary dict."""
    spec = TYPES[type_name]
    path = Path(path)
    if not path.exists():
        raise TypeError_(f"{type_name}: {path} does not exist")
    return spec.validate(path)


def _mask_iou(a: Path, b: Path) -> float:
    from PIL import Image

    def load(p):
        files = _frames(p, (".png",)) if p.is_dir() else [p]
        return np.stack([np.asarray(Image.open(f)) != 0 for f in files])

    x, y = load(a), load(b)
    _need(x.shape == y.shape, f"mask shapes differ {x.shape} vs {y.shape}")
    union = np.logical_or(x, y).sum()
    return float(np.logical_and(x, y).sum() / union) if union else 1.0


def _npy_pairs(a: Path, b: Path):
    if a.is_dir():
        fa, fb = _frames(a, (".npy",)), _frames(b, (".npy",))
        _need(len(fa) == len(fb), f"frame count differs {len(fa)} vs {len(fb)}")
        return [(np.load(x).astype(np.float64), np.load(y).astype(np.float64))
                for x, y in zip(fa, fb)]
    return [(np.load(a).astype(np.float64), np.load(b).astype(np.float64))]


def _depth_abs_rel(a: Path, b: Path) -> float:
    """Mean abs-rel depth error over all frames (file or depth_seq dir)."""
    errs = []
    for pred, gt in _npy_pairs(a, b):
        valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
        errs.append(np.abs(pred[valid] - gt[valid]) / gt[valid])
    return float(np.mean(np.concatenate(errs)))


def _disparity_rel(a: Path, b: Path) -> float:
    """Abs-rel error after least-squares scale/shift alignment."""
    errs = []
    for pred, gt in _npy_pairs(a, b):
        valid = np.isfinite(pred) & np.isfinite(gt) & (np.abs(gt) > 1e-6)
        x, y = pred[valid].ravel(), gt[valid].ravel()
        A = np.stack([x, np.ones_like(x)], 1)
        s_, t_ = np.linalg.lstsq(A, y, rcond=None)[0]
        errs.append(np.abs(s_ * x + t_ - y) / np.abs(y))
    return float(np.mean(np.concatenate(errs)))


def _normal_angle(a: Path, b: Path) -> float:
    """Mean angular error in degrees between two normal maps."""
    x, y = np.load(a), np.load(b)
    _need(x.shape == y.shape, "normal shapes differ")
    nx = np.linalg.norm(x, axis=2)
    ny = np.linalg.norm(y, axis=2)
    valid = (nx > 0.5) & (ny > 0.5)
    cos = (x[valid] * y[valid]).sum(1) / (nx[valid] * ny[valid])
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def _pose_rot_err(a: Path, b: Path) -> float:
    """Max rotation error (degrees) between matching poses."""
    pa, pb = _load_json(a)["poses"], _load_json(b)["poses"]
    _need(len(pa) == len(pb), "pose count differs")
    errs = [0.0]
    for x, y in zip(pa, pb):
        R = (np.asarray(x["T_cam_obj"])[:3, :3].T
             @ np.asarray(y["T_cam_obj"])[:3, :3])
        errs.append(float(np.degrees(np.arccos(
            np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    return max(errs)


def _box_iou(p, q) -> float:
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1])
             - inter)
    return inter / union if union > 0 else 0.0


def _bbox_iou(a: Path, b: Path) -> float:
    """Mean best-match IoU of reference boxes (label must match)."""
    pred = _load_json(a)["boxes"]
    ref = _load_json(b)["boxes"]
    if not ref:
        return 1.0 if not pred else 0.0
    scores = []
    for r in ref:
        cands = [p for p in pred if str(p.get("label")) == str(r.get("label"))]
        scores.append(max((_box_iou(p["xyxy"], r["xyxy"]) for p in cands),
                          default=0.0))
    return float(np.mean(scores))


def _mask_class_iou(a: Path, b: Path) -> float:
    """Mean IoU over ids present in either mask (label-aware)."""
    from PIL import Image

    def load(p):
        files = _frames(p, (".png",)) if p.is_dir() else [p]
        return np.stack([np.asarray(Image.open(f)) for f in files])

    x, y = load(a), load(b)
    _need(x.shape == y.shape, f"mask shapes differ {x.shape} vs {y.shape}")
    ids = sorted((set(np.unique(x)) | set(np.unique(y))) - {0})
    if not ids:
        return 1.0
    ious = []
    for i in ids:
        u = np.logical_or(x == i, y == i).sum()
        ious.append(np.logical_and(x == i, y == i).sum() / u)
    return float(np.mean(ious))


def _tracks_epe(a: Path, b: Path) -> float:
    """Mean end-point error (px) over points visible in the reference."""
    x, y = np.load(a), np.load(b)
    _need(x["xy"].shape == y["xy"].shape, "track shapes differ")
    vis = y["visible"].astype(bool)
    return float(np.linalg.norm(x["xy"][vis] - y["xy"][vis], axis=-1).mean())


def _trajectory_ate(a: Path, b: Path) -> float:
    """RMSE of camera centres after similarity (Umeyama) alignment."""
    da, db = _load_json(a), _load_json(b)
    P = np.asarray(da["T_world_cam"], dtype=float)[:, :3, 3]
    Q = np.asarray(db["T_world_cam"], dtype=float)[:, :3, 3]
    if "frame_index" in da and "frame_index" in db:
        # match by frame index; a few untracked frames are tolerated
        ia = {int(f): i for i, f in enumerate(da["frame_index"])}
        ib = {int(f): i for i, f in enumerate(db["frame_index"])}
        common = sorted(set(ia) & set(ib))
        _need(len(common) >= max(3, int(0.9 * len(ib))),
              f"only {len(common)}/{len(ib)} reference frames matched")
        P = P[[ia[f] for f in common]]
        Q = Q[[ib[f] for f in common]]
    _need(P.shape == Q.shape, "trajectory length differs")
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    if np.allclose(X, 0) or np.allclose(Y, 0):
        return float(np.sqrt(((X - Y) ** 2).sum(1).mean()))
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (X ** 2).sum(1).mean()
    aligned = scale * (R @ X.T).T + mq
    return float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))


def _image_psnr(a: Path, b: Path) -> float:
    """Mean PSNR (dB) over frames (image file or image_seq dir)."""
    from PIL import Image

    def load(p):
        files = _frames(p, IMAGE_EXT) if p.is_dir() else [p]
        return [np.asarray(Image.open(f).convert("RGB"), dtype=np.float64)
                for f in files]

    xs, ys = load(a), load(b)
    _need(len(xs) == len(ys), f"frame count differs {len(xs)} vs {len(ys)}")
    out = []
    for x, y in zip(xs, ys):
        _need(x.shape == y.shape, f"image shapes differ {x.shape} {y.shape}")
        mse = float(np.mean((x - y) ** 2))
        out.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(out))


def _pose_err(a: Path, b: Path) -> float:
    """Max translation error (metres) between matching poses."""
    pa = _load_json(a)["poses"]
    pb = _load_json(b)["poses"]
    _need(len(pa) == len(pb), "pose count differs")
    return max(
        float(np.linalg.norm(np.asarray(x["T_cam_obj"])[:3, 3]
                             - np.asarray(y["T_cam_obj"])[:3, 3]))
        for x, y in zip(pa, pb)
    ) if pa else 0.0


COMPARATORS: Dict[str, Callable[[Path, Path], float]] = {
    "mask_iou": _mask_iou,
    "mask_class_iou": _mask_class_iou,
    "depth_abs_rel": _depth_abs_rel,
    "disparity_rel": _disparity_rel,
    "normal_angle_deg": _normal_angle,
    "pose_translation_err": _pose_err,
    "pose_rotation_err_deg": _pose_rot_err,
    "bbox_iou": _bbox_iou,
    "image_psnr": _image_psnr,
    "tracks_epe": _tracks_epe,
    "trajectory_ate": _trajectory_ate,
}


def compare(metric: str, output: str | Path,
            expected: str | Path) -> float:
    """Compute a comparison metric between an output and a reference."""
    if metric not in COMPARATORS:
        raise ValueError(f"unknown metric {metric}; known {list(COMPARATORS)}")
    return COMPARATORS[metric](Path(output), Path(expected))


def describe_types(names: Optional[List[str]] = None) -> str:
    names = names or sorted(TYPES)
    return "\n".join(
        f"- {n} ({TYPES[n].kind}): {TYPES[n].format}" for n in names
    )
