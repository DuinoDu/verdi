"""plane_layout node: RANSAC plane extraction (Open3D, CPU), gravity-aware
labelling (floor / wall / ceiling / table / other) and a floor-anchored
world frame (z up = against gravity, origin on the floor)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

COLORS = {"floor": (200, 160, 60), "wall": (70, 130, 220),
          "ceiling": (160, 160, 160), "table": (220, 60, 60),
          "other": (90, 200, 90), "none": (40, 40, 40)}


# ------------------------------------------------------------------ input
def _undistort_norm(u, v, K, dist):
    """Pixel -> normalized undistorted coords (iterative OpenCV model)."""
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    d = np.zeros(5)
    d[:min(5, len(dist))] = dist[:5]
    if not np.any(d):
        return x, y
    k1, k2, p1, p2, k3 = d
    x0, y0 = x.copy(), y.copy()
    for _ in range(20):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - dx) / rad
        y = (y0 - dy) / rad
    return x, y


def _points_from_depth(ctx: Context) -> np.ndarray:
    if not ctx.has_input("camera"):
        raise NodeError("depth input needs the camera input",
                        hint="pass -i camera=<camera.json> of the depth map")
    depth = np.load(ctx.input("depth")).astype(np.float64)
    cam = io.read_camera(ctx.input("camera"))
    K, dist = cam["K"].astype(np.float64), cam["dist"]
    h, w = depth.shape
    cw, ch = int(cam["width"]), int(cam["height"])
    if (cw, ch) != (w, h):
        if abs(cw / ch - w / h) > 0.01:
            raise NodeError(f"camera {cw}x{ch} does not match depth {w}x{h}")
        K = K.copy()
        K[0, 0] *= w / cw
        K[0, 2] = (K[0, 2] + 0.5) * w / cw - 0.5
        K[1, 1] *= h / ch
        K[1, 2] = (K[1, 2] + 0.5) * h / ch - 0.5
    max_depth = float(ctx.param("max_depth_m", 10.0))
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    ok = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    x, y = _undistort_norm(u[ok], v[ok], K, dist)
    z = depth[ok]
    return np.stack([x * z, y * z, z], 1)


def _points_from_ply(path: Path) -> np.ndarray:
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float64)
    return pts[np.isfinite(pts).all(1)]


# ----------------------------------------------------------------- planes
def _fit(pts: np.ndarray):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n, -float(n @ c)


def _extract(pts: np.ndarray, ctx: Context) -> List[Dict[str, Any]]:
    import open3d as o3d

    o3d.utility.random.seed(int(ctx.param("seed", 0)))
    thr = float(ctx.param("distance_threshold_m", 0.02))
    max_planes = int(ctx.param("max_planes", 12))
    min_ratio = float(ctx.param("min_inlier_ratio", 0.02))
    min_pts = max(int(ctx.param("min_inliers", 200)),
                  int(min_ratio * len(pts)))
    voxel = float(ctx.param("voxel_m", 0.02))
    split = bool(ctx.param("split_components", True))
    remaining = np.arange(len(pts))
    planes: List[Dict[str, Any]] = []
    attempts = 0
    while len(planes) < max_planes and len(remaining) >= min_pts \
            and attempts < 3 * max_planes:
        attempts += 1
        pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pts[remaining]))
        model, inl = pc.segment_plane(thr, 3, int(ctx.param(
            "ransac_iterations", 2000)))
        inl = np.asarray(inl)
        if len(inl) < min_pts:
            break
        # least-squares refit + re-collect inliers
        n, d = _fit(pts[remaining[inl]])
        dist = np.abs(pts[remaining] @ n + d)
        inl = np.nonzero(dist <= thr)[0]
        if split and len(inl) >= min_pts:
            sub = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pts[remaining[inl]]))
            lab = np.asarray(sub.cluster_dbscan(eps=3 * voxel,
                                                min_points=4))
            if lab.max() >= 0:
                big = np.bincount(lab[lab >= 0]).argmax()
                comp = inl[lab == big]
                if len(comp) >= min_pts:
                    inl = comp
                    n, d = _fit(pts[remaining[inl]])
        if len(inl) < min_pts:
            # component too small: drop these points from the pool, retry
            remaining = np.delete(remaining, inl)
            continue
        planes.append({"normal": n, "d": d, "idx": remaining[inl]})
        remaining = np.delete(remaining, inl)
    return planes


def _gravity_init(ctx: Context) -> np.ndarray:
    g = ctx.param("gravity", None) or [0.0, 1.0, 0.0]
    g = np.asarray(g, dtype=np.float64)
    if g.shape != (3,) or np.linalg.norm(g) < 1e-9:
        raise NodeError("param gravity must be a non-zero 3-vector")
    return g / np.linalg.norm(g)


def _plane_extent(p_world: np.ndarray, n_world: np.ndarray):
    """2D oriented extent (major, minor) and area estimate of a plane."""
    a = np.cross(n_world, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n_world, a)
    uv = np.stack([p_world @ a, p_world @ b], 1)
    uv -= uv.mean(0)
    _, _, vt = np.linalg.svd(uv, full_matrices=False)
    q = uv @ vt.T
    lo, hi = np.percentile(q, 1, axis=0), np.percentile(q, 99, axis=0)
    return [float(x) for x in (hi - lo)]


def layout(ctx: Context) -> None:
    has_pc, has_depth = ctx.has_input("pointcloud"), ctx.has_input("depth")
    if has_pc == has_depth:
        raise NodeError("pass exactly one of pointcloud or depth (+camera)")
    pts_all = (_points_from_ply(ctx.input("pointcloud")) if has_pc
               else _points_from_depth(ctx))
    if len(pts_all) < 1000:
        raise NodeError(f"only {len(pts_all)} valid points")
    import open3d as o3d

    voxel = float(ctx.param("voxel_m", 0.02))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_all))
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    pts = np.asarray(pc.points, dtype=np.float64)
    ctx.log(f"{len(pts_all)} points -> {len(pts)} after {voxel} m voxels")
    raw = _extract(pts, ctx)
    if not raw:
        raise NodeError("no plane found",
                        hint="increase distance_threshold_m or lower "
                        "min_inlier_ratio / min_inliers")

    # orient normals towards the input-frame origin (camera): n.o + d > 0
    for p in raw:
        if p["d"] < 0:
            p["normal"], p["d"] = -p["normal"], -p["d"]

    # gravity: hint, refined by the biggest near-horizontal plane
    tol = np.radians(float(ctx.param("angle_tol_deg", 15.0)))
    g = _gravity_init(ctx)
    gtol = np.radians(float(ctx.param("gravity_tol_deg", 45.0)))
    for p in raw:
        ctx.log(f"plane n={p['normal'].round(3).tolist()} d={p['d']:.3f} "
                f"inliers={len(p['idx'])} angle_to_hint="
                f"{np.degrees(np.arccos(min(1, abs(p['normal'] @ g)))):.1f}")
    horiz = [p for p in raw if abs(p["normal"] @ g) > np.cos(gtol)]
    if not horiz:
        raise NodeError(
            "no plane is within gravity_tol_deg of perpendicular to the "
            f"gravity hint {g.round(3).tolist()}",
            hint="pass gravity=[gx,gy,gz] (direction of gravity in the "
            "input frame) or raise gravity_tol_deg")
    up = None
    # floor candidates: horizontal, normal facing up (against gravity),
    # almost nothing below them
    below_max = float(ctx.param("max_below_ratio", 0.02))
    min_floor = float(ctx.param("min_floor_area_m2", 0.5))
    cands = []
    for p in horiz:
        n_up = p["normal"] if p["normal"] @ g < 0 else -p["normal"]
        h_pts = pts @ n_up  # height along this plane's up direction
        h0 = float(pts[p["idx"]].mean(0) @ n_up)
        below = float(np.mean(h_pts < h0 - 3 * float(ctx.param(
            "distance_threshold_m", 0.02))))
        ext = _plane_extent(pts[p["idx"]], n_up)
        p["_below"] = below
        ctx.log(f"  floor test: h={h0:.3f} below={below:.3f} "
                f"extent={np.round(ext, 2).tolist()}")
        if (p["normal"] @ g < 0 and below <= below_max
                and ext[0] * ext[1] >= min_floor):
            cands.append((h0, -len(p["idx"]), id(p), p, n_up))
    anchor = ctx.param("anchor", "floor")
    if cands and anchor in ("floor", "auto"):
        cands.sort(key=lambda c: (c[0], c[1]))
        floor, up = cands[0][3], cands[0][4]
        anchor_used = "floor"
    elif anchor in ("support", "auto"):
        ups = [p for p in horiz if p["normal"] @ g < 0]
        if not ups:
            raise NodeError("no up-facing horizontal plane found",
                            hint="check the gravity hint")
        floor = max(ups, key=lambda p: len(p["idx"]))
        up = floor["normal"].copy()
        anchor_used = "support"
    else:
        raise NodeError(
            "no floor plane found (horizontal, facing up, >= "
            f"min_floor_area_m2={min_floor} and <= {below_max:.0%} of the "
            "points below it)",
            hint="the floor must be visible (check the gravity hint, "
            "default +y = OpenCV camera down, or relax max_below_ratio / "
            "min_floor_area_m2); for table-top views without floor use "
            "anchor=support or anchor=auto")
    up = up / np.linalg.norm(up)

    # world frame: z = up, origin = projection of the input origin on the
    # floor, x = input z axis (camera forward) projected on the floor
    o_floor = -floor["d"] * floor["normal"]  # foot point of origin
    fwd = np.array([0, 0, 1.0]) - (np.array([0, 0, 1.0]) @ up) * up
    if np.linalg.norm(fwd) < 1e-3:  # camera looks straight down/up
        fwd = np.array([0, -1.0, 0]) - (np.array([0, -1.0, 0]) @ up) * up
    x = fwd / np.linalg.norm(fwd)
    y = np.cross(up, x)
    R_wi = np.stack([x, y, up], 0)  # rows = world axes in input frame
    T_wi = np.eye(4)
    T_wi[:3, :3] = R_wi
    T_wi[:3, 3] = -R_wi @ o_floor

    pts_w = (R_wi @ pts.T).T + T_wi[:3, 3]
    ceiling_min = float(ctx.param("ceiling_min_height_m", 1.8))
    table_range = ctx.param("table_height_m", [0.4, 1.2])
    wall_min = float(ctx.param("min_wall_area_m2", 0.5))
    planes_out = []
    label_of = np.full(len(pts), -1, int)
    for k, p in enumerate(raw):
        n_w = R_wi @ p["normal"]
        P = pts_w[p["idx"]]
        d_w = -float(n_w @ P.mean(0))
        c = P.mean(0)
        is_h = abs(n_w[2]) > np.cos(tol)
        is_v = abs(n_w[2]) < np.sin(tol)
        ext = _plane_extent(P, n_w)
        area = ext[0] * ext[1]
        if p is floor:
            label = "floor" if anchor_used == "floor" else "table"
        elif is_h and n_w[2] < 0 and c[2] >= ceiling_min:
            label = "ceiling"
        elif is_h and n_w[2] > 0 and anchor_used == "floor" and \
                table_range[0] <= c[2] <= table_range[1]:
            label = "table"
        elif is_v and area >= wall_min:
            label = "wall"
        else:
            label = "other"
        label_of[p["idx"]] = k
        planes_out.append({
            "id": k, "label": label,
            "normal": p["normal"].tolist(), "d": float(p["d"]),
            "normal_world": n_w.tolist(), "d_world": d_w,
            "inliers": int(len(p["idx"])),
            "centroid_world": c.tolist(),
            "bounds_world": {"min": P.min(0).tolist(),
                             "max": P.max(0).tolist()},
            "extent_m": ext, "area_m2": float(area),
            "height_m": float(c[2]) if is_h else None,
            "orientation": "horizontal" if is_h else (
                "vertical" if is_v else "slanted"),
        })
    planes_out.sort(key=lambda q: -q["inliers"])
    pmap = {q["id"]: i for i, q in enumerate(planes_out)}
    for i, q in enumerate(planes_out):
        q["id"] = i
    label_of = np.array([pmap.get(i, -1) for i in label_of])

    ppath = ctx.output_path("planes", "planes.json")
    io.write_json(ppath, {
        "convention": "n.x + d = 0 in metres; normal/d in the INPUT frame "
                      "(oriented towards the input origin, i.e. the "
                      "camera), normal_world/d_world in the world frame",
        "points_used": int(len(pts)), "voxel_m": voxel,
        "planes": planes_out,
        "counts": {lab: sum(q["label"] == lab for q in planes_out)
                   for lab in ("floor", "wall", "ceiling", "table",
                               "other")}})
    ctx.set_output("planes", ppath)

    g_ref = -up
    cam_h = float(T_wi[2, 3])
    world = {
        "definition": "z up (against gravity, = anchor plane normal), "
                      "origin = foot of the input-frame origin (camera "
                      "centre for depth input) on the anchor plane (floor, "
                      "or the support plane if anchor=support), x = input z axis "
                      "(camera forward) projected onto the floor, y = z "
                      "cross x (left)",
        "T_world_input": T_wi.tolist(),
        "T_world_cam": T_wi.tolist() if has_depth else None,
        "gravity_input": g_ref.tolist(),
        "gravity_hint": g.tolist(),
        "gravity_hint_angle_deg": float(np.degrees(np.arccos(np.clip(
            g @ g_ref, -1, 1)))),
        "anchor": anchor_used,
        "floor_plane_id": pmap[raw.index(floor)] if anchor_used == "floor"
        else None,
        "anchor_plane_id": pmap[raw.index(floor)],
        "input_origin_height_m": cam_h,
    }
    wpath = ctx.output_path("world", "world.json")
    io.write_json(wpath, world)
    ctx.set_output("world", wpath)

    if ctx.has_input("trajectory"):
        tr = io.read_json(ctx.input("trajectory"))
        Ts = np.asarray(tr["T_world_cam"], dtype=np.float64)
        tr["T_world_cam"] = [(T_wi @ T).tolist() for T in Ts]
        tpath = ctx.output_path("trajectory", "trajectory.json")
        io.write_json(tpath, tr)
        ctx.set_output("trajectory", tpath)

    if bool(ctx.param("labelled_pointcloud", True)):
        names = [q["label"] for q in planes_out]
        col = np.array([COLORS[names[i]] if i >= 0 else COLORS["none"]
                        for i in label_of], np.uint8)
        _write_ply(ctx.output_path("pointcloud", "labelled.ply"), pts_w, col,
                   label_of)
        ctx.set_output("pointcloud",
                       ctx.output_dir / "labelled.ply")
    ctx.metadata.update({"planes": len(planes_out),
                         "counts": {lab: sum(q["label"] == lab
                                             for q in planes_out)
                                    for lab in COLORS if lab != "none"},
                         "input_origin_height_m": cam_h})


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray,
               plane: np.ndarray) -> None:
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("plane", "<i4")])
    arr = np.empty(len(xyz), dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["plane"] = plane
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(arr)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property int plane\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


if __name__ == "__main__":
    raise SystemExit(main({"extract_planes": layout}))
