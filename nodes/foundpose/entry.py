"""FoundPose node: render_templates (onboarding) / estimate (coarse pose).

Drives the upstream FoundPose utils (template rendering, DINOv2 features,
PCA + k-means visual words, tf-idf template retrieval, cyclic-buddy
2D-3D matching, PnP-RANSAC) without the BOP dataset plumbing of the
upstream scripts. Upstream works in millimetres internally; this node
converts to metres at its boundary.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

# DINOv2 variants (upstream extractor names); vits14-reg/layer 9 is the
# configuration of the released LM-O/TUD-L configs.
EXTRACTORS = {
    "vits14-reg": ("dinov2_vits14_reg", "dinov2_vits14_reg4_pretrain.pth",
                   "dinov2_version=vits14-reg_stride=14_facet=token_layer=9"
                   "_logbin=0_norm=1"),
}
# Default template camera when none is given (LM-O camera, 640x480); the
# crop camera makes templates almost independent of it.
DEFAULT_K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899],
                      [0.0, 0.0, 1.0]])
DEFAULT_WH = (640, 480)
OBJ_ID = 1


# ----------------------------------------------------------------- setup
def _upstream(ctx: Context):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for p in (ctx.repo / "external" / "dinov2", ctx.repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if not (ctx.repo / "external" / "dinov2" / "dinov2").is_dir():
        raise NodeError("dinov2 submodule missing in upstream checkout",
                        kind="setup", hint="run `verdi setup foundpose`")


_EGL_CHECKED = False


def _select_egl_device(ctx: Context) -> None:
    """Pick the first EGL device that actually renders.

    On the H20 the NVIDIA EGL device creates a context but rasterises
    nothing (all-black colour, depth = znear), while Mesa llvmpipe works;
    on the RTX 5090 host NVIDIA EGL works. Probe each device (the one of the
    assigned GPU first) with a tiny box render; EGL_DEVICE_ID overrides.
    """
    global _EGL_CHECKED
    if _EGL_CHECKED or os.environ.get("EGL_DEVICE_ID"):
        _EGL_CHECKED = True
        return
    import subprocess

    probe = (
        "import numpy as np, pyrender, trimesh\n"
        "from pyrender.platforms import egl\n"
        "import sys\n"
        "if sys.argv[1] == 'n': print(len(egl.query_devices())); raise SystemExit\n"
        "T = np.eye(4); T[2, 3] = 0.8\n"
        "r = pyrender.OffscreenRenderer(64, 64)\n"
        "s = pyrender.Scene(bg_color=np.zeros(4))\n"
        "s.add(pyrender.Mesh.from_trimesh(trimesh.creation.box((0.1,) * 3)))\n"
        "s.add(pyrender.IntrinsicsCamera(fx=60, fy=60, cx=32, cy=32, "
        "znear=0.1, zfar=100.0), pose=T)\n"
        "_, d = r.render(s)\n"
        "print('OK' if abs(float(d[32, 32]) - 0.75) < 0.01 and d[0, 0] == 0 "
        "else 'BAD')\n")
    env = dict(os.environ, PYOPENGL_PLATFORM="egl")
    out = subprocess.run([sys.executable, "-c", probe, "n"], env=env,
                         capture_output=True, text=True)
    n = int(out.stdout.strip() or 0)
    order = list(range(n))
    # prefer the EGL device of the GPU this run was given (EGL devices are
    # enumerated in the same PCI order as CUDA on NVIDIA multi-GPU hosts)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if vis.isdigit() and int(vis) < n:
        order.remove(int(vis))
        order.insert(0, int(vis))
    for idx in order:
        env["EGL_DEVICE_ID"] = str(idx)
        out = subprocess.run([sys.executable, "-c", probe, "r"], env=env,
                             capture_output=True, text=True, timeout=300)
        ok = out.stdout.strip().endswith("OK")
        ctx.log(f"EGL device {idx}: {'ok' if ok else 'does not render'}")
        if ok:
            os.environ["EGL_DEVICE_ID"] = str(idx)
            _EGL_CHECKED = True
            return
    raise NodeError("no EGL device renders correctly (pyrender)",
                    kind="setup", hint="install a working EGL driver "
                    "(libegl1 / Mesa) or set EGL_DEVICE_ID")


def _extractor(ctx: Context, name: str, device: str):
    """Build the upstream DinoFeatureExtractor from the pinned weights."""
    import torch

    _upstream(ctx)
    base, fname, ext_name = EXTRACTORS[name]
    ckpt = ctx.weight(name.replace("-", "_")) / fname
    if not ckpt.exists():
        raise NodeError(f"missing {ckpt}", kind="setup",
                        hint="run `verdi setup foundpose`")
    # dinov2 hub code downloads into $TORCH_HOME/hub/checkpoints/<fname>;
    # point it at the pinned file instead of the network.
    th = Path(tempfile.mkdtemp(prefix="foundpose_torch_"))
    (th / "hub" / "checkpoints").mkdir(parents=True)
    (th / "hub" / "checkpoints" / fname).symlink_to(ckpt)
    torch.hub.set_dir(str(th / "hub"))
    from utils import feature_util

    ext = feature_util.make_feature_extractor(ext_name)
    return ext.to(device), ext_name


def _load_mesh_m(path: Path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise NodeError(f"{path}: could not load a triangle mesh",
                        kind="request")
    vis = mesh.visual
    if isinstance(vis, TextureVisuals) and isinstance(vis.material,
                                                      PBRMaterial):
        if getattr(vis.material, "baseColorTexture", None) is None:
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
    extent = float(np.max(mesh.extents))
    if not 0.005 < extent < 5.0:
        raise NodeError(f"mesh extent {extent:.4g} m looks not metric",
                        kind="request", hint="mesh must be in metres")
    return mesh


def _diameter_m(mesh) -> float:
    pts = np.asarray(mesh.vertices)
    if len(pts) > 2000:
        pts = pts[np.random.RandomState(0).choice(len(pts), 2000, False)]
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    return float(d.max())


# ------------------------------------------------------------ templates
def _template_views(opts):
    from utils import geometry, misc

    radius_mm = 1000.0 * opts["distance_m"]
    views_sphere = misc.sample_views(min_n_views=opts["num_viewpoints"],
                                     radius=radius_mm, mode="fibonacci")[0]
    n_inplane = int(opts["num_inplane_rotations"])
    views = []
    for v in views_sphere:
        for k in range(n_inplane):
            R_in = geometry.rotation_matrix_numpy(
                2 * np.pi / n_inplane * k, np.array([0, 0, 1]))[:3, :3]
            views.append({"R": R_in.dot(v["R"]), "t": R_in.dot(v["t"])})
    return views


def _render_templates(ctx: Context, mesh_path: Path, out_dir: Path, K, wh,
                      opts):
    """Port of upstream scripts/gen_templates.py, parallel over processes.

    Each worker owns one pyrender/EGL context (llvmpipe on the H20).
    """
    import multiprocessing as mp

    _select_egl_device(ctx)
    views = _template_views(opts)
    for sub in ("rgb", "depth", "mask"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 8)
    workers = int(ctx.param("render_workers", 0)) or max(
        1, min(12, ncpu // 8, len(views)))
    items = [(tid, v["R"], v["t"]) for tid, v in enumerate(views)]
    chunks = [items[i::workers] for i in range(workers)]
    args = [(str(ctx.repo), str(mesh_path), str(out_dir), np.asarray(K),
             tuple(wh), dict(opts), c) for c in chunks if c]
    os.environ.setdefault("LP_NUM_THREADS",
                          str(max(1, min(8, ncpu // workers))))
    ctx.log(f"rendering {len(items)} templates with {len(args)} workers")
    with mp.get_context("spawn").Pool(len(args)) as pool:
        results = pool.map(_render_worker, args)
    meta = []
    for res in results:
        if isinstance(res, str):
            raise NodeError(res, hint="check mesh units / distance_m")
        meta.extend(res)
    meta.sort(key=lambda m: m["id"])
    io.write_json(out_dir / "metadata.json", meta)
    ctx.log(f"rendered {len(meta)} templates")
    return meta


def _render_worker(args):
    """Render a subset of templates; returns metadata list or error str."""
    repo, mesh_path, out_dir, K, wh, opts, items = args
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for p in (os.path.join(repo, "external", "dinov2"), repo):
        if p not in sys.path:
            sys.path.insert(0, p)
    out_dir = Path(out_dir)
    try:
        return _render_items(_load_mesh_m(Path(mesh_path)), out_dir, K, wh,
                             opts, items)
    except NodeError as exc:
        return str(exc)


def _render_items(mesh, out_dir: Path, K, wh, opts, items):
    import cv2
    from utils import misc, renderer_builder, structs
    from utils.renderer_base import RenderType
    from utils.structs import AlignedBox2f, PinholePlaneCameraModel

    np.random.seed(0)
    patch = 14
    image_side = patch * int(max(wh) / patch)
    cam = PinholePlaneCameraModel(
        width=image_side, height=image_side, f=(K[0, 0], K[1, 1]),
        c=(K[0, 2] - 0.5 * (wh[0] - image_side),
           K[1, 2] - 0.5 * (wh[1] - image_side)))
    ss = float(opts["ssaa_factor"])
    rcam = PinholePlaneCameraModel(
        width=int(cam.width * ss), height=int(cam.height * ss),
        f=(cam.f[0] * ss, cam.f[1] * ss), c=(cam.c[0] * ss, cam.c[1] * ss))
    renderer = renderer_builder.build(
        renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER)
    # upstream loads BOP mm models and scales to m; give it the metric mesh
    renderer.object_meshes[OBJ_ID] = mesh
    renderer.add_object_model(obj_id=OBJ_ID, model_path="<in-memory>")
    crop_size = (int(opts["crop_size"]), int(opts["crop_size"]))
    meta = []
    types_ = [RenderType.COLOR, RenderType.DEPTH, RenderType.MASK]
    for tid, R_v, t_v in items:
        m2c = structs.RigidTransform(R=R_v, t=t_v)
        R_c2m = m2c.R.T
        c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(m2c.t))
        rcam_c2w = PinholePlaneCameraModel(
            width=rcam.width, height=rcam.height, f=rcam.f, c=rcam.c,
            T_world_from_eye=misc.get_rigid_matrix(c2m))
        out = renderer.render_object_model(
            obj_id=OBJ_ID, camera_model_c2w=rcam_c2w, render_types=types_,
            return_tensors=False, debug=False)
        out[RenderType.MASK] = (255 * out[RenderType.MASK]).astype(np.uint8)
        ys, xs = out[RenderType.MASK].nonzero()
        if len(xs) == 0:
            raise NodeError(f"template {tid}: object not visible",
                            hint="check mesh units / distance_m")
        box = np.array(misc.calc_2d_box(xs, ys))
        obox = AlignedBox2f(left=box[0], top=box[1], right=box[2],
                            bottom=box[3])
        if (obox.left == 0 or obox.top == 0
                or obox.right == rcam.width - 1
                or obox.bottom == rcam.height - 1):
            raise NodeError("the object does not fit the template viewport",
                            hint="increase distance_m")
        crop_box = misc.calc_crop_box(box=obox, make_square=True)
        ccam = misc.construct_crop_camera(
            box=crop_box, camera_model_c2w=rcam_c2w,
            viewport_size=(int(crop_size[0] * ss), int(crop_size[1] * ss)),
            viewport_rel_pad=float(opts["crop_rel_pad"]))
        for key in list(out):
            if key == RenderType.DEPTH:
                out[key] = misc.warp_depth_image(
                    src_camera=rcam_c2w, dst_camera=ccam,
                    src_depth_image=out[key])
            elif key == RenderType.COLOR:
                interp = (cv2.INTER_AREA if crop_box.width >= ccam.width
                          else cv2.INTER_LINEAR)
                out[key] = misc.warp_image(src_camera=rcam_c2w,
                                           dst_camera=ccam,
                                           src_image=out[key],
                                           interpolation=interp)
            else:
                out[key] = misc.warp_image(src_camera=rcam_c2w,
                                           dst_camera=ccam,
                                           src_image=out[key],
                                           interpolation=cv2.INTER_NEAREST)
        cam_c2w = ccam.copy()
        sf = crop_size[0] / float(ccam.width)
        cam_c2w.width, cam_c2w.height = crop_size
        cam_c2w.c = (cam_c2w.c[0] * sf, cam_c2w.c[1] * sf)
        cam_c2w.f = (cam_c2w.f[0] * sf, cam_c2w.f[1] * sf)
        if ss != 1.0:
            for key in list(out):
                out[key] = misc.resize_image(
                    image=out[key], size=crop_size,
                    interpolation=(cv2.INTER_AREA if key == RenderType.COLOR
                                   else cv2.INTER_NEAREST))
        rgb = np.asarray(255.0 * out[RenderType.COLOR], np.uint8)
        depth_mm = out[RenderType.DEPTH].astype(np.float32)
        mask = out[RenderType.MASK] > 0
        name = io.frame_name(tid)
        io.write_image(out_dir / "rgb" / name, rgb)
        io.write_mask(out_dir / "mask" / name, mask.astype(np.uint8))
        np.save(out_dir / "depth" / (name[:-4] + ".npy"),
                (depth_mm / 1000.0).astype(np.float32))
        T_m2c = np.linalg.inv(cam_c2w.T_world_from_eye)  # world = model
        T_m2c_m = T_m2c.copy()
        T_m2c_m[:3, 3] /= 1000.0
        Kc = np.array([[cam_c2w.f[0], 0, cam_c2w.c[0]],
                       [0, cam_c2w.f[1], cam_c2w.c[1]], [0, 0, 1]])
        meta.append({"id": tid, "rgb": f"rgb/{name}", "mask": f"mask/{name}",
                     "depth": f"depth/{name[:-4]}.npy",
                     "T_cam_obj": T_m2c_m.tolist(), "K": Kc.tolist(),
                     "width": crop_size[0], "height": crop_size[1]})
    return meta


def _load_templates(tdir: Path):
    meta_path = tdir / "metadata.json"
    if not meta_path.exists():
        raise NodeError(f"{tdir} has no metadata.json", kind="request",
                        hint="pass a dir produced by foundpose "
                        "render_templates")
    return io.read_json(meta_path)


# ---------------------------------------------------------- repre
def _build_repre(ctx: Context, tdir: Path, meta, extractor, ext_name,
                 opts, device):
    """Port of upstream scripts/gen_repre.py (raw repre, PCA, words, tfidf)."""
    import torch
    from utils import (cluster_util, feature_util, projector_util,
                       repre_util, template_util)
    from utils.structs import PinholePlaneCameraModel

    feats, f2v, verts, f2t, tpls, cams = [], [], [], [], [], []
    with torch.no_grad():
        for i, t in enumerate(meta):
            K = np.asarray(t["K"])
            T_m2c = np.asarray(t["T_cam_obj"], dtype=np.float64).copy()
            T_m2c[:3, 3] *= 1000.0  # upstream works in mm
            T_c2m = np.linalg.inv(T_m2c)
            cam = PinholePlaneCameraModel(
                width=t["width"], height=t["height"], f=(K[0, 0], K[1, 1]),
                c=(K[0, 2], K[1, 2]), T_world_from_eye=T_c2m)
            img = torch.as_tensor(np.array(io.read_image(tdir / t["rgb"]))).to(
                device, torch.float32).permute(2, 0, 1) / 255.0
            depth = torch.as_tensor(np.load(tdir / t["depth"]) * 1000.0).to(
                device, torch.float32)
            from PIL import Image

            mask = torch.as_tensor(
                (np.asarray(Image.open(tdir / t["mask"])) > 0).astype(
                    np.float32)).to(device)
            T_c2m_t = torch.as_tensor(T_c2m, dtype=torch.float32,
                                      device=device)
            fv, fvid, vm = feature_util.get_visual_features_registered_in_3d(
                image_chw=img, depth_image_hw=depth, object_mask=mask,
                camera=cam, T_model_from_camera=T_c2m_t, extractor=extractor,
                grid_cell_size=float(opts["grid_cell_size"]), debug=False)
            feats.append(fv)
            f2v.append(fvid)
            verts.append(vm)
            f2t.append(i * torch.ones(fv.shape[0], dtype=torch.int32,
                                      device=device))
            tpls.append((img * 255).to(torch.uint8))
            c = cam.copy()
            c.extrinsics = torch.linalg.inv(T_c2m_t)
            cams.append(c)
    repre = repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(verts), feat_vectors=torch.cat(feats),
        feat_opts=repre_util.FeatureOpts(extractor_name=ext_name),
        feat_to_vertex_ids=torch.cat(f2v),
        feat_to_template_ids=torch.cat(f2t), templates=torch.stack(tpls),
        template_cameras_cam_from_model=cams)
    fvec = repre.feat_vectors
    if fvec.shape[0] < int(opts["cluster_num"]):
        raise NodeError(f"only {fvec.shape[0]} template features for "
                        f"{opts['cluster_num']} visual words",
                        hint="render more templates or lower cluster_num")
    pca = projector_util.PCAProjector(n_components=int(opts["pca_components"]),
                                      whiten=False)
    pca.fit(fvec, max_samples=100000)
    repre.feat_raw_projectors.append(pca)
    fvec = pca.transform(fvec)
    # CPU k-means: faiss-gpu wheels abort on the H20 (cudaSuccess assert);
    # upstream picks GPU only because the samples are on CUDA.
    centroids, cids, _ = cluster_util.kmeans(
        samples=fvec.cpu(), num_centroids=int(opts["cluster_num"]),
        verbose=False)
    repre.feat_cluster_centroids = centroids.to(device)
    repre.feat_to_cluster_ids = cids.to(device)
    desc = repre_util.TemplateDescOpts(desc_type="tfidf")
    repre.template_desc_opts = desc
    repre.template_descs, repre.feat_cluster_idfs = \
        template_util.calc_tfidf_descriptors(
            feat_vectors=fvec, feat_words=repre.feat_cluster_centroids,
            feat_to_word_ids=repre.feat_to_cluster_ids,
            feat_to_template_ids=repre.feat_to_template_ids,
            num_templates=len(repre.templates),
            tfidf_knn_k=desc.tfidf_knn_k,
            tfidf_soft_assign=desc.tfidf_soft_assign,
            tfidf_soft_sigma_squared=desc.tfidf_soft_sigma_squared)
    repre.feat_vis_projectors = [pca]
    repre.feat_vectors = fvec
    return repre


REPRE_KEYS = ("extractor", "grid_cell_size", "pca_components", "cluster_num")


def _repre_opts(ctx: Context):
    return {"extractor": ctx.param("extractor", "vits14-reg"),
            "grid_cell_size": 14.0,
            "pca_components": int(ctx.param("pca_components", 256)),
            "cluster_num": int(ctx.param("cluster_num", 2048))}


def _render_opts(ctx: Context):
    return {"num_viewpoints": int(ctx.param("num_viewpoints", 57)),
            "num_inplane_rotations": int(ctx.param("num_inplane_rotations",
                                                   14)),
            "distance_m": float(ctx.param("distance_m", 0.0)),
            "crop_size": 420, "crop_rel_pad": 0.2,
            "ssaa_factor": float(ctx.param("ssaa_factor", 4.0))}


def _onboard(ctx: Context, mesh_path: Path, tdir: Path, K, wh, device,
             build_repre: bool):
    import torch
    from utils import repre_util

    mesh = _load_mesh_m(mesh_path)
    ropts = _render_opts(ctx)
    diam = _diameter_m(mesh)
    if ropts["distance_m"] <= 0:
        ropts["distance_m"] = round(4.0 * diam, 4)
    meta = _render_templates(ctx, mesh_path, tdir, K, wh, ropts)
    cfg = {"render": ropts, "mesh_diameter_m": diam,
           "camera_K": np.asarray(K).tolist(), "camera_wh": list(wh)}
    repre = None
    if build_repre:
        popts = _repre_opts(ctx)
        extractor, ext_name = _extractor(ctx, popts["extractor"], device)
        repre = _build_repre(ctx, tdir, meta, extractor, ext_name, popts,
                             device)
        repre_util.save_object_repre(repre, str(tdir))
        cfg["repre"] = popts
        del extractor
        torch.cuda.empty_cache()
    io.write_json(tdir / "config.json", cfg)
    return meta, repre


# ---------------------------------------------------------------- tasks
def render_templates(ctx: Context) -> None:
    device = ctx.device
    if ctx.has_input("camera"):
        cam = io.read_camera(ctx.input("camera"))
        K, wh = cam["K"], (int(cam["width"]), int(cam["height"]))
    else:
        K, wh = DEFAULT_K, DEFAULT_WH
    _upstream(ctx)
    tdir = ctx.output_path("templates")
    meta, _ = _onboard(ctx, ctx.input("mesh"), tdir, K, wh, device,
                       bool(ctx.param("build_repre", True)))
    ctx.set_output("templates", tdir)
    ctx.set_output("masks", tdir / "mask")
    ctx.metadata["num_templates"] = len(meta)


def _query_mask(ctx: Context, hw):
    from PIL import Image

    mask, box = None, None
    if ctx.has_input("mask"):
        ids = np.asarray(Image.open(ctx.input("mask")))
        if ids.ndim == 3:
            ids = ids[..., 0]
        if ids.shape != tuple(hw):
            raise NodeError(f"mask is {ids.shape}, image is {tuple(hw)}",
                            kind="request")
        mid = int(ctx.param("mask_id", 0))
        mask = (ids == mid) if mid else (ids != 0)
        if not mask.any():
            raise NodeError("object mask is empty", kind="request")
    if ctx.has_input("bbox"):
        boxes = io.read_json(ctx.input("bbox")).get("boxes", [])
        if not boxes:
            raise NodeError("bbox_set has no boxes", kind="request")
        best = max(boxes, key=lambda b: b.get("score") or 0.0)
        box = [float(v) for v in best["xyxy"]]
    if mask is None and box is None:
        raise NodeError("need a mask or a bbox of the object", kind="request",
                        hint="pass -i mask=... (preferred) or -i bbox=...")
    if mask is None:
        # box only: all pixels in the box are treated as object pixels
        mask = np.zeros(hw, bool)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
    if box is None:
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max()),
               float(ys.max())]
    return mask.astype(np.uint8), box


def estimate(ctx: Context) -> None:
    import cv2
    import torch

    device = ctx.device
    _upstream(ctx)
    from utils import (corresp_util, feature_util, knn_util, misc,
                       pnp_util, projector_util, repre_util)
    from utils.misc import warp_image
    from utils.structs import AlignedBox2f, PinholePlaneCameraModel
    rgb = io.read_image(ctx.input("image"))
    hw = rgb.shape[:2]
    cam = io.read_camera(ctx.input("camera"))
    if (int(cam["height"]), int(cam["width"])) != tuple(hw):
        raise NodeError("camera size differs from image size",
                        kind="request")
    K = cam["K"]
    mask, box = _query_mask(ctx, hw)
    popts = _repre_opts(ctx)

    # ---- object representation (templates dir or on-the-fly onboarding)
    repre = None
    if ctx.has_input("templates"):
        tdir = ctx.input("templates")
        meta = _load_templates(tdir)
        cfg = io.read_json(tdir / "config.json") \
            if (tdir / "config.json").exists() else {}
        if (tdir / "repre.pth").exists() and all(
                cfg.get("repre", {}).get(k) == popts[k] for k in REPRE_KEYS):
            repre = repre_util.load_object_repre(str(tdir),
                                                 tensor_device=device)
        else:
            extractor, ext_name = _extractor(ctx, popts["extractor"], device)
            repre = _build_repre(ctx, tdir, meta, extractor, ext_name,
                                 popts, device)
    elif ctx.has_input("mesh"):
        tdir = ctx.output_path("_templates")
        _, repre = _onboard(ctx, ctx.input("mesh"), tdir, K,
                            (int(cam["width"]), int(cam["height"])), device,
                            True)
    else:
        raise NodeError("need templates (from render_templates) or a mesh",
                        kind="request")
    for k in ("feat_vectors", "feat_to_template_ids", "vertices",
              "feat_cluster_centroids", "template_descs",
              "feat_cluster_idfs", "feat_to_vertex_ids",
              "feat_to_cluster_ids"):
        v = getattr(repre, k, None)
        if torch.is_tensor(v):
            setattr(repre, k, v.to(device))
    extractor, _ = _extractor(ctx, popts["extractor"], device)

    # ---- kNN indices (as upstream infer.py)
    words_knn = knn_util.KNN(k=repre.template_desc_opts.tfidf_knn_k,
                             metric=repre.template_desc_opts.tfidf_knn_metric)
    words_knn.fit(repre.feat_cluster_centroids)
    tpl_knn = []
    for tid in range(len(repre.template_cameras_cam_from_model)):
        ids = torch.nonzero(repre.feat_to_template_ids == tid).flatten()
        index = knn_util.KNN(k=1, metric="l2")
        index.fit(repre.feat_vectors[ids].cpu())
        tpl_knn.append(index)

    # ---- crop the query around the object (virtual camera)
    crop_size = (420, 420)
    orig_cam = PinholePlaneCameraModel(
        width=int(cam["width"]), height=int(cam["height"]),
        f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]),
        T_world_from_eye=np.eye(4))
    obox = AlignedBox2f(left=box[0], top=box[1], right=box[2],
                        bottom=box[3])
    crop_box = misc.calc_crop_box(box=obox, make_square=True)
    ccam = misc.construct_crop_camera(box=crop_box, camera_model_c2w=orig_cam,
                                      viewport_size=crop_size,
                                      viewport_rel_pad=0.2)
    img = rgb.astype(np.float32) / 255.0
    interp = (cv2.INTER_AREA if crop_box.width >= ccam.width
              else cv2.INTER_LINEAR)
    img_c = warp_image(src_camera=orig_cam, dst_camera=ccam, src_image=img,
                       interpolation=interp)
    mask_c = warp_image(src_camera=orig_cam, dst_camera=ccam,
                        src_image=mask, interpolation=cv2.INTER_NEAREST)

    with torch.no_grad():
        x = torch.as_tensor(img_c).to(device, torch.float32).permute(2, 0, 1)
        fmap = extractor(x.unsqueeze(0))["feature_maps"][0]
        grid = feature_util.generate_grid_points(
            grid_size=crop_size, cell_size=popts["grid_cell_size"]).to(device)
        qpts = feature_util.filter_points_by_mask(
            grid, torch.as_tensor(mask_c).to(device))
        if len(qpts) < 6:
            raise NodeError(f"only {len(qpts)} query points inside the mask",
                            kind="request", hint="object too small / mask")
        qfeat = feature_util.sample_feature_map_at_points(
            feature_map_chw=fmap, points=qpts,
            image_size=(img_c.shape[1], img_c.shape[0])).contiguous()
        qfeat = projector_util.project_features(
            feat_vectors=qfeat, projectors=repre.feat_raw_projectors
        ).contiguous()
        corresp = corresp_util.establish_correspondences(
            query_points=qpts, query_features=qfeat, object_repre=repre,
            template_matching_type="tfidf", template_knn_indices=tpl_knn,
            feat_matching_type="cyclic_buddies",
            top_n_templates=int(ctx.param("top_n_templates", 5)),
            top_k_buddies=int(ctx.param("top_k_buddies", 300)),
            visual_words_knn_index=words_knn, debug=False)

    best = None
    for cid, c in enumerate(corresp):
        if len(c["coord_2d"]) < 6:
            continue
        ok, R, t, inl, q = pnp_util.estimate_pose(
            corresp=c, camera_c2w=ccam, pnp_type="opencv",
            pnp_ransac_iter=int(ctx.param("pnp_ransac_iter", 400)),
            pnp_inlier_thresh=float(ctx.param("pnp_inlier_thresh", 10.0)),
            pnp_required_ransac_conf=0.99, pnp_refine_lm=True)
        ctx.log(f"template {int(c['template_id'])}: "
                f"{len(c['coord_2d'])} corresp, quality {q}")
        if ok and (best is None or q > best["quality"]):
            best = {"R": R, "t": t, "quality": q,
                    "template_id": int(c["template_id"]),
                    "n_corresp": int(len(c["coord_2d"]))}
    if best is None:
        raise NodeError("PnP failed for all retrieved templates",
                        hint="check mask/camera; the object may be too "
                        "small or the mesh may not match the object")
    T_m2vc = np.eye(4)
    T_m2vc[:3, :3] = best["R"]
    T_m2vc[:3, 3] = np.asarray(best["t"]).reshape(3)
    T = ccam.T_world_from_eye.dot(T_m2vc)  # world = original camera
    T[:3, 3] /= 1000.0
    if not np.all(np.isfinite(T)):
        raise NodeError("non-finite pose from PnP")
    out = ctx.output_path("poses", "poses.json")
    io.write_pose_set(out, [{
        "T_cam_obj": T, "label": ctx.param("label", "object"),
        "score": float(best["quality"]),  # number of PnP inliers
        "template_id": best["template_id"],
        "num_corresp": best["n_corresp"]}])
    ctx.set_output("poses", out)


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    raise SystemExit(main({"render_templates": render_templates,
                           "estimate": estimate}))
