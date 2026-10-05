import json

import numpy as np
import pytest

from verdi import types
from verdi.types import io
from verdi.types.registry import TypeError_


def test_image_seq_and_mask(tmp_path):
    frames = [np.full((8, 10, 3), i * 20, np.uint8) for i in range(3)]
    seq = io.write_image_seq(tmp_path / "f", frames, fps=10)
    s = types.validate("image_seq", seq)
    assert s["count"] == 3 and s["width"] == 10 and s["fps"] == 10

    ids = np.zeros((8, 10), np.uint8)
    ids[2:4, 2:4] = 1
    m = io.write_mask(tmp_path / "m.png", ids, {1: {"label": "cup"}})
    s = types.validate("mask", m)
    assert s["instances"] == 1
    assert types.compare("mask_iou", m, m) == 1.0


def test_depth_and_pose(tmp_path):
    d = io.write_depth(tmp_path / "d.npy", np.ones((4, 5)) * 2.0)
    assert types.validate("depth", d)["median"] == 2.0
    p = io.write_pose_set(tmp_path / "p.json",
                          [{"T_cam_obj": np.eye(4), "label": "a"}])
    assert types.validate("pose_set", p)["count"] == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"poses": [{"T_cam_obj": [[1, 0], [0, 1]]}]}))
    with pytest.raises(TypeError_):
        types.validate("pose_set", bad)


def test_bbox_rejects_xywh_like(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"boxes": [{"xyxy": [10, 10, 5, 5]}]}))
    with pytest.raises(TypeError_):
        types.validate("bbox_set", p)


def test_new_metrics_and_types(tmp_path):
    from verdi.core.testing import _resolve

    ids = np.zeros((6, 6), np.uint8)
    ids[:3] = 1
    ids[3:] = 2
    a = io.write_mask(tmp_path / "a.png", ids)
    b = io.write_mask(tmp_path / "b.png", np.where(ids == 1, 1, 0))
    assert types.compare("mask_iou", a, a) == 1.0
    assert types.compare("mask_class_iou", a, b) == 0.5

    bx = tmp_path / "bx.json"
    io.write_bbox_set(bx, [{"xyxy": [0, 0, 10, 10], "label": "cup"}])
    by = tmp_path / "by.json"
    io.write_bbox_set(by, [{"xyxy": [0, 0, 10, 5], "label": "cup"}])
    assert abs(types.compare("bbox_iou", bx, by) - 0.5) < 1e-6

    T = np.eye(4)
    R90 = np.eye(4)
    R90[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pa = io.write_pose_set(tmp_path / "pa.json", [{"T_cam_obj": T}])
    pb = io.write_pose_set(tmp_path / "pb.json", [{"T_cam_obj": R90}])
    assert abs(types.compare("pose_rotation_err_deg", pa, pb) - 90) < 1e-6

    d = tmp_path / "d"
    d.mkdir()
    for i in range(2):
        io.write_depth(d / io.frame_name(i, ".npy"), np.ones((3, 3)))
    assert types.compare("depth_abs_rel", d, d) == 0.0

    traj = tmp_path / "t.json"
    Ts = []
    for i in range(4):
        M = np.eye(4)
        M[:3, 3] = [i, 0, i * 0.5]
        Ts.append(M.tolist())
    traj.write_text(json.dumps({"T_world_cam": Ts}))
    assert types.compare("trajectory_ate", traj, traj) < 1e-9

    n = np.zeros((4, 4, 3), np.float32)
    n[..., 2] = -1
    np.save(tmp_path / "n.npy", n)
    assert types.compare("normal_angle_deg", tmp_path / "n.npy",
                         tmp_path / "n.npy") < 1e-3

    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps({"cameras": [
        {"K": [[500, 0, 3], [0, 500, 2], [0, 0, 1]], "width": 6,
         "height": 4}]}))
    assert types.validate("camera_seq", cs)["fx"] == 500
    assert _resolve({"objects": [{"az": 3}]}, "objects.0.az") == 3


def test_config_paths_and_mirror(tmp_path, monkeypatch):
    from verdi.core import config, offline, paths

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[paths]\nweights = "/bucket/w"\n[network]\n'
        'github_mirror = "/m"\n[build]\ncuda_home = "/c"\ncuda_arch = "12.0"\n')
    monkeypatch.setenv("VERDI_HOME", str(home))
    config.load.cache_clear()
    assert str(paths.weights_dir("x")) == "/bucket/w/x"
    env = paths.node_env("x")
    assert env["GIT_CONFIG_KEY_0"] == "url.file:///m/github.com/.insteadOf"
    assert env["GIT_CONFIG_KEY_1"] == "protocol.file.allow"
    assert config.build_cfg(str(home))["cuda_arch"] == "12.0"
    config.load.cache_clear()

    class M:
        node_dir = tmp_path
    (tmp_path / "manifest.toml").write_text(
        'repo = "https://github.com/a/b.git"\nx = "https://github.com/c/d"')
    (tmp_path / "uv.lock").write_text(
        'source = { git = "https://github.com/e/f.git?rev=1#abc" }')
    assert offline.github_repos(M) == {"a/b", "c/d", "e/f"}


def test_weights_skip_when_present(tmp_path, monkeypatch):
    import hashlib

    from verdi.core import install

    f = tmp_path / "w" / "k" / "model.bin"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"abc")
    h = hashlib.sha256(b"abc").hexdigest()
    w = {"url": "https://example.invalid/model.bin", "sha256": h}
    assert install._already_done("k", w, f.parent) == f"sha256:{h}"
    assert install._already_done("k", {**w, "sha256": "0" * 64},
                                 f.parent) is None
    hf = {"hf": "org/m", "revision": "r1"}
    assert install._already_done("h", hf, tmp_path / "h") is None
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / install.HF_MARKER).write_text("hf:org/m@r1")
    assert install._already_done("h", hf, tmp_path / "h") == "hf:org/m@r1"


def test_image_psnr(tmp_path):
    a = io.write_image_seq(tmp_path / "a", [np.zeros((4, 4, 3), np.uint8)])
    b = io.write_image_seq(tmp_path / "b", [np.full((4, 4, 3), 10, np.uint8)])
    assert types.compare("image_psnr", a, a) == 99.0
    assert 28 < types.compare("image_psnr", a, b) < 29


def test_trajectory_ate_frame_index(tmp_path):
    def traj(path, frames):
        Ts = []
        for f in frames:
            M = np.eye(4)
            M[:3, 3] = [f, 0.1 * f * f, 0]
            Ts.append(M.tolist())
        path.write_text(json.dumps({"T_world_cam": Ts, "frame_index": frames}))
        return path

    ref = traj(tmp_path / "r.json", list(range(20)))
    run = traj(tmp_path / "p.json", [f for f in range(20) if f != 7])
    assert types.compare("trajectory_ate", run, ref) < 1e-9


def test_pointmap_type(tmp_path):
    pm = np.zeros((4, 5, 3), np.float32)
    pm[..., 2] = 2.0
    np.save(tmp_path / "p.npy", pm)
    s = types.validate("pointmap", tmp_path / "p.npy")
    assert s["median_z"] == 2.0 and s["valid_ratio"] == 1.0


def test_scale_sidecar_and_metric_inputs(tmp_path):
    from types import SimpleNamespace

    from verdi.core import envelope
    from verdi.core.manifest import Port

    d = io.write_depth(tmp_path / "d.npy", np.ones((4, 5)))
    assert types.validate("depth", d)["scale_status"] == "unspecified"
    seq = tmp_path / "seq"
    seq.mkdir()
    io.write_depth(seq / "000000.npy", np.ones((4, 5)))
    io.write_scale(seq, "relative", "arbitrary", "vggt")
    s = types.validate("depth_seq", seq)
    assert s["scale_status"] == "relative" and s["scale_source"] == "vggt"
    with pytest.raises(ValueError):
        io.write_scale(d, "metres?", "m", "x")

    def task(scale):
        return SimpleNamespace(name="t", inputs={"depths": Port("depths", "depth_seq", scale=scale)},
                               params={}, outputs={})
    m = SimpleNamespace(name="n")
    with pytest.raises(envelope.RequestError, match="needs metric data"):
        envelope.build_request(m, task("metric"), {"depths": str(seq)}, {}, "cpu", tmp_path)
    req = envelope.build_request(m, task("any"), {"depths": str(seq)}, {}, "cpu", tmp_path)
    assert req["inputs"]["depths"]["summary"]["scale_status"] == "relative"
    io.write_scale(seq, "metric", "metres", "aruco_scale.scale_align")
    envelope.build_request(m, task("metric"), {"depths": str(seq)}, {}, "cpu", tmp_path)


def test_input_content_digest(tmp_path, monkeypatch):
    import hashlib
    from verdi.core import hashing

    f = tmp_path / "d.npy"; f.write_bytes(b"abc")
    (tmp_path / "d.npy.scale.json").write_text('{"scale_status": "metric"}')
    r = hashing.digest(f)
    assert r["content_sha256"] == hashlib.sha256(b"abc").hexdigest() and r["hash_verified"]
    assert r["sidecars"]["d.npy.scale.json"] == hashlib.sha256(b'{"scale_status": "metric"}').hexdigest()
    d = tmp_path / "seq"; (d / "sub").mkdir(parents=True)
    (d / "b.txt").write_bytes(b"2"); (d / "a.txt").write_bytes(b"1"); (d / "sub" / "c.txt").write_bytes(b"33")
    lines = "".join(f"{n}\t{len(c)}\t{hashlib.sha256(c).hexdigest()}\n" for n, c in
                    (("a.txt", b"1"), ("b.txt", b"2"), ("sub/c.txt", b"33")))
    r = hashing.digest(d)
    assert r["manifest_sha256"] == hashlib.sha256(lines.encode()).hexdigest() and r["files"] == 3
    (d / "scale.json").write_text("{}")
    assert hashing.digest(d)["manifest_sha256"] != r["manifest_sha256"]     # sidecar in the manifest
    monkeypatch.setenv("VERDI_HASH_MAX_BYTES", "1")
    w = hashing.digest(d)
    assert w["hash_verified"] is False and "manifest_sha256" not in w and "weak_fingerprint" in w


def test_splatfacto_scale_resolution():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "nodes" / "splatfacto" / "entry.py"
    spec = importlib.util.spec_from_file_location("splat_entry", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    r = m.resolve_scale
    assert r({"trajectory": None, "depth": None})["scale_status"] == "unknown"
    assert r({"trajectory": "metric", "depth": "metric"})["scale_status"] == "metric"
    assert r({"trajectory": None, "depth": "metric"})["scale_status"] == "metric"
    assert r({"trajectory": "relative", "depth": None})["scale_status"] == "relative"
    c = r({"trajectory": "relative", "depth": "metric"})
    assert c["scale_status"] == "unknown" and c["conflict"] is True
    assert r({"trajectory": "metric_from_input_poses", "depth": "metric"})["scale_status"] == "metric_from_input_poses"
