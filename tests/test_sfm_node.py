"""CPU unit tests of the sfm node's pure parts (no pycolmap run)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sfm_entry", ROOT / "nodes" / "sfm" / "entry.py")
sfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfm)
from verdi.sdk import NodeError  # noqa: E402

K = [[400.0, 0, 320], [0, 401.0, 240], [0, 0, 1]]


def test_camera_models():
    assert sfm.camera_to_colmap({"K": K, "width": 640, "height": 480, "dist": [0] * 5}, 640, 480)[0] == "PINHOLE"
    m, p, _ = sfm.camera_to_colmap({"K": K, "width": 640, "height": 480, "dist": [0.1, 0.01, 0.001, 0.002]}, 640, 480)
    assert m == "OPENCV" and p == [400.0, 401.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.002]
    m, p, _ = sfm.camera_to_colmap({"K": K, "width": 640, "height": 480, "dist": [0.1, 0.01, 0.0, 0.0], "model": "fisheye"}, 640, 480)
    assert m == "OPENCV_FISHEYE" and len(p) == 8


@pytest.mark.parametrize("cam", [
    {"K": K, "width": 640, "height": 480, "dist": [0.1, 0, 0, 0, 0.01]},          # k3
    {"K": K, "width": 320, "height": 240, "dist": [0] * 5},                       # size
    {"K": K, "width": 640, "height": 480, "dist": [0.1, 0.0], "model": "fisheye"},  # fisheye needs 4
    {"K": [[400.0, 2.0, 320], [0, 401.0, 240], [0, 0, 1]], "width": 640, "height": 480},  # skew
])
def test_camera_refusals(cam):
    with pytest.raises(NodeError):
        sfm.camera_to_colmap(cam, 640, 480)


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(20, 3))
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    G = 2.5 * (R @ P.T).T + np.array([1.0, -2.0, 0.5])
    s, R2, t = sfm.umeyama(P, G)
    assert abs(s - 2.5) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [1.0, -2.0, 0.5])
