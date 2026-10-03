"""Build the plane_layout fixtures (run from nodes/plane_layout with the node venv).

room.ply: synthetic room with known planes, expressed in an OpenCV camera
frame (camera 1.5 m above the floor, pitched 20 deg down, rolled 5 deg):
  floor z=0 (4 x 5 m), walls x=0 and y=5, ceiling z=2.6 (half), table top
  z=0.75 (1.2 x 0.8 m), a 0.3 m box on the floor, 5 mm gaussian noise,
  1% random outliers.
bedroom_depth.npy / bedroom_camera.json: depth_pro output for
nodes/sam2/tests/fixture/bedroom/000000.jpg (2x subsampled), real case.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
DENS = 1600  # points per m^2


def rect(o, a, b):
    o, a, b = map(np.asarray, (o, a, b))
    n = int(DENS * np.linalg.norm(a) * np.linalg.norm(b))
    s, t = rng.random((n, 1)), rng.random((n, 1))
    return o + s * a + t * b


parts = [
    rect([0, 0, 0], [4, 0, 0], [0, 5, 0]),            # floor
    rect([0, 0, 0], [0, 5, 0], [0, 0, 2.6]),          # wall x=0
    rect([0, 5, 0], [4, 0, 0], [0, 0, 2.6]),          # wall y=5
    rect([0, 2.5, 2.6], [4, 0, 0], [0, 2.5, 0]),      # ceiling (back half)
    rect([1.5, 2.5, 0.75], [1.2, 0, 0], [0, 0.8, 0]), # table top
    rect([3.0, 1.5, 0.3], [0.3, 0, 0], [0, 0.3, 0]),  # box top
    rect([3.0, 1.5, 0.0], [0.3, 0, 0], [0, 0, 0.3]),  # box side
    rect([3.0, 1.5, 0.0], [0, 0.3, 0], [0, 0, 0.3]),  # box side
]
P = np.concatenate(parts)
P += rng.normal(0, 0.005, P.shape)
out = rng.random((len(P) // 100, 3)) * [4, 5, 2.6]
P = np.concatenate([P, out])

# camera: at (2, 0.5, 1.5), looking along +y, pitch 20 deg down, roll 5 deg
pitch, rollr = np.radians(20), np.radians(5)
fwd = np.array([0, np.cos(pitch), -np.sin(pitch)])
right0 = np.array([1.0, 0, 0])
down0 = np.cross(fwd, right0)
right = np.cos(rollr) * right0 + np.sin(rollr) * down0
down = np.cross(fwd, right)
R_wc = np.stack([right, down, fwd], 1)
C = np.array([2.0, 0.5, 1.5])
Pc = (P - C) @ R_wc  # = R_wc^T (P - C)
hdr = ("ply\nformat binary_little_endian 1.0\n"
       f"element vertex {len(Pc)}\nproperty float x\nproperty float y\n"
       "property float z\nend_header\n")
with open(HERE / "room.ply", "wb") as f:
    f.write(hdr.encode())
    f.write(Pc.astype("<f4").tobytes())
up_cam = R_wc.T @ np.array([0, 0, 1.0])
print("points", len(Pc), "up in cam", up_cam.round(4),
      "floor d (cam)", round(float(1.5), 4))

dp = Path("../depth_pro")
d = np.load(dp / "tests/expected/bedroom0_depth.npy")[::2, ::2]
np.save(HERE / "bedroom_depth.npy", d.astype(np.float32))
cam = json.loads(sorted((dp / "tests/fixture").glob("*.json"))[0].read_text())
(HERE / "bedroom_camera.json").write_text(json.dumps(cam))
print("bedroom depth", d.shape, float(np.median(d)), cam)
