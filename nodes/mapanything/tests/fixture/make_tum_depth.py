"""tum_depth/: 3 RGB-D frames of TUM RGB-D fr1_desk (CC BY 4.0,
https://cvg.cit.tum.de/data/datasets/rgbd-dataset): RGB undistorted with
the fr1 calibration (cv2.undistort, new K = old K), Kinect depth (already
registered to RGB by the dataset, /5000 -> metres) remapped with the same
undistortion (nearest), rgb/depth pairs with |dt| < 5 ms.
usage: python make_tum_depth.py <rgbd_dataset_freiburg1_desk dir> <out dir>"""
import sys, json
from pathlib import Path
import cv2, numpy as np
src, out = Path(sys.argv[1]), Path(sys.argv[2])
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.0]])
D = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
def lst(name):
    rows = [l.split() for l in (src / name).read_text().splitlines() if not l.startswith("#")]
    return [(float(t), p) for t, p in rows]
rgb, dep = lst("rgb.txt"), lst("depth.txt")
pairs = []
for t, p in rgb:
    td, pd = min(dep, key=lambda x: abs(x[0] - t))
    if abs(td - t) < 0.005:
        pairs.append((t, p, td, pd))
sel = [pairs[int(i)] for i in np.linspace(0, len(pairs) - 1, 5)[1:4]]
m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (640, 480), cv2.CV_32FC1)
(out / "rgb").mkdir(parents=True, exist_ok=True); (out / "depth").mkdir(parents=True, exist_ok=True)
meta = []
for i, (t, p, td, pd) in enumerate(sel):
    im = cv2.imread(str(src / p))
    cv2.imwrite(str(out / "rgb" / f"{i:06d}.png"), cv2.remap(im, m1, m2, cv2.INTER_LINEAR))
    d = cv2.imread(str(src / pd), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
    d = cv2.remap(d, m1, m2, cv2.INTER_NEAREST)
    np.save(out / "depth" / f"{i:06d}.npy", d.astype(np.float32))
    meta.append({"index": i, "rgb": p, "depth": pd, "dt_ms": round((td - t) * 1000, 2)})
(out / "camera.json").write_text(json.dumps({"K": K.tolist(), "width": 640, "height": 480, "dist": [0, 0, 0, 0, 0]}, indent=1))
(out / "source.json").write_text(json.dumps({"dataset": "TUM RGB-D fr1_desk", "frames": meta}, indent=1))
print(meta)
