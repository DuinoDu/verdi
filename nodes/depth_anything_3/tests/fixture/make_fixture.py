"""Build the DA3 fixtures from other nodes' committed fixtures.

tum_desk/: every 8th frame of mast3r_slam's TUM RGB-D fr1_desk subset
  (205 frames = every 3rd rgb frame), UNDISTORTED with the TUM fr1
  calibration (cv2.undistort, new K = old K) so the frames are pinhole;
  tum_desk_camera.json = that K (dist 0); tum_desk_gt.json = matching TUM
  mocap ground-truth T_world_cam (metres) - real, independent poses.
tum_desk_gt_x2.json: the same poses with translations x2 (pose-scale test).
mustard/: foundationpose's first mustard frame: RGB, sensor depth (metres)
  and its pinhole K (real RGB-D ground truth for metric depth checks).
Run with any python with numpy + opencv: python make_fixture.py
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NODES = HERE.parents[2]
src = NODES / "mast3r_slam" / "tests"
cam = json.loads((src / "fixture" / "fr1_desk_camera.json").read_text())
K = np.asarray(cam["K"], dtype=np.float64)
dist = np.asarray(cam["dist"], dtype=np.float64)
gt = json.loads((src / "expected" / "gt_trajectory.json").read_text())
frames = sorted((src / "fixture" / "fr1_desk").glob("*.jpg"))
step = 8
out = HERE / "tum_desk"
shutil.rmtree(out, ignore_errors=True)
out.mkdir()
sel = list(range(0, len(frames), step))
by_index = {int(i): j for j, i in enumerate(gt["frame_index"])}
Ts, names, stamps = [], [], []
for k, i in enumerate(sel):
    img = cv2.imread(str(frames[i]))
    und = cv2.undistort(img, K, dist, None, K)
    cv2.imwrite(str(out / f"{k:06d}.png"), und)
    j = by_index[i]
    Ts.append(gt["T_world_cam"][j])
    names.append(frames[i].name)
    stamps.append(gt["timestamps"][j])
(HERE / "tum_desk_camera.json").write_text(json.dumps(
    {"K": K.tolist(), "width": cam["width"], "height": cam["height"],
     "dist": [0, 0, 0, 0, 0]}, indent=1))
meta = {"source": "TUM RGB-D fr1_desk mocap ground truth (metres), "
        f"every {step}-th frame of mast3r_slam fixture fr1_desk",
        "source_frames": names, "timestamps": stamps,
        "frame_index": list(range(len(sel)))}
(HERE / "tum_desk_gt.json").write_text(json.dumps({"T_world_cam": Ts, **meta}, indent=1))
T2 = []
for T in Ts:
    T = np.asarray(T, dtype=np.float64).copy()
    T[:3, 3] *= 2.0
    T2.append(T.tolist())
(HERE / "tum_desk_gt_x2.json").write_text(json.dumps(
    {"T_world_cam": T2, **meta, "source": meta["source"] + "; translations x2"}, indent=1))

fp = NODES / "foundationpose" / "tests" / "fixture" / "mustard"
mo = HERE / "mustard"
shutil.rmtree(mo, ignore_errors=True)
mo.mkdir()
shutil.copy(fp / "rgb" / "000000.jpg", mo / "rgb.jpg")
shutil.copy(fp / "depth" / "000000.npy", mo / "depth.npy")
shutil.copy(fp / "camera.json", mo / "camera.json")
print(len(sel), "tum frames")
