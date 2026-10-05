"""SC-01 fixtures (run from the repo root on 063). Sources: mapanything tum_desk fixture."""
import json, shutil
from pathlib import Path
import numpy as np, cv2
R = Path("nodes"); F = R / "sfm/tests/fixture"; F.mkdir(parents=True, exist_ok=True)
src = R / "mapanything/tests/fixture"
if (F / "tum_desk").exists(): shutil.rmtree(F / "tum_desk")
shutil.copytree(src / "tum_desk", F / "tum_desk")
for n in ("tum_desk_camera.json", "tum_desk_gt.json"): shutil.copy(src / n, F / n)
cam = json.load(open(F / "tum_desk_camera.json")); K = np.array(cam["K"], float); w, h = cam["width"], cam["height"]
print("tum camera", cam)
# synthetic fisheye: render every fisheye pixel from the pinhole frame
Kf = K.copy(); Kf[0, 0] *= 0.8; Kf[1, 1] *= 0.8
D = np.array([0.05, -0.01, 0.003, -0.0005])
uv = np.stack(np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)), -1).reshape(-1, 1, 2)
und = cv2.fisheye.undistortPoints(uv, Kf, D, P=K).reshape(h, w, 2).astype(np.float32)
fe = F / "tum_fisheye"; (fe / "frames").mkdir(parents=True, exist_ok=True)
for p in sorted((F / "tum_desk").glob("*.png")):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    out = cv2.remap(im, und[..., 0], und[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(str(fe / "frames" / p.name), out)
json.dump({"K": Kf.tolist(), "width": w, "height": h, "dist": D.tolist(), "model": "fisheye",
           "note": "synthetic: tum_desk frames re-rendered through this OPENCV_FISHEYE camera (tests/fixture/make_fixture.py)"},
          open(fe / "camera.json", "w"), indent=1)
inside = ((und[..., 0] >= 0) & (und[..., 0] <= w - 1) & (und[..., 1] >= 0) & (und[..., 1] <= h - 1)).mean()
print("fisheye pixels with pinhole source:", round(float(inside), 3))
b = F / "black"; b.mkdir(exist_ok=True)
for i in range(4): cv2.imwrite(str(b / f"{i:06d}.png"), np.zeros((240, 320, 3), np.uint8))
m = F / "mixed_res"; m.mkdir(exist_ok=True)
for i, p in enumerate(sorted((F / "tum_desk").glob("*.png"))[:3]):
    im = cv2.imread(str(p)); im = cv2.resize(im, (320, 240)) if i == 2 else im; cv2.imwrite(str(m / f"{i:06d}.png"), im)
c3 = dict(cam); c3["dist"] = [0.01, 0.0, 0.0, 0.0, 0.002]; json.dump(c3, open(F / "camera_k3.json", "w"))
c2 = dict(cam); c2["K"] = (K * np.array([[1.01], [1.01], [1]])).tolist()
json.dump({"cameras": [cam] * 25 + [c2]}, open(F / "cameras_differ.json", "w"))
print("ok")
