"""Calibration variants of the synthetic rig (make_fixture.py) for the
projection modes: supplied rectification, K_rect, an inconsistent supplied
rectification (R1/R2 from extrinsics rotated 0.5 deg) and a wide fisheye
(negative k1) on which cv2.fisheye.stereoRectify degenerates (fx ~ 6e-4)."""
import json
from pathlib import Path
import cv2
import numpy as np
H = Path(__file__).resolve().parent
c = json.loads((H / "calib.json").read_text())
K = np.array(json.loads((H.parent / "expected" / "rect_camera.json").read_text())["K"])
R, T = np.array(c["R"]), np.array(c["T"])
def rect_for(Rx, Tx):
    R1, R2, *_ = cv2.stereoRectify(np.array(c["left"]["K"]), np.zeros(5), np.array(c["right"]["K"]),
                                   np.zeros(5), (c["width"], c["height"]), Rx, Tx.reshape(3, 1),
                                   flags=cv2.CALIB_ZERO_DISPARITY)
    tx = float((R2 @ Tx)[0])
    P1 = np.hstack([K, np.zeros((3, 1))]); P2 = P1.copy(); P2[0, 3] = K[0, 0] * tx
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    Q = [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1 / tx, 0]]
    return {"R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q}
sup = dict(c, rectification=dict(rect_for(R, T), source="test: pinhole rotations + rect_camera K"))
(H / "calib_supplied.json").write_text(json.dumps(sup, indent=1))
(H / "calib_k_rect.json").write_text(json.dumps(dict(c, K_rect=K.tolist()), indent=1))
Rbad = cv2.Rodrigues(np.array([0, np.radians(0.5), 0]))[0] @ R
bad = dict(c, rectification=dict(rect_for(Rbad, T), source="test: WRONG extrinsics (R rotated 0.5 deg)"))
(H / "calib_supplied_inconsistent.json").write_text(json.dumps(bad, indent=1))
wide = json.loads(json.dumps(c))
wide["left"]["D"] = [-0.03, 0.0, -0.003, 0.0]; wide["right"]["D"] = [-0.03, 0.0, -0.003, 0.0]
wide["note"] = "wide fisheye analogue: cv2.fisheye.stereoRectify degenerates (fx ~ 6e-4)"
(H / "calib_degenerate.json").write_text(json.dumps(wide, indent=1))
print("written")
