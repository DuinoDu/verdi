#!/usr/bin/env bash
# End-to-end stereo example on the synthetic fisheye rig fixture (analytic
# ground truth): SBS frames -> stereo_rectify -> foundation_stereo (seq)
# -> depth error vs truth, all through the verdi CLI.
#
#   source ~/verdi_env.sh            # apex: VERDI_HOME, verdi on PATH
#   bash pipelines/real2sim/example_stereo.sh /tmp/verdi_example cuda:0
#
# For real data replace SBS/CALIB with your clip (dir of %06d.png or .mp4)
# and calibration json (format: pipelines/real2sim/README.md).
set -euo pipefail
OUT=${1:-/tmp/verdi_example}; DEV=${2:-cuda:0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SBS=${SBS:-$ROOT/nodes/stereo_rectify/tests/fixture/sbs}
CALIB=${CALIB:-$ROOT/nodes/stereo_rectify/tests/fixture/calib.json}
GT=${GT:-$ROOT/nodes/stereo_rectify/tests/expected/rect_depth}
mkdir -p "$OUT"

verdi run stereo_rectify --task rectify -i stereo="$SBS" -i calib="$CALIB" \
  --out "$OUT/rect" --device cpu -q > "$OUT/rect.result.json"
B=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['baseline_rect_m'])" "$OUT/rect/rectification.json")
echo "rectified: baseline_rect_m=$B, epipolar check:" \
  "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['epipolar_check_frame0'])" "$OUT/rect/rectification.json")"

verdi run foundation_stereo --task estimate_depth_seq -i left="$OUT/rect/left" \
  -i right="$OUT/rect/right" -i camera="$OUT/rect/camera.json" \
  -i valid="$OUT/rect/valid_left.png" -p baseline="$B" -p lr_check=true \
  --out "$OUT/fs" --device "$DEV" -q > "$OUT/fs.result.json"
python3 - "$OUT" "$GT" <<'PY'
import glob, json, sys
import numpy as np
out, gt = sys.argv[1], sys.argv[2]
r = json.load(open(f"{out}/fs.result.json"))
print("foundation_stereo:", r["status"], "duration", r["provenance"]["duration_sec"], "s",
      "weights", r["provenance"]["weights"])
if gt and glob.glob(f"{gt}/*.npy"):
    for g in sorted(glob.glob(f"{gt}/*.npy")):
        G = np.load(g); P = np.load(f"{out}/fs/depth/{g.split('/')[-1]}")
        m = (G > 0) & (P > 0)
        e = np.abs(P[m] - G[m])
        print(f"  {g.split('/')[-1]}: valid {float((P > 0).mean()):.3f}, abs-rel {float((e / G[m]).mean()):.4f},"
              f" median |err| {float(np.median(e)) * 1000:.2f} mm")
PY
echo "outputs in $OUT (rect/: rectified frames, camera.json, rectification.json; fs/: depth, valid, info.json)"
