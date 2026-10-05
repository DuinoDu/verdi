#!/usr/bin/env bash
# Object shape + pose initialisation: SAM 3 text mask -> SAM 3D Objects in
# the METRIC mode (depth + camera) -> FoundationPose on the baked metric mesh.
# Shows the mesh / scale / pose contract and the guard against normalised
# meshes.  Default inputs: the YCB mustard RGB-D fixture.
#
#   source ~/verdi_env.sh && cd <verdi checkout>
#   bash pipelines/real2sim/example_object.sh /tmp/verdi_obj cuda:0
#
# Real data: IMAGE (rgb), DEPTH (.npy metres aligned to IMAGE, e.g.
# foundation_stereo on the rectified left image), CAMERA (K of IMAGE, dist 0),
# TEXT (noun phrase).
set -euo pipefail
OUT=${1:-/tmp/verdi_obj}; DEV=${2:-cuda:0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
F=$ROOT/nodes/sam3d_objects/tests/fixture/mustard
IMAGE=${IMAGE:-$F/color.jpg}; DEPTH=${DEPTH:-$F/depth.npy}; CAMERA=${CAMERA:-$F/camera.json}
TEXT=${TEXT:-"mustard bottle"}
mkdir -p "$OUT"
j() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(eval(sys.argv[2]))" "$@"; }

# 1. instance mask (id i = boxes[i-1], sorted by score)
verdi run sam3 --task segment_text -i image="$IMAGE" -p text="$TEXT" \
  --out "$OUT/seg" --device "$DEV" -q > "$OUT/seg.json"
echo "sam3: $(j "$OUT/seg/boxes.json" "[(b['id'], b['label'], round(b['score'],3)) for b in d['boxes']]")"

# 2. shape + metric pose of instance 1 (depth + camera => metres)
verdi run sam3d_objects --task reconstruct -i image="$IMAGE" -i mask="$OUT/seg/mask.png" \
  -i depth="$DEPTH" -i camera="$CAMERA" -p mask_id=1 \
  --out "$OUT/s3d" --device "$DEV" -q > "$OUT/s3d.json"
j "$OUT/s3d/poses.json" "{k: d['poses'][0][k] for k in ('scale_status','scale','size_metric_mesh','mesh','mesh_metric')}"
#    meshes/obj_1.glb        normalised, p_cam = S_cam_glb @ [v,1]  (scale sidecar: relative)
#    meshes_metric/obj_1.glb scale baked, p_cam = R v + t with T_cam_obj (metric)

# 3. FoundationPose on the METRIC mesh (no scale applied again)
verdi run foundationpose --task estimate -i image="$IMAGE" -i depth="$DEPTH" \
  -i mask="$OUT/seg/mask.png" -i camera="$CAMERA" -i mesh="$OUT/s3d/meshes_metric/obj_1.glb" \
  -p mask_id=1 --out "$OUT/fp" --device "$DEV" -q > "$OUT/fp.json"
python3 - "$OUT" <<'PY'
import json, sys
import numpy as np
o = sys.argv[1]
a = np.array(json.load(open(f"{o}/s3d/poses.json"))["poses"][0]["T_cam_obj"])
p = json.load(open(f"{o}/fp/poses.json"))["poses"][0]
b = np.array(p["T_cam_obj"])
dR = a[:3, :3].T @ b[:3, :3]
ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
print(f"foundationpose valid={p['valid']} depth_inlier={p['diagnostics']['depth_inlier_ratio']:.2f}"
      f" mask_iou={p['diagnostics'].get('mask_iou', float('nan')):.2f}")
print(f"|t_fp - t_sam3d| = {np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000:.1f} mm, rotation diff {ang:.1f} deg"
      " (same mesh frame; differences = refinement, or symmetry for symmetric objects)")
PY

# 4. guard: the normalised mesh is refused as a metric mesh
if verdi run foundationpose --task estimate -i image="$IMAGE" -i depth="$DEPTH" \
     -i mask="$OUT/seg/mask.png" -i camera="$CAMERA" -i mesh="$OUT/s3d/meshes/obj_1.glb" \
     -p mask_id=1 --out "$OUT/fp_bad" --device "$DEV" -q > "$OUT/fp_bad.json"; then
  echo "ERROR: normalised mesh was accepted"; exit 1
else
  echo "guard ok: $(j "$OUT/fp_bad.json" "d['error']['message'][:120]")"
fi
