# real2sim -> verdi: head fisheye stereo, table / pepper / basket

real2sim owns time sync, calibration conversion, observation fusion, scene
compilation and physics evaluation. verdi owns model inference and the data
contract of every output. This page lists the nodes, the exact calls and
what each output means. Every run returns a JSON result on stdout
(`status`, `outputs.<name>.path` + `summary`, `error.kind/message/hint`,
`provenance`: node version, upstream commit, weight revisions + sha256,
env lock hash, GPU, duration, `run_dir` with `request.json`, `result.json`,
`log.txt`). Failures exit non-zero with `status=error`; no node falls back
to another model or fabricates output.

## Where verdi runs and how to call it

| host | role | how |
|---|---|---|
| **apex** (RTX 5090 32 GB, same machine as real2sim) | primary: local CLI | `source ~/verdi_env.sh` then `verdi run ...` with local paths (no upload / download) |
| 063 (8x RTX 5090, offline) | development / reference host, all nodes set up | not reachable from apex directly; only for verdi maintainers |

```bash
source ~/verdi_env.sh            # VERDI_HOME=~/verdi_home, PATH += ~/ws/verdi/.venv/bin
verdi list                       # nodes, setup state, last test status
verdi describe <node>            # tasks, typed inputs/outputs, params
verdi run <node> --task <task> -i name=path ... -p key=value \
    --out /abs/out/dir --device cuda:0      # or --device cpu for algorithm nodes
```

* Inputs are file / directory paths (frame dirs `%06d.png|jpg`, videos are
  split automatically where `image_seq` is expected).
* `--out` fixes the output directory (otherwise
  `$VERDI_HOME/runs/<node>/<run_id>/out`). `result.json` and `log.txt` stay
  in `provenance.run_dir`.
* Cache: weights / venvs / upstream checkouts in `$VERDI_HOME` (set up once
  with `verdi setup <node>`); results are never cached: every run is new.
* Python: `subprocess.run(["verdi", "run", ...], capture_output=True)` and
  `json.loads(stdout)`; or `from verdi.core import manifest, runner;
  runner.run(manifest.load("<node>"), task, inputs, params, device=...)`.
* One GPU on apex: pass `--device cuda:0`; runs are one-shot processes
  (model load each call, `*_seq` tasks load once per sequence).

## Pipeline for the head stereo rig

```
SBS video 2560x720 (each eye 1280x720) + calib.json
  └─ stereo_rectify.rectify (CPU) ──> rectified left/right %06d.png, camera.json (K_rect),
        │                             valid.png, rectification.json (R1,R2,P1,P2,Q,T_left_rect, baseline,
        │                             epipolar check: median |dy| px; FAILS if > max_epipolar_dy_px)
        ├─ foundation_stereo.estimate_depth_seq ──> metric z-depth (rectified-left frame), valid, info
        ├─ sam3.segment_text / track_text / track_prompts ──> instance masks + per-frame visibility
        ├─ depth_anything_3.reconstruct  (visual poses + reference_poses = FK -> pose_eval)
        ├─ mapanything.reconstruct       (images + K [+ depth] -> metric depth / poses)
        └─ foundationpose.estimate / track (mesh + RGB-D + K + mask -> T_cam_obj + valid)
```

### 1. Rectification (who does what)

`stereo_rectify` implements the OpenCV fisheye (Kannala-Brandt k1..k4) or
pinhole stereo rectification; **real2sim provides the calibration** in this
json (one eye's size; extrinsics in the cv2.stereoCalibrate convention):

```json
{"model": "fisheye", "width": 1280, "height": 720, "units": "m",
 "left":  {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "D": [k1,k2,k3,k4]},
 "right": {"K": [[...]], "D": [k1,k2,k3,k4]},
 "R": [[...3x3...]], "T": [-0.0605, ty, tz]}
```
`X_right = R @ X_left + T` (so T ~ [-baseline, 0, 0]); `T_right_left` (4x4)
may be given instead. If your calibration uses another model (e.g. a
different fisheye or omnidirectional model), convert it or tell us.

```bash
verdi run stereo_rectify --task rectify -i stereo=clip_sbs.mp4 -i calib=calib.json \
  -p balance=0.0 --out $OUT/rect --device cpu
```
Depth computed on the rectified images lives in the **rectified left frame**;
`rectification.T_left_rect` maps it to the original left camera (OpenCV).

### 2. FoundationStereo metric depth

```bash
B=$(python -c "import json;print(json.load(open('$OUT/rect/rectification.json'))['baseline_rect_m'])")
verdi run foundation_stereo --task estimate_depth_seq -i left=$OUT/rect/left -i right=$OUT/rect/right \
  -i camera=$OUT/rect/camera.json -i valid=$OUT/rect/valid_left.png -p baseline=$B \
  -p lr_check=true --out $OUT/fs --device cuda:0
```
z = fx * baseline / (disparity + doffs) at the input resolution (doffs = 0
after stereo_rectify); `info.json` records fx, baseline, inference size,
K at inference size and the disparity rescale when `scale < 1`. There is
no calibrated confidence: `valid` = geometric checks (+ left-right
consistency with `lr_check=true`, `lr_error_px`).

### 3. SAM 3 instances

```bash
verdi run sam3 --task segment_text -i image=frame.png -p text="table. green pepper. basket" --out $OUT/seg
verdi run sam3 --task track_prompts -i frames=$OUT/rect/left -i prompts=prompts.json --out $OUT/trk
```
`prompts.json`: `{"points": [{"xy": [x, y], "positive": true, "obj_id": 1, "frame": 0, "label": "pepper"}], "boxes": [{"xyxy": [x0,y0,x1,y1], "obj_id": 2, "frame": 0, "label": "basket"}]}`.
`tracks.json`: `visible[t][i]` (non-empty SAM 3 mask for `ids[i]` on frame
t), `score[t][i]`, per-frame area / bbox. Occluded, out-of-view and lost
are not distinguished by SAM 3.

### 4. Depth Anything 3 / MapAnything (multi-view)

* Visual poses (independent of FK): `depth_anything_3 reconstruct -i
  frames=... -i reference_poses=fk.json` -> `pose_eval.json` (Sim(3) and
  SE(3) ATE, rotation error, step-length ratio). Same for `mapanything`.
* Geometry with known poses: `-i poses=fk.json -i camera=K.json`
  (conditioning). `info.scale_status` = `input_pose_scale` (DA3,
  align_to_input_ext_scale=true) and `pose_source` states the poses came
  from the input: such a run cannot validate FK.
* Calibrated camera, no poses (recommended for the head camera):
  `depth_anything_3 reconstruct -i frames=... -i camera=K_rect.json`
  -> poses stay visual, the nested metric scale uses the KNOWN focal
  (`info.metric_focal = known`). Without K the nested "metres" follow its
  own focal estimate, which was 1.7x off on TUM fr1 desk (depth 1.8-2x too
  far). Single image: `estimate_depth -p model=metric -i camera=K.json`.
* Measured (independent ground truth, see each node's NOTES.md): TUM mocap
  26 frames, DA3 visual poses ATE 13 mm (Sim3) / 52 mm (rigid); MapAnything
  36 / 50 mm. TUM Kinect depth: DA3METRIC + K abs-rel 0.024; DA3 nested +
  K 3-frame reconstruct 0.10; MapAnything images + K 0.32, + depth 0.03.
  Monocular metric scale is NOT reliable enough to replace stereo or a
  marker for grasp-level geometry; use FoundationStereo depth (synthetic
  60.5 mm rig: abs-rel 0.4-0.5 %) or condition on it.
* DA3 models: `model=nested` (DA3NESTED-GIANT-LARGE-1.1, metres),
  `giant` / `large` (DA3-*-1.1, relative scale), `use_ray_pose=true`
  for the ray head; `estimate_depth -p model=metric -i camera=K.json` for
  DA3METRIC-LARGE (metres = focal * output / 300).

### 5. FoundationPose

```bash
verdi run foundationpose --task track -i frames=rgb_dir -i depths=depth_dir -i mask=mask0.png \
  -i camera=K.json -i mesh=basket.glb [-i masks=$OUT/trk/masks] -p mask_id=2 --out $OUT/fp
```
RGB and depth must be aligned pinhole images with the K given (e.g. the
rectified left image + FoundationStereo depth + rectified K); mesh in
metres. Per frame: `T_cam_obj` (mesh -> OpenCV camera) + `valid` +
diagnostics (`depth_inlier_ratio`, `occluded_ratio`, `mask_iou`). The
model-free FoundationPose mode is not packaged.

## Sample data we need from real2sim (train split only)

Please put on apex, e.g. under `~/verdi_inputs/real2sim_train/<clip_id>/`:

1. `sbs.mp4` (or `sbs/%06d.png`): 3-10 s of the head stereo stream, 2560x720,
   left eye on the left half; `timestamps.json` `{"frame_index": [...],
   "t_ns": [...]}` if you want ids/time carried through.
2. `calib.json` in the format above (both eyes' K + fisheye D, R, T, units).
3. Optional `fk_T_world_cam.json`: `{"T_world_cam": [4x4 per frame], "frame_index": [...]}`
   of the LEFT eye optical frame (OpenCV axes), same frames as 1. Used only
   as `reference_poses` (evaluation) or explicit conditioning runs.
4. Optional object meshes in metres (`basket.glb`, ...) for FoundationPose,
   and one first-frame point/box per object for SAM 3 prompts.
5. Any independent measurement you have (tape-measured table height, basket
   size, ArUco board in view) to check metric scale; we will not tune
   anything on grasp / throw results.
