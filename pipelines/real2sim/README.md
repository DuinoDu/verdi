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
| 063 (8x RTX 5090, offline) | reference host, all 38 nodes set up | `scripts/verdi_remote.py` from apex (ssh/rsync via the ubuntu relay) |

Set up and tested on apex (2026-10-05, commit cf9215a, all fixture tests
pass): stereo_rectify, sam3, foundation_stereo, foundationpose,
depth_anything_3, mapanything. Measured on apex (one-shot `verdi run`,
model load included, RTX 5090):

| run | time | peak VRAM |
|---|---|---|
| sam3 segment_text (1 image) | 9.5 s | 5.8 GiB |
| sam3 track_prompts (6 frames 960x540) | 10.5 s | 6.5 GiB |
| foundation_stereo estimate_depth_seq (2 pairs 640x360, lr_check) | 9.3 s | 3.5 GiB |
| foundation_stereo at 1280x720 (063) | 1.4 s/frame (2.1 with lr_check) | 7.2 GiB |
| foundationpose track (3 frames 640x480) | 8.5 s | 4.9 GiB |
| depth_anything_3 reconstruct (26 frames 512x384, nested, known K) | 11.7 s | 14.9 GiB |
| mapanything reconstruct (26 frames, apache, K) | 17.2 s | 20.3 GiB |

Runnable end-to-end example (synthetic fisheye rig with analytic truth;
set SBS / CALIB / GT to run it on your own clip):

```bash
source ~/verdi_env.sh && cd ~/ws/verdi
bash pipelines/real2sim/example_stereo.sh /tmp/verdi_example cuda:0
# rectified: baseline_rect_m=0.0605, epipolar median |dy| 0.054 px
# 000000.npy: valid 0.970, abs-rel 0.0042, median |err| 1.78 mm
```

Remote runs on 063 from apex (when apex's GPU is busy or for nodes not set
up on apex): `scripts/verdi_remote.py` = same CLI + result contract, inputs
uploaded and outputs downloaded via ssh/rsync over the ubuntu relay
(apex -LAN-> ubuntu -> jump host -> 063; ~30 s overhead per call, 063 is
shared: pick a free GPU):

```bash
python3 scripts/verdi_remote.py foundation_stereo --task estimate_depth \
  -i left=L.png -i right=R.png -i camera=K.json -p baseline=0.0605 \
  --out /abs/out --device cuda:1        # exit 0 ok / 1 node error / 2 transport
# outputs in /abs/out/out/, result.json (paths rewritten to local), log.txt
```

Depth scale contract: every depth / depth_seq / pointcloud summary carries
`scale_status`; producers that are not metric write a sidecar
(`depth.npy.scale.json`, `<dir>/scale.json`, `cloud.ply.scale.json`), and
metric inputs (foundationpose, sam3d_objects, plane_layout, ...) refuse
`relative` / `input_pose_scale` data with a request error.

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

### 6. SAM 3D Objects -> metric object mesh -> FoundationPose

Contract of `sam3d_objects reconstruct` (per object k = mask id):

| output | frame / units | how to use |
|---|---|---|
| `meshes/obj_k.glb` | normalised object frame, glTF +Y up, ~[-0.5, 0.5]^3, scale sidecar `relative` | only with `S_cam_glb`: p_cam = S_cam_glb @ [v, 1] (= R (scale v) + t up to upstream per-axis scale); refused by metric consumers |
| `meshes_metric/obj_k.glb` | same axes, linear part baked: v' = R^T L v (L = S_cam_glb[:3,:3]); metres with depth + camera (sidecar `metric`), else MoGe units (`relative`) | p_cam = R v' + t with `T_cam_obj` = [R \| t]; do NOT apply `scale` again |
| `poses.json` | `T_cam_obj` (rigid, OpenCV), `S_cam_glb`, `bake_glb_to_metric`, `scale` (cbrt det L), `size` / `size_metric_mesh` (m), `scale_status`, `mask_id`, `mesh`, `mesh_metric` | the baked mesh + T_cam_obj is the object-frame mesh for simulation (it is not re-centred: the origin is the upstream object centre) |

Preferred path for shape initialisation = the metric mode (depth + camera
of the same image, e.g. foundation_stereo depth + rectified K). Hidden /
contact surfaces are generated, not measured: verify with the render_mask
/ overlay outputs and FoundationPose diagnostics. Verified on the YCB
mustard RGB-D fixture (063, ~2.5 min incl. model loads):

```bash
bash pipelines/real2sim/example_object.sh /tmp/verdi_obj cuda:0
# sam3: [(1, 'mustard bottle', 0.797)]
# {'scale_status': 'metric', 'scale': 0.1958, 'size_metric_mesh': [0.0997, 0.1957, 0.0576], ...}
#   (real YCB mustard ~0.096 x 0.191 x 0.058 m)
# foundationpose valid=True depth_inlier=0.98 mask_iou=0.95
# |t_fp - t_sam3d| = 12.9 mm, rotation diff 176.3 deg   (near front/back symmetry)
# guard ok: input mesh: .../meshes/obj_1.glb is declared scale_status='relative' ...
```

The exact commands inside (verified against `verdi run --help`:
`verdi run NODE --task T -i name=path -p name=value --out DIR --device D [-q]`):

```bash
verdi run sam3 --task segment_text -i image=IMG -p text="green pepper" --out S --device cuda:0 -q
verdi run sam3d_objects --task reconstruct -i image=IMG -i mask=S/mask.png -i depth=D.npy \
  -i camera=K.json -p mask_id=1 --out O --device cuda:0 -q
verdi run foundationpose --task estimate -i image=IMG -i depth=D.npy -i mask=S/mask.png \
  -i camera=K.json -i mesh=O/meshes_metric/obj_1.glb -p mask_id=1 --out P --device cuda:0 -q
```

Object proposals / relations: reuse `qwen2_5_vl` / `moondream` (verdi
describe them); AACR, relation graphs and scene compilation stay in real2sim.

### 7. SAM 3 prompt recovery (object missed by track_text)

`track_text` only creates tracks for phrases detected on `prompt_frame` and
later frames; a phrase that is never detected gets no id
(`metadata.ids_per_phrase[phrase] == []`, `phrases_without_detection`). That
is "never initialised", not "tracked then lost". Recover with
`track_prompts` (SAM 3 tracker of the same checkpoint; no SAM 2) seeded on a
frame where the object is visible:

prompt_set schema (one obj_id per object; all prompts of one obj_id on ONE frame):

```json
{"points": [{"xy": [u, v], "positive": true,  "obj_id": 101, "frame": 9, "label": "green pepper", "source": "<who/what placed it>"},
            {"xy": [u, v], "positive": false, "obj_id": 101, "frame": 9, "label": "green pepper", "source": "..."}],
 "boxes":  [{"xyxy": [x0, y0, x1, y1], "obj_id": 101, "frame": 9, "label": "green pepper", "source": "..."}]}
```

* `xy` / `xyxy`: pixels of the frames passed as `frames` (e.g. the rectified
  left frames); coordinates come from the consumer, verdi never invents them.
* `frame`: 0-based index into the sorted input frames (`rect/left/000009.png`
  = 9 = source frame of that file per `rectification.json frames[9].source`).
* Propagation runs forward to the last frame AND backward to frame 0 from the
  earliest prompted frame (automatic; there is no direction parameter).
* Extra keys such as `source`, `source_frame`, `t_s` are accepted and kept in the
  prompt file as consumer provenance; the node does not interpret them. Store new
  human / VLM prompt files with their own request/result: a track_prompts recovery is
  never reported as a text-prompt success. Zero tracks over all frames after
  track_text means "never initialised", not "occluded everywhere".
* Use obj_ids that do not collide with the `track_text` ids you merge with
  (e.g. 101+); the output id = obj_id.

```bash
verdi run sam3 --task track_prompts -i frames=$RUN/rect/left -i prompts=$RUN/pepper_prompts.json \
  --out $RUN/segment_pepper --device cuda:0 -q
```

Traceability: `tracks.json` gives per id `label` (from the prompt),
`prompt_frames` {obj_id: frame}, per-frame `visible` / `score` / area / bbox;
the prompt file itself (including `source` fields) is recorded by path in
the run's `request.json` (hash it in the caller, as real2sim's bridge does).
Semantics: `visible=false` only means "no SAM 3 mask on this frame" (it does
not distinguish occluded / out of view / lost); for point/box-prompted ids
SAM 3 fixes `score` to 1.0, which is NOT a calibrated confidence. An empty
result (no mask on any frame) is a normal output; whether the scene is
incomplete is decided by the consumer.

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
