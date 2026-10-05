# Verdi delivery status for real2sim / SceneAgent (three separate columns)

As of main `26114eb` (2026-10-06). Columns are independent: a pass in one
never implies the next.

* **Entry**: the node/task exists in main (commit where the current behaviour landed).
* **Executed (fixture / real call)**: `verdi test` fixture regression (STATUS.toml;
  `tested_at`, HEAD at test time = parent of the commit that landed the tested
  tree) and/or a real call by the consumer (mode, evidence path). This column says the
  call RUNS and meets its regression checks; it is not accuracy.
* **Independent geometric validation**: against a reference that is independent of the
  model and of its inputs. "not provided" = no reference given for real2sim scenes.
  Non-geometric nodes: "n/a" plus a separate quality-assessment status.
  Fixture metrics against synthetic analytic truth / public datasets are listed as
  *fixture-only* evidence and are NOT scene accuracy. Visible-depth agreement, mask IoU,
  reprojection error, fit PSNR, priors, generated assets and conditioning inputs are
  self-consistency, never ground truth.

Hosts: 063 (reference) and apex (deployment, 7 nodes set up). Real calls: real2sim
`outputs/verdi/train_pilot_v{1,2}` (3 train windows = known debug regression set; the
calibration history touched heldout, so NOT a clean / independent validation).

## P0

| node.task | Entry | Executed (fixture / real call) | Independent geometric validation |
|---|---|---|---|
| stereo_rectify.rectify (supplied / k_rect / opencv projection, multi-frame epipolar diagnostics) | 7e6a9cf, 01e587c | fixture 7/7 pass (063 2026-10-05 23:30; apex): synthetic fisheye remap vs direct render PSNR 37 dB, |dy| 0.05 px, wrong-R / inconsistent-R1R2 / degenerate-projection refused. Real: 3 windows, mode `supplied`, consistency checks 0.0 deg / B 0.060517 m; pooled median |dy| 0.89 / 1.18 / 0.78 px (ep003_grasp / ep003_place / ep037_grasp; place exceeds the 1 px diagnostic threshold) | not provided (no rig ground truth); fixture-only: analytic synthetic rig |
| foundation_stereo.estimate_depth[_seq] (valid, info, doffs, lr_check) | 6584419 | fixture 5/5 pass (063 21:33; apex): synthetic analytic depth abs-rel 0.27 % (pair), chain SBS->rectify->FS 0.42-0.52 %; distorted camera refused. Real: 3 windows `estimate_depth_seq`, lr_check; image valid 33-84 % (frame 9 place: 53.7 %, 40 % LR-inconsistent) | not provided; fixture-only: synthetic analytic depth |
| sam3.segment_text / segment_prompts / track_text / track_prompts (per-frame visible / score) | 6584419 | fixture 5/5 pass (063 20:03; apex): track_prompts mask IoU 0.978, occlusion frames visible=false. Real: track_text 3 windows (table / basket / hand 12/12; pepper missed by track_text init in one window -> recovery doc, track_prompts) | n/a (segmentation). Quality: visual checks only; `visible=false` = no mask (not occlusion/loss); prompted score = 1.0 is not a confidence |
| depth_anything_3.reconstruct / estimate_depth (visual vs conditioned, known-K metric focal, DA3METRIC) | cf9215a, bda8ba2 | fixture 15 cases pass (063 21:37; apex): TUM mocap visual ATE 13 mm Sim3 / 52 mm SE3; conditioned x2 -> input_pose_scale; refusals. Real: 3 windows, visual mode with known K; consumer reports table depth/stereo median ratio 0.86-1.13 and SE3 disagreement with FK 3.5-5.4 mm | not provided. Ratios vs stereo and FK disagreement are cross-consistency (FK = reference input, not truth); fixture-only: TUM mocap / Kinect depth |
| foundationpose.estimate / track (diagnostics, valid) | 6584419, 5b7019c | fixture 5/5 pass (063 21:49; apex): wrong-size mesh valid=false; relative depth / relative mesh refused. Real: pepper (ep037) **executed ok, valid=true** (median visible residual 2.79 mm, IoU 0.886); basket (ep003_place) **executed ok, valid=false** (depth_inlier 0.222 < 0.3, median 0.02083 m, IoU 0.711) | not provided. `valid` = visible-depth / mask agreement only; it is not pose accuracy |
| scale contract (depth / depth_seq / pointcloud / mesh sidecars, metric inputs refuse non-metric) | bda8ba2, 5b7019c | pytest + fixtures pass (vggt / mast3r relative sidecars, FP refusals) | n/a (contract). Quality: machine-checkable; data without sidecar = `unspecified` (caller responsibility) |
| sam3d_objects.reconstruct (meshes_metric, S_cam_glb, bake_glb_to_metric) | 5b7019c, 202d506 | fixture 2/2 pass (063 21:52; apex): metric mode size 0.192 m (YCB mustard ~0.191); bake reproduces S_cam_glb to 1e-8 m. Real: pepper and basket executed ok | not provided. Whole mesh is model-generated; observation support unknown (no per-face labels); basket mesh larger than the stereo rim rectangle (R2S-DIAG report) |

### Real-call outcome, kept separate from execution

| window | executed | geometry flags (not accuracy) |
|---|---|---|
| ep003_grasp | all stages ok | (consumer pilot 1 report) |
| ep003_place | all stages ok | basket FP valid=false; epipolar 1.18 px > 1 px diagnostic; mesh not accepted |
| ep037_grasp | all stages ok | pepper FP valid=true (observation agreement only) |

Empty masks / undetected phrases are normal outputs (`ids_per_phrase` empty), not
execution failures; scene completeness is decided by real2sim.

## P1 / SceneAgent-relevant existing nodes

| node.task | Entry | Executed | Independent geometric validation |
|---|---|---|---|
| mapanything.reconstruct | cf9215a | fixture 6 cases pass (063 21:35; apex): TUM mocap ATE 36 mm Sim3; real: 3 windows with stereo depth conditioning (table depth / stereo ratio 0.97-1.03 is NOT independent: stereo was an input) | not provided |
| vggt.reconstruct / mast3r_slam.slam | 26114eb tree (scale sidecars bda8ba2) | fixture pass (063 21:33) | not provided; outputs relative scale |
| cotracker.track_* | f121ef4 tree | fixture pass (2026-10-04) | n/a; quality: fixture EPE only |
| splatfacto.train (splatfacto-big) | f121ef4 tree | fixture pass (063 21:36); distortion currently only warned (SC-03 will refuse); optimized cameras not exported | not provided; PSNR = fit only |
| qwen2_5_vl.structured | f121ef4 tree | fixture pass (2026-10-04) | n/a; quality: schema-valid output only, priors are not measurements |
| partfield.segment ([F,448] features + face labels) / p3sam.segment | f121ef4 tree | fixture pass (063 21:49) | n/a; quality: part segmentation, not articulation. Licence: optional only (non-commercial / regional terms) |
| groundingdino.detect | f121ef4 tree | fixture pass (2026-10-04) | n/a |

## Planned (not delivered): SC-02 features, SC-03 minimal contracts, SC-01 SfM
See the manager estimate (2026-10-06): all three columns are "not yet" until implemented.
