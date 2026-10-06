# Verdi delivery status / requirement ledger (scope VD-SCOPE-20261005)

As of main `19ff1cb` + the scope commit (2026-10-06). This file replaces the earlier
SC / R2S-DIAG layout; no code or evidence was deleted or moved.

## 0. Scope (VD-SCOPE-20261005, user ruling; overrides out-of-scope parts of SC-20261005 / R2S-DIAG)

Verdi = **generic model inference**:
* model / weights / dependency pinning and licence + source registration;
* generic infer task entries;
* model-native pre/post-processing (resize / crop / pad and their pixel maps, saving
  masks / features / meshes / poses);
* stating real units / coordinates / scale source and how an output was generated;
* generic request / result schema, input content hash, log / error / empty-result semantics;
* generic run capability (device selection, isolation, remote run, resume) and infer fixtures.

**Not Verdi (handed over to the consumer, section C):**
* traditional geometry pipelines (COLMAP / SfM, camera registration / undistortion workflows);
* scene training and orchestration (nerfstudio / splatfacto, RGB -> SfM -> 3DGS, Gaussian
  semantics / physics / appearance);
* business post-processing (feature clustering / part propagation / fusion / joint hypotheses);
* calibration refit;
* basket attribution / geometry repair / containers / simulation;
* SceneAgent physical-prior schema and render checks;
* real-scale / cavity / task-tolerance acceptance.

VLM nodes take the caller's prompt / schema. Property field systems, region naming rules,
prompts and business quality belong to the consumer. Model outputs (pose / residuals / mask /
`valid`) are still delivered as they come; Verdi makes **no business usability judgement**.

GPU release acceptance covers generic infer only. Neither splatfacto training nor basket
geometry is a Verdi completion condition. The proposed training regression window (W2 for
splatfacto) is **cancelled**. Infer regressions still follow the existing resource
arrangement; this ruling does not open GPU time.

Column meaning (sections A / B):
* **Entry** = the commit where the behaviour landed.
* **Executed** = `verdi test` fixture regression (STATUS.toml) and/or a real consumer call.
  It shows the call runs; it is not accuracy.
* **Independent validation** = against a reference independent of the model and its inputs.
  Fixture metrics on public / synthetic data are fixture-only. Mask IoU, visible-depth
  agreement, reprojection error, fit PSNR, priors and generated assets are
  self-consistency, never ground truth.

Hosts:
* 063: reference host.
* apex: 7 nodes set up.
* 018: bootstrapped 2026-10-06; general runtime only, no GPU job run there yet.

## A. Verdi infer — completed (entry + executed; GPU fixture run on 063, apex where noted)

| node.task | Entry | Executed (fixture / real call) | Independent validation |
|---|---|---|---|
| foundation_stereo.estimate_depth[_seq] (valid, info, doffs, lr_check). Input contract: the CALLER supplies a rectified stereo pair with its K and baseline (distorted camera refused); model-native resize / crop / pad, pixel map and disparity -> z-depth (metres) conversion stay in Verdi | 6584419 | fixture 5/5 (063; apex): synthetic analytic depth abs-rel 0.27 %, chain 0.42-0.52 %; distortion refused. Real: 3 windows `estimate_depth_seq` | not provided; fixture-only: synthetic analytic depth |
| sam3.segment_text / segment_prompts / track_text / track_prompts | 6584419 | fixture 5/5 (063; apex): track_prompts IoU 0.978, occluded frames visible=false. Real: 3 windows | n/a. `visible=false` = no mask; prompted score 1.0 is not a confidence. Prompt choice = consumer |
| depth_anything_3.reconstruct / estimate_depth | cf9215a, bda8ba2 | fixture 15 cases (063; apex): TUM visual ATE 13 mm Sim3; conditioned -> input_pose_scale; refusals. Real: 3 windows | not provided; fixture-only: TUM mocap / Kinect |
| foundationpose.estimate / track (diagnostics, `valid`) | 6584419, 5b7019c | fixture 5/5 (063; apex). Real: pepper ok `valid=true`; basket ok `valid=false` (delivered as computed) | not provided; `valid` = visible-depth / mask agreement, not pose accuracy, not a business verdict |
| sam3d_objects.reconstruct (meshes_metric, S_cam_glb, bake) | 5b7019c, 202d506 | fixture 2/2 (063; apex): metric size 0.192 m (YCB ~0.191). Real: pepper / basket ok. The 3055644 change (evidence text in poses.json) is covered by section B | not provided; whole mesh is model-generated |
| mapanything.reconstruct | cf9215a | fixture 6 cases (063; apex): TUM ATE 36 mm Sim3 | not provided |
| vggt.reconstruct / mast3r_slam.slam | 26114eb tree | fixture pass (063) | not provided; relative scale |
| cotracker.track_* | f121ef4 tree | fixture pass | n/a |
| groundingdino.detect | f121ef4 tree | fixture pass | n/a |
| qwen2_5_vl.structured (caller prompt / schema) | f121ef4 tree | fixture pass | n/a; schema-valid output only; field system / prompts / quality = consumer |
| partfield.segment / p3sam.segment (model outputs: features + labels) | f121ef4 tree | fixture pass (063) | n/a; licence: optional only |
| dinov2_features.extract (CLS / registers / patch + pixel map, region pooling) | 48cabe2; TF32 fix in the SC-02 GPU commit | CPU fixture 5/5 (063, 018). **GPU fixture 5/5 on 063 cuda:3 (RTX 5090)**. CPU/GPU numeric consistency: first pass **FAIL** (patch min cosine 0.9937, cuDNN TF32), rerun with TF32 disabled **PASS** (all arrays min cosine >= 0.99999997). Evidence: `pipelines/real2sim/evidence/sc02_gpu_20261006.md` | n/a (features). Numeric regression only; task quality not evaluated |
| clip_features.encode_text / encode_regions (official B/32, L2, cosine only) | 48cabe2 | CPU fixture 6/6 (063, 018). **GPU fixture 6/6 on 063 cuda:3**. CPU/GPU consistency **PASS** (roi min cosine 0.999998, argmax identical) | n/a. Numeric regression only; quality not evaluated; licence MIT, model card registered separately |

Generic contracts and run capability (completed):

| item | Entry | Executed | Note |
|---|---|---|---|
| scale contract (sidecars; metric ports refuse non-metric) | bda8ba2, 5b7019c | pytest + fixtures | data without a sidecar = `unspecified` |
| input content hash in provenance (`hash_algorithm`, file / dir manifest, sidecars, `hash_verified`) | 3055644 | pytest + real CPU call | differs from the real2sim bridge's name+content digest |
| result / error semantics (NodeError -> status error, rc 1; empty mask = normal output) | existing | all refusal fixtures | — |
| remote runner `scripts/verdi_remote.py` (run-id isolation, ACTIVE / DONE, verified download, `--resume`, `--list`, exact-id `--delete`, `--device auto`) | cb70413 | CPU state simulation tests (3) | `--host 063/017/018`: scope commit |

## B. Model infer calls — pending test (no GPU started for these)

| node.task | Entry | Done so far | Pending |
|---|---|---|---|
| sam3d_objects.reconstruct after 3055644 (evidence field only) | 3055644 | **covered without GPU re-run**: the diff adds one dict (`evidence`) to poses.json AFTER inference, built only from `metric` and `int(k)`, which the same `poses.append` literal already uses (static check: both assigned in `reconstruct`). No model / pre- / post-processing / mesh / pose change since the real GPU calls at 01e587c (pepper, basket) and the GPU fixture 2/2 (063 2026-10-05 21:52). Generic runner changes of 3055644 (input hashes) are device-independent and exercised by CPU runs | residual risk: none for inference; poses.json gains an additive key (pose_set validator checks T_cam_obj only). Re-run only at the next scheduled infer regression, not as a separate GPU job |
| 018 / 017 as extra infer hosts | scope commit | 018: toolchain + repo from the bucket bundle, pytest 29/29, `verdi doctor` OK; stereo_rectify / sfm / dinov2_features / clip_features fixtures pass on CPU | GPU fixture regression on 018 (only in a declared slot); 017 not bootstrapped (all 8 GPUs partly used by others) |

## C. 已移交 real2sim (handed over; not Verdi development debt, not model unavailability, not a Verdi acceptance gate)

The code and evidence stay in the repo as they are and may be reused. `nodes/sfm` and
`nodes/splatfacto` are kept as **legacy** nodes for real2sim to reuse. Business maintenance and
acceptance of them belong to real2sim; Verdi does not commit to further development on them.

### C.1 Handover record (SceneAgent requester ffa6354a; as reported by the requester / Lead, not re-verified by Verdi)

real2sim commit ids below are **real2sim repository** commits, registered separately from Verdi main commits.

| item | value |
|---|---|
| status | business first batch **reviewed and merged by the Lead** into real2sim main; main synthetic CPU regression 46/46 pass (requester's report). Supersedes the earlier edfba3c record (pre-merge status, outdated) |
| business commit | real2sim `7afb380213e461869363fcb3e2a57678a9f998ad`; Lead read-only re-review pass message `3f18fc78-88b0-4604-af6b-a76fd002e109`; merged into real2sim main `056c4df` |
| README risk addendum | real2sim `dba54e7f033365120367cb2bde26fbb12908c73b`; Lead review / merge message `534cd0c0-60ac-4233-b97e-42a77a95bebb`; real2sim main currently `dba54e7` (requester's report) |
| responsibility entry | apex `/mnt/data1/min.du/ws/real2sim/pipeline/sceneagent/README.md` (moved from the former worktree) |
| business CPU evidence (relative to the real2sim root) | `pipeline/sceneagent/.checks/main_merge_7afb380/summary.json`, `cpu_tests.log`, `source_policy_negatives/result.json`, `synthetic_evidence/negative_geometry_evidence.json` |
| scope on their side | DINO patch clustering / CLIP label candidates, VLM business prompt / schema / physical-prior checks, same-camera partitioned depth residuals, SfM -> 3DGS orchestration / explicit scale anchor |
| NOT recorded by Verdi | these are the requester's / Lead's business code and synthetic CPU evidence: not a Verdi model-call pass, not real geometry / physical-property acceptance, not a complete SceneAgent reproduction. Verdi did not review, merge or re-check their geometry; legacy items (stereo_rectify / SfM / splatfacto) stay with real2sim; their original outputs and contamination / not-accepted marks are untouched |

| item | where (commit / path) | how to call (existing) | state at handover |
|---|---|---|---|
| SC-01 SfM, legacy (pycolmap 4.2.1 CPU; one shared undistorted K; relative scale; registration flags) | `nodes/sfm` @ 19ff1cb; NOTES.md | `verdi run sfm --task reconstruct -i frames=DIR [-i camera=cam.json] [-i reference_poses=traj.json] --out OUT --device cpu` | fixture 6/6 CPU (TUM ATE 0.0146 m on the public fixture only) |
| splatfacto training, legacy (+ SC-03 training extensions: distortion refusal, optimizer default off, gaussian scale / world sidecar, frame mapping) | `nodes/splatfacto` @ f121ef4 + 3055644 | `verdi run splatfacto --task train -i frames=OUT/frames -i camera=OUT/camera.json -i trajectory=OUT/trajectory.json [-i points=OUT/points.ply] -p max_num_iterations=N --out OUT2 --device cuda:K` | GPU fixture passed before 3055644; not re-run after it (window cancelled) |
| RGB -> SfM -> 3DGS chain | sfm outputs -> splatfacto inputs | the two calls above | CPU contract check only: splatfacto `_load_inputs` accepts sfm outputs, scale -> relative. Training on sfm outputs never run |
| optimized-camera export; per-frame K; render_depth / business render checks | `pipelines/real2sim/render_depth_contract.md` (plan only) | — | not implemented |
| R2S-DIAG basket analysis (attribution, residual / support statistics, size comparison) | apex `~/verdi_reports/r2s_diag_20261005/`: `report.md` (v2), `report_v1.md`, `evidence.json`, `support_all_frames.json`, `render_depth_{sam3d_initial,foundationpose}.npy`, `residual_*.jpg`, `basket_ev.py`, `basket_ev2.py`, `basket_frames.py` | `python3 basket_ev.py`, then `basket_ev2.py` / `basket_frames.py` (paths hard-coded to real2sim `outputs/verdi/train_pilot_v2/runs/ep003_place` and the report dir; read-only, CPU) | v2 report delivered; follow-up analysis = real2sim |
| stereo_rectify, legacy (geometric rectification + calibration / epipolar diagnostics, incl. multi-frame dy vs disparity, supplied / k_rect projections) | `nodes/stereo_rectify` @ 7e6a9cf, 01e587c (fixture 7/7 on 063 / apex / 018, CPU node); `pipelines/real2sim/README.md` §rectify; `example_stereo.sh` | `verdi run stereo_rectify --task rectify -i stereo=SBS -i calib=calib.json [-p ...] --out OUT --device cpu` (unchanged) | 已移交 real2sim (manager ruling 2026-10-06): rectification parameters / mapping choice, undistortion, extrinsic fitting, epipolar diagnostics and workflow = real2sim; tool kept unchanged for reuse |
| SAM3 prompt recovery notes (ep003) | apex `~/verdi_reports/r2s_diag_20261005/sam3_prompt_recovery_ep003.md`; `pipelines/real2sim/README.md` §7 | `verdi run sam3 --task track_prompts ...` | usage notes of the generic node; prompt policy = consumer |
| business quality, renderer / render checks, codebook, physics / physical priors, joints / articulation, independent ground truth | real2sim (see C.1) | — | 已移交 real2sim; never a Verdi item |
| runner evidence for the lost-run incident | apex `.../runner_evidence.md`; `scripts/verdi_remote_ACCEPTANCE.md` | — | runner itself stays in A |

## Requirement ledger (status after VD-SCOPE-20261005)

| requirement | status |
|---|---|
| P0 SAM3 / FS / DA3 / FP / cross-machine calls | A (done) |
| P1 MapAnything / VGGT / MASt3R scale / CoTracker / SAM3D notes | A (done) |
| fisheye supplied rectification (stereo_rectify, legacy) | 已移交 real2sim (C); FS rectified-input contract stays in A |
| SC-02 DINOv2 / CLIP features | A (GPU fixture + CPU/GPU consistency on 063, 2026-10-06; dinov2 first-pass FAIL kept, fixed by disabling TF32) |
| SC-03 input content hash; infer output contracts (scale / coords / generation statement) | A (done) |
| SC-03 splatfacto training extensions, optimized-camera export, render_depth | 已移交 real2sim (C) |
| SC-01 SfM / SfM -> 3DGS orchestration | 已移交 real2sim (C; legacy node kept) |
| R2S-DIAG v1 / v2 reports | delivered; follow-up 已移交 real2sim (geometry_audit_v1 is theirs) |
| SceneAgent business (clustering, labels, VLM prompts / schema / priors, partitioned residuals, codebook, physics, joints) | 已移交 real2sim, C.1 (real2sim `7afb380` merged into real2sim main `056c4df`, README `dba54e7`; Lead-reviewed per requester) |
| runner fix / resume / isolation; extra hosts 017 / 018 | A; 018 GPU regression in B |
