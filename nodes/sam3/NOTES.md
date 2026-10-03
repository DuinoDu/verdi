# sam3

## How upstream is used

- `facebookresearch/sam3` is a uv git dependency pinned to `46957e47`
  (the commit SimFoundry pins in `install_simfoundry.sh`). No patches;
  SimFoundry ships none for SAM 3.
- `segment_text` = SimFoundry `SAM3.predict_segmentation`:
  `build_sam3_image_model` + `Sam3Processor(confidence_threshold)`, one
  `set_text_prompt` per phrase on shared image features, bf16 autocast.
- `segment_prompts` = SimFoundry `predict_segmentation_with_point`
  (sam_v3_gmask.py): image model with `enable_inst_interactivity=True`,
  `Sam3Image.predict_inst(point_coords, point_labels, box)` per obj_id.
- `track_text` = SimFoundry `predict_video_segmentation`: video predictor
  `start_session` / `add_prompt(text)` / `propagate_in_video` (both
  directions). The node uses the single-process `Sam3VideoPredictor`
  instead of `build_sam3_video_predictor` (= `Sam3VideoPredictorMultiGPU`,
  which sets up torch.distributed even on one GPU).
- SAM 3.1 (`facebook/sam3.1`, multiplex video model) is not wired: SimFoundry
  uses SAM 3.

## Weights

- Official `facebook/sam3` on HF is gated; the H20 HF account (token
  present) gets `403 not in the authorized list`. The checkpoint is fetched
  from the ModelScope mirror `facebook/sam3` and pinned by sha256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, which
  is the LFS sha256 the official HF repo reports for `sam3.pt` (read via
  `model_info(files_metadata=True)`, which works without access), so the
  file is byte-identical. Several HF re-uploads (jetjodh/sam3, ...) carry the
  same hash too.
- Only `sam3.pt` (3.45 GB) is needed: the BPE vocab ships in the package and
  `checkpoint_path` + `load_from_HF=False` avoid any hub download.

## Conventions

- mask ids: `segment_text` id i = boxes[i-1] (score-sorted, smaller mask
  wins on overlap); `segment_prompts` id = obj_id; `track_text` ids = SAM 3
  object id + 1 (+ offset per phrase), consistent over frames.
- box score = detection prob * presence prob (upstream `scores`).

## Pitfalls

- Inference must run under bf16 autocast (fused ViT MLP emits bf16).
- `model_builder` imports `pkg_resources` (setuptools must stay installed).
- The video frame loader sorts by integer file stem: frames are symlinked as
  `%06d<ext>` into a temp dir.
- Cold vepfs: loading the 3.4 GB checkpoint dominates runtime; it is
  prefetched (`ctx.weight(..., prefetch=True)`).

## Fixture

`tests/fixture/truck.jpg` and `bedroom/` (6 frames) copied from the sam2
node fixture. Expected outputs were checked visually (overlay) before being
saved under `tests/expected/`.

## the RTX 5090 host (RTX 5090, sm_120)

- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build), python 3.11; lock
  regenerated on the RTX 5090 host (aliyun index). `sm_120` added to gpu.arch.
- The sam3 repo was missing from the the RTX 5090 host GitHub mirror (dangling symlink)
  until `bin_mirrorthe RTX 5090 host.sh facebookresearch/sam3` succeeded (GitHub TLS
  errors on the laptop needed retries).
- sam3.pt was prefetched into the bucket from the ModelScope mirror by url;
  setup only verifies its sha256 (official HF LFS oid).
- First tested on the RTX 5090 host (this node was never tested on the H20). Overlays of
  all three tasks were checked visually (text: truck + 4 tires incl. the
  partly hidden far-side tires; prompts: point -> rear side window pane,
  box -> rear tire; track: both children, consistent ids over 6 frames);
  the the RTX 5090 host outputs are the references in `tests/expected/`
  (truck_text_*, truck_prompts_mask.png, bedroom_child/).
