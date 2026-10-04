# langsam

- Upstream: `lang-sam` (luca-medeiros/lang-segment-anything @ 918043e) as a
  git dependency, plus `sam-2` at the exact URL+commit its requirements.txt
  pins (facebookresearch/segment-anything-2 @ c2ec8e1; uv refuses two
  different URLs for one package). torch overridden to 2.7.1 (PyPI cu126),
  transformers pinned 4.57.1. SAM2_BUILD_CUDA=0.
- Weights are pinned and passed explicitly (`sam_ckpt_path`,
  `gdino_model_ckpt_path`/`gdino_processor_ckpt_path`), so upstream never
  downloads by itself (by default it pulls SAM ckpts from
  dl.fbaipublicfiles.com and GDINO from the HF hub).
- Importing `lang_sam` globally enters `torch.autocast(bfloat16)` and enables
  TF32 (upstream behaviour) -> scores are bf16-rounded.
- Legacy mapping: langsam_predict -> `segment` (one image) and
  `segment_frames` (image dir / video, per-frame independent, no tracking);
  langsam_for_aigc -> `crop_roi` (RGBA 1024 crop, picks the top-scoring
  detection, margin 20; legacy dilation was 0 px so dropped); langsam_sam
  with use_auto_mask -> `auto_masks` (the non-auto branch raised
  NotImplementedError in legacy and is not ported; use sam2 for prompts).
- Legacy wrote one binary png per detection + vis jpg + metadata json; now
  one instance-id mask (id i = boxes[i-1], boxes sorted by score) + bbox_set.
- `crop_roi` output is an RGBA png typed `image` (the image type documents
  RGB; there is no RGBA/alpha-image type).
- Text normalization: phrases split on . , ; newline, lowercased, joined
  with ". " (Grounding DINO convention).
- First run after setup takes ~100 s (cold weights on the shared fs);
  later runs ~10 s.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build); uv.lock regenerated
  on the RTX 5090 host (git deps via the local GitHub mirror); `sm_120` added to gpu.arch.
  No code changes; H20 references pass (mask_iou >= 0.9995).
- hf-mirror answered 429 on the tree API for hours; the weights were
  identical to already-downloaded ones, so they were copied in the bucket
  with `.verdi_complete` markers: gdino from groundingdino/base (same repo
  + revision), sam2.1 checkpoints from sam2/* after checking that each
  repo's main commit (X-Repo-Commit) equals the revision pinned here.
