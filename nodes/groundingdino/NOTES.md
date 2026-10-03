# groundingdino

- Upstream: HF `transformers` (4.57.1, locked) `AutoModelForZeroShotObjectDetection`
  port of Grounding DINO; weights `IDEA-Research/grounding-dino-{base,tiny}`
  pinned by revision (safetensors + tokenizer/config only). No original
  `GroundingDINO` repo / CUDA ms-deform-attn build needed.
- Text is normalized to lowercase "a. b. c." (the model needs '.'-separated
  phrases, lowercase); commas/semicolons/newlines also split phrases.
- Labels come from `post_process_grounded_object_detection(...)["text_labels"]`:
  the decoded tokens above text_threshold, which can be a sub-span or a merge
  of the given phrases (e.g. "truck tire"); filter by substring if needed.
- Legacy defaults: prompt "all objects." with 0.3/0.3 thresholds; here text
  is required and thresholds default to the upstream demo values 0.35/0.25.
- Boxes are clipped to the image and sorted by score.
- Test: truck.jpg "truck. tire." -> 1 truck + 3 tires (the third is the
  partly hidden far-side front tire; it is a correct detection). No bbox
  metric exists in core, so the test checks labels and count;
  `tests/expected/truck_boxes.json` is kept as visual reference.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build); uv.lock regenerated
  on the RTX 5090 host (aliyun index); `sm_120` added to gpu.arch. No code changes; the
  H20 references / checks pass unchanged on sm_120.
- Pitfalls: an old H20 uv.lock left on the RTX 5090 host makes `uv sync --frozen` fetch
  from the unreachable H20 PyPI mirror (delete it and re-lock); hf-mirror
  redirects Xet-backed files to an unreachable CDN (HF_HUB_DISABLE_XET=1,
  now set by core) and sometimes answers 429 on the tree API (retry later).
