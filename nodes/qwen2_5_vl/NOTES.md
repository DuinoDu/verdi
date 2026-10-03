# qwen2_5_vl notes

## How upstream is used
- No upstream checkout: HF `transformers==4.51.3`
  (`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor`) and
  `qwen-vl-utils==0.0.11` (`process_vision_info`: image resizing to the
  28-px grid within `min_pixels`/`max_pixels`, frame lists as video).
- Weights: 7b and 3b are pre-fetched by setup (pinned HF revisions);
  32b is `lazy` (65 GB) and is downloaded on first `model=32b` use with the
  revision pinned in `entry.py` (`MODELS`).
- Decoding is greedy (`do_sample=False`) so answers are reproducible.
- Attention: PyTorch SDPA (flash-attn not installed; avoids a CUDA build).

## Prompt input (design choice)
Both are supported, exactly one must be given: `-p prompt="..."` for short
prompts, `-i prompt_file=x.txt` (type `text`) for long / multi-line ones.

## structured task
- Schema is checked with `jsonschema` (`validator_for`, default draft
  2020-12, `format` checked with the draft's FORMAT_CHECKER).
- The schema is put in the system prompt; the reply is `json.loads`-ed
  (one surrounding ```json fence is stripped, nothing else is repaired).
- On parse/validation failure the model gets ONE corrective turn with the
  error text; if that also fails -> NodeError (no fabricated output).
  Both replies and errors are in `result.metadata.attempts`.

## frames
`frames` (image_seq / video file) are uniformly subsampled to `max_frames`
(default 32). `frames_mode=video` passes them as one video (per-frame
budget capped at 360*420 px), `frames_mode=images` as separate images.

## Pitfalls met
- torch must be >= 2.6 (PyPI torch 2.7.1 = cu126). cu121 builds SIGFPE on
  H20 inside `generate` (cuBLAS 12.1 m=1 GEMV).
- Loading the 7B safetensors via mmap from the vepfs network filesystem
  took ~5 min per 4 GB shard (scattered page faults; first test run timed
  out). `entry.py` now reads the shards sequentially once (~35 s for
  15 GB) before `from_pretrained`, then mmap hits the page cache.
- Setup download from hf-mirror was slow (~2 MB/s, read timeouts); the 7B
  snapshot was seeded from an existing local HF cache of the same revision
  (snapshot_download then verifies by hash).

## the RTX 5090 host (RTX 5090 32 GB, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128), transformers 4.51.3
  unchanged; all tests pass with identical answers to the H20.
- model=32b (~70 GB bf16, single-GPU device_map) does not fit a 32 GB GPU:
  documented in the description and a known_issue; use 7b/3b there.
- The first 3B snapshot download from hf-mirror stalled on its last file
  for > 1 h (process alive, no progress); killing and re-running setup
  resumed and finished. hf-mirror also returns 429 when many setups run
  at once; just retry.
