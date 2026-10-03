# moondream notes

## How upstream is used
- Model code is the HF repo's own remote code (`hf_moondream.py`,
  `moondream.py`, ...) at the pinned revision 2025-06-21
  (`9a7d4024050840e001defacec2b00727e89149e6`), downloaded as weights.
  `entry.py` imports it as a package directly from the weights dir
  (`moondream2_hf.hf_moondream.HfMoondream.from_pretrained(dir)`) instead
  of `AutoModelForCausalLM(trust_remote_code=True)`.
- Tokenizer: the remote code calls `Tokenizer.from_pretrained("moondream/starmie-v1")`
  (unpinned network fetch). It is declared as weight `starmie_tokenizer`
  (pinned revision) and `entry.py` redirects that call to the local file.
- Text tasks use `temperature=0` (upstream default 0.5) for reproducible
  answers. detect/point are deterministic (argmax) upstream.
- Coordinates: upstream returns normalised [0,1] boxes / points; converted
  to pixels (boxes clipped to the image). No confidence is available, so
  bbox_set boxes have no `score`.

## Pitfalls met
- transformers 4.52 `trust_remote_code` from a *local dir* copies only part
  of the module files into `HF_HOME/modules` (FileNotFoundError layers.py /
  rope.py) -> direct package import (above).
- At this revision `encode_image(img, settings)` reads `settings["variant"]`
  unguarded -> KeyError when any settings dict is passed with a PIL image.
  Images are encoded first with `model.encode_image(img)` and the
  EncodedImage is passed to query/caption/detect/point.
- First load from the vepfs network FS via safetensors mmap took ~6 min;
  the weights file is read sequentially once before loading (page cache).
- detect "wheel" on truck.jpg returns 3 boxes: the 3rd is the far-side
  front wheel visible under the body (correct). The test uses "truck".

## Dropped from legacy
Lance dataset batch mode (`lance_image_col`, `output_key`) and the
`unittest` dummy-caption mode (fabricated output).

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 / torchvision 0.23.0 (PyPI cu128 build); uv.lock regenerated
  on the RTX 5090 host (aliyun index); `sm_120` added to gpu.arch. No code changes; the
  H20 references / checks pass unchanged on sm_120.
- Pitfalls: an old H20 uv.lock left on the RTX 5090 host makes `uv sync --frozen` fetch
  from the unreachable H20 PyPI mirror (delete it and re-lock); hf-mirror
  redirects Xet-backed files to an unreachable CDN (HF_HUB_DISABLE_XET=1,
  now set by core) and sometimes answers 429 on the tree API (retry later).
