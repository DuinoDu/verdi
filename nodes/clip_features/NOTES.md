# clip_features — registration notes

## Source (pinned)
| item | value |
|---|---|
| code | official openai/CLIP, commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` ([upstream]; imported from the checkout, no build) |
| weights | official `ViT-B-32.pt`, URL `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt` (= `_MODELS["ViT-B/32"]` in clip/clip.py of that commit), 353,976,522 bytes, sha256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` (verified by `verdi setup`) |
| runtime deps | ftfy, regex, packaging, tqdm (+ torch 2.8.0 / torchvision 0.23.0) |

## Licence (kept separate from the model card)
* Repository `LICENSE`: MIT License, Copyright (c) 2021 OpenAI; sha256 of the file at the pin
  `987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db`.
* The weights are published by OpenAI at the URL hard-coded in that MIT repository; there is
  no separate weight licence file.

## Model card (`model-card.md` at the pin) — intended / out-of-scope use, quoted
* "The primary intended users of these models are AI researchers."
* Out-of-scope: "**Any** deployed use case of the model - whether commercial or not - is
  currently out of scope. Non-deployed use cases such as image search in a constrained
  environment, are also not recommended unless there is thorough in-domain testing of the
  model with a specific, fixed class taxonomy."
* Surveillance / facial recognition: always out of scope. English only.
* Training data: "We do not intend for this dataset to be used as the basis for any
  commercial or deployed model".

These are the model card's statements, not a licence term and not a Verdi decision.
Task evaluation for SceneAgent: **unknown (not done)**.

## Contract
* Outputs are L2-normalised float32; `logit_scale` is never applied; `similarity` = raw
  cosine (uncalibrated, not a probability).
* Text: official tokenizer; > 77 tokens = error unless `on_overflow=truncate` (recorded).
* ROI: `pad_square` (default, whole ROI kept, mean-colour padding recorded) or `center_crop`
  (cut-away part recorded as `retained_xyxy_input` / `retained_fraction`).
* float32 on CPU and CUDA (clip.load would keep fp16 on CUDA; the node casts to float32).

## Fixture evidence (CPU only, 063, 2026-10-06; regression / sanity, not quality)
6/6 cases pass with `--device cpu`. Zero-shot argmax over 5 prompts on the fixture ROIs:
mustard -> mustard bottle (top1-top2 margin 0.095), keyboard_f1 -> keyboard (0.016),
monitor_f1 -> monitor (0.064), book_f2 -> book (0.006), keyboard_f2 (partial, top edge of
the frame) -> **monitor** (wrong), tiny 5x5 px ROI -> monitor (meaningless). The small margins
and the wrong partial-view label are why no quality claim is made. GPU call: see below.

## GPU evidence (063 cuda:3, 2026-10-06)
GPU fixture 6/6. CPU/GPU consistency passes (roi min cosine 0.999998, argmax identical). The torch default cuDNN TF32 is still on for the patch convolution; it was not changed in this round. Numeric regression only. See `pipelines/real2sim/evidence/sc02_gpu_20261006.md`.
