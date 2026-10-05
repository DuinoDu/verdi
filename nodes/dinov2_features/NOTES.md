# dinov2_features — registration notes

## Source (pinned, reused from the existing sam3d_objects cache; no new download)
| item | value |
|---|---|
| code | facebookresearch/dinov2 archive `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (hubconf.py sha256 `c1f5090e…6f64`), `torch.hub.load(local, "dinov2_vitl14_reg", pretrained=False)` |
| weights | `dinov2_vitl14_reg4_pretrain.pth`, 1,217,607,321 bytes, sha256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`, `load_state_dict(strict=True)` |
| licence | code and weights Apache-2.0 |

## Contract
* CLS [F, 1024], registers [F, 4, 1024] (separate), patch [F, h, w, 1024]: final-LayerNorm
  tokens of `forward_features`; not L2-normalised unless `l2_normalize=true`.
* Resize only (PIL bicubic), no crop / no padding: both sides multiples of 14, long side ~
  `max_side`; u_net = sx u_in, v_net = sy v_in; `patch_centers.npy` and the bounds formula in
  `info.pixel_map`. The whole image is valid.
* Regions: bbox and/or mask; exact area coverage per patch cell; coverage-weighted mean of
  cells >= `min_coverage`; region with no such cell -> `valid=false`, zero row, reason.
* Embedding space is DINOv2-reg4 only (not comparable with CLIP).

## Fixture evidence (CPU only, 063, 2026-10-06; regression / sanity, not quality)
5/5 cases pass with `--device cpu` (layout, l2/float16, 3 refusals). pytest: patch coverage
equals 8x8 supersampling within 0.02. Region cosine (sanity): keyboard frame 1 vs keyboard
frame 12 (partial) 0.688 = highest cross-region value; keyboard vs book 0.164, mustard vs
others <= 0.21. Task evaluation: unknown (not done). GPU call: not yet run.
