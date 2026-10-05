# mapanything node notes

- Upstream facebookresearch/map-anything @ 3d10cf7a (uv git dependency),
  weights facebook/map-anything-apache (default, Apache-2.0) and
  facebook/map-anything (model=nc, CC-BY-NC-4.0).
- Offline init: uniception's DINOv2 encoder calls torch.hub.load(
  "facebookresearch/dinov2", pretrained=True) (GitHub + ImageNet weights,
  then overwritten by the checkpoint). The node redirects that call to the
  pinned dinov2 archive (weights.dinov2_repo, same as oneposeviagen) with
  source=local, pretrained=False.
- Pixel mapping: preprocess_inputs resizes + centre-crops to the 518 table
  (16:9 -> 518x294). The exact input -> network affine is derived from
  upstream's own crop code (with K: K_in vs processed K; without K: the
  no-intrinsics branch replicated on a dummy K). Outputs are resampled to
  the input grid (nearest for depth) with 0 outside the crop.
- Fixtures shared with depth_anything_3 (tum_desk mocap poses, mustard
  sensor depth).

- Measured on real data (063, RTX 5090, apache checkpoint, 518 table):
  * TUM fr1 desk mocap, 26 frames: ATE 36 mm after Sim(3), 50 mm rigid;
    with K 35 mm; conditioned on the mocap poses: trajectory ATE 7 mm
    (re-estimate of the given poses, not a check).
  * TUM Kinect depth (3 frames): images + K abs-rel 0.32 (1.2-1.35x too far),
    images only 0.19; + sensor depth conditioning 0.031.
  * mustard (anisotropic K fx 320 / fy 417): K not followed (output fx ~ fy
    ~ 496), depth ~2x; with depth conditioning abs-rel 0.066.
  * ~25-50 s per case incl. model load; 26 frames 640x480.
- model=nc (CC-BY-NC) is lazy: fetched into HF_HOME on first use.
