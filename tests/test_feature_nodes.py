"""CPU unit tests of the SC-02 feature nodes' pure geometry (no model, no weights)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(node):
    spec = importlib.util.spec_from_file_location(f"{node}_entry", ROOT / "nodes" / node / "entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dinov2_patch_coverage_matches_supersampling():
    d = _load("dinov2_features")
    h, w, gh, gw = 384, 512, 28, 37
    rng = np.random.default_rng(0)
    mask = rng.random((h, w)) > 0.3
    r = {"bbox_xyxy": [40.3, 17.7, 301.2, 250.5], "mask": mask}
    cov = d.patch_coverage(d._region_raster(r, h, w), gh, gw)
    # brute force: 8x8 sub-samples per pixel, assign every sub-sample to its patch cell
    k = 8
    u = (np.arange(w * k) + 0.5) / k
    v = (np.arange(h * k) + 0.5) / k
    inb = ((u >= 40.3) & (u < 301.2))[None, :] & ((v >= 17.7) & (v < 250.5))[:, None]
    sub = inb & np.repeat(np.repeat(mask, k, 0), k, 1)
    ci = np.minimum((v * gh / h).astype(int), gh - 1)
    cj = np.minimum((u * gw / w).astype(int), gw - 1)
    acc = np.zeros((gh, gw))
    np.add.at(acc, (ci[:, None].repeat(w * k, 1), cj[None, :].repeat(h * k, 0)), sub)
    ref = acc / ((h * k / gh) * (w * k / gw))
    assert np.abs(cov - ref).max() < 0.02


def test_dinov2_grid_and_centers():
    d = _load("dinov2_features")
    g = d._grid(720, 1280, 518)
    assert g["grid_hw"] == [21, 37] and g["network_hw"] == [294, 518]
    # patch (0, 0) centre maps back inside the first 1/37 of the image width
    assert 0 < 7 / g["sx"] < 1280 / 37


def test_clip_roi_pad_square_keeps_everything_center_crop_records_loss():
    c = _load("clip_features")
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 255
    r = {"region_id": "a", "bbox_xyxy": [10.5, 20.0, 170.0, 60.2], "mask": None}
    a, rec = c.roi_image(img, r, "pad_square", "none")
    assert a.shape == (224, 224, 3) and rec["retained_fraction"] == 1.0
    assert rec["crop_xyxy_int"] == [10, 20, 170, 61] and rec["padded_side"] == 160
    a2, rec2 = c.roi_image(img, r, "center_crop", "none")
    assert rec2["cropped_away"] == "left/right" and abs(rec2["retained_fraction"] - 41 / 160) < 1e-6


def test_clip_texts_refuse_empty(tmp_path):
    c = _load("clip_features")
    p = tmp_path / "t.txt"
    p.write_text("a\n\nb\n")
    from verdi.sdk import NodeError
    with pytest.raises(NodeError):
        c.read_texts(p)
