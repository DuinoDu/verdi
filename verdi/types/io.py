"""Readers/writers for the unified types (numpy + Pillow only)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from verdi.types.registry import IMAGE_EXT, SCALE_STATES, scale_sidecar

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# ---------------------------------------------------------------- images
def read_image(path: Path) -> np.ndarray:
    """Return HxWx3 RGB uint8."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def write_image(path: Path, rgb: np.ndarray) -> Path:
    from PIL import Image

    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
    return path


def list_frames(path: Path, exts: Sequence[str] = IMAGE_EXT) -> List[Path]:
    return sorted(p for p in Path(path).iterdir() if p.suffix.lower() in exts)


def frame_name(index: int, ext: str = ".png") -> str:
    return f"{index:06d}{ext}"


def write_image_seq(out_dir: Path, frames: Iterable[np.ndarray],
                    fps: Optional[float] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rgb in enumerate(frames):
        write_image(out_dir / frame_name(i), rgb)
    if fps:
        (out_dir / "meta.json").write_text(json.dumps({"fps": fps}))
    return out_dir


def ffmpeg_exe() -> str:
    """System ffmpeg, else the static binary shipped by imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found: apt-get install -y ffmpeg or "
                           "pip install imageio-ffmpeg") from exc


def video_to_image_seq(video: Path, out_dir: Path) -> Path:
    """Split a video into ``%06d.png`` frames with ffmpeg (+ meta.json)."""
    import re

    exe = ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(video),
         "-start_number", "0", str(out_dir / "%06d.png")],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-1000:]}")
    fps = None
    m = re.search(r"([\d.]+) fps", proc.stderr)
    if m:
        fps = float(m.group(1))
    (out_dir / "meta.json").write_text(
        json.dumps({"fps": fps, "source": str(video)}))
    return out_dir


# ----------------------------------------------------------- depth/masks
def write_depth(path: Path, depth_m: np.ndarray) -> Path:
    arr = np.asarray(depth_m, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0
    np.save(path, arr)
    return path


def write_mask(path: Path, ids: np.ndarray,
               labels: Optional[Dict[int, Dict[str, Any]]] = None) -> Path:
    """Write an instance-id mask png (uint8 if <256 ids else uint16)."""
    from PIL import Image

    ids = np.asarray(ids)
    dtype = np.uint8 if ids.max(initial=0) < 256 else np.uint16
    Image.fromarray(ids.astype(dtype)).save(path)
    if labels:
        Path(str(path)[:-4] + ".json").write_text(
            json.dumps({str(k): v for k, v in labels.items()}, indent=2))
    return path


def masks_to_ids(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Stack binary masks [N,H,W] into one id map (later masks win)."""
    masks = [np.asarray(m).astype(bool) for m in masks]
    if not masks:
        raise ValueError("no masks")
    ids = np.zeros(masks[0].shape, dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        ids[m] = i
    return ids


# ------------------------------------------------------------------ json
def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_bbox_set(path: Path, boxes: List[Dict[str, Any]]) -> Path:
    return write_json(path, {"boxes": boxes})


def write_pose_set(path: Path, poses: List[Dict[str, Any]]) -> Path:
    for p in poses:
        p["T_cam_obj"] = np.asarray(p["T_cam_obj"], dtype=float).tolist()
    return write_json(path, {"poses": poses})


def read_camera(path: Path) -> Dict[str, Any]:
    data = read_json(path)
    data["K"] = np.asarray(data["K"], dtype=float)
    data["dist"] = np.asarray(data.get("dist", [0, 0, 0, 0, 0]), dtype=float)
    return data


def write_camera(path: Path, K: np.ndarray, width: int, height: int,
                 dist: Optional[Sequence[float]] = None) -> Path:
    return write_json(path, {
        "K": np.asarray(K, dtype=float).tolist(), "width": int(width),
        "height": int(height), "dist": list(dist or [0, 0, 0, 0, 0])})


def write_scale(path: Path, scale_status: str, units: str, source: str,
                note: str = "") -> Path:
    """Declare the scale of a depth / depth_seq / pointcloud output.

    metric | metric_from_input_poses are accepted by inputs that need metres;
    relative | input_pose_scale | input_depth_scale are refused there."""
    if scale_status not in SCALE_STATES:
        raise ValueError(f"scale_status {scale_status!r} not in {SCALE_STATES}")
    return write_json(scale_sidecar(Path(path)), {
        "scale_status": scale_status, "units": units, "source": source,
        "note": note})
