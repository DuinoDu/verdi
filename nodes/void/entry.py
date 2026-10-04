"""VOID node: inpaint (remove masked objects from a frame sequence).

Reproduces SimFoundry auto_bg steps 2/3 (2_run_void_pass1.py,
3_run_void_pass2.py): the upstream scripts are run per chunk in this venv
with the same arguments, chunk plans and cross-fade stitching.
  Pass 1: inference/cogvideox_fun/predict_v2v.py (quadmask_cogvideox config,
          void_pass1.safetensors), chunks of <=173 frames, >=30 overlap.
  Pass 2: inference/cogvideox_fun/inference_with_pass1_warped_noise.py
          (void_pass2.safetensors), 85-frame windows, warped noise from the
          Pass 1 video (Go-with-the-Flow / RAFT), warm-up-suppressed fades.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

RAFT_FILE = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
CHUNK_SIZE = 173        # pass 1: VAE cap 197, (L-1) % 4 == 0
MIN_OVERLAP = 30        # pass 1
WINDOW = 85             # transformer temporal window (pass 2 chunk length)
WARMUP = 5              # pass 2 chunk warm-up suppression


# ------------------------------------------------------------ video io
def _write_mp4(path: Path, frames: np.ndarray, fps: int, lossless: bool):
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "0" if lossless else "10", str(path)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                          capture_output=True)
    if proc.returncode != 0:
        raise NodeError(f"ffmpeg failed: {proc.stderr.decode()[-500:]}")


def _read_mp4(path: Path) -> np.ndarray:
    import mediapy

    return np.asarray(mediapy.read_video(str(path)))[..., :3]


def _resize(frames: np.ndarray, h: int, w: int, nearest=False) -> np.ndarray:
    import cv2

    if frames.shape[1:3] == (h, w):
        return frames
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return np.stack([cv2.resize(f, (w, h), interpolation=interp)
                     for f in frames])


# ------------------------------------------------- SimFoundry chunking
def _plan_pass1(n: int) -> list:
    if n <= CHUNK_SIZE:
        return [0]
    k = max(2, (n + CHUNK_SIZE - 1) // CHUNK_SIZE)
    while True:
        starts = sorted(set(np.linspace(0, n - CHUNK_SIZE, k).round()
                            .astype(int).tolist()))
        if len(starts) < k:
            k += 1
            continue
        ov = [(a + CHUNK_SIZE) - b for a, b in zip(starts[:-1], starts[1:])]
        if min(ov) >= MIN_OVERLAP or k >= n // 2 + 1:
            return starts
        k += 1


def _plan_pass2(n: int) -> list:
    if n <= WINDOW:
        return [0]
    k = max(2, (n + WINDOW - 1) // WINDOW)
    return sorted(set(np.linspace(0, n - WINDOW, k).round().astype(int)
                      .tolist()))


def _stitch(outputs, starts, lengths, total, warmup):
    """Linear cross-fade (warmup=0: pass 1; warmup=5: pass 2)."""
    h, w = outputs[0].shape[1:3]
    acc = np.zeros((total, h, w, 3), np.float64)
    wsum = np.zeros(total, np.float64)
    ends = [s + n - 1 for s, n in zip(starts, lengths)]

    def w_curr(k, kmax):
        if k < warmup or warmup >= kmax:
            return 0.0
        return (k - warmup) / (kmax - warmup)

    for ci, (s, n, fr) in enumerate(zip(starts, lengths, outputs)):
        e = s + n - 1
        prev_end = ends[ci - 1] if ci > 0 else -1
        next_start = starts[ci + 1] if ci + 1 < len(starts) else total
        for t in range(s, min(e, total - 1) + 1):
            if t <= prev_end:
                wt = w_curr(t - s, prev_end - s)
            elif t >= next_start:
                wt = 1.0 - w_curr(t - next_start, e - next_start)
            else:
                wt = 1.0
            acc[t] += wt * fr[t - s]
            wsum[t] += wt
    if (wsum <= 0).any():
        raise NodeError("internal: frames without chunk coverage")
    return (acc / wsum[:, None, None, None]).clip(0, 255).astype(np.uint8)


def _pad_4k1(frames: np.ndarray) -> np.ndarray:
    """Mirror-pad to a 4k+1 length (the VAE drops other tail frames)."""
    n = len(frames)
    target = ((n - 1 + 3) // 4) * 4 + 1
    while len(frames) < target:
        frames = np.concatenate([frames, frames[::-1]])
    return frames[:target]


# ---------------------------------------------------------------- run
def _env(ctx: Context, work: Path) -> dict:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    # torchvision raft_large(pretrained=True) looks in TORCH_HOME
    th = work / "torch_home" / "hub" / "checkpoints"
    th.mkdir(parents=True, exist_ok=True)
    link = th / RAFT_FILE
    if not link.exists():
        os.symlink(ctx.weight("raft") / RAFT_FILE, link)
    env["TORCH_HOME"] = str(work / "torch_home")
    # rp.select_torch_device picks a GPU by *physical* index from nvidia-smi,
    # which is invalid when CUDA_VISIBLE_DEVICES exposes a single GPU (the
    # runner always does). A sitecustomize on PYTHONPATH (also loaded by the
    # Go-with-the-Flow subprocess) pins it to the one visible device.
    site = work / "_sitecustomize"
    site.mkdir(parents=True, exist_ok=True)
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site)] + [x for x in [env.get("PYTHONPATH")] if x])
    return env


_SITECUSTOMIZE = """\
import os
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("",):
    try:
        import rp.r as _rpr
        import torch as _torch

        def _one_visible_device(*_a, **_k):
            return _torch.device("cuda:0")

        _rpr.select_torch_device = _one_visible_device
        import rp as _rp
        _rp.select_torch_device = _one_visible_device
    except Exception:  # rp not importable here: nothing to patch
        pass
"""


def _run(cmd, ctx: Context, env: dict, what: str) -> None:
    ctx.log("$", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ctx.repo), env=env)
    if proc.returncode != 0:
        raise NodeError(f"{what} failed (exit {proc.returncode})",
                        hint="see log.txt of the run for the traceback")


def _stage(dirp: Path, video, quad, prompt, fps):
    dirp.mkdir(parents=True, exist_ok=True)
    _write_mp4(dirp / "input_video.mp4", video, fps, lossless=False)
    _write_mp4(dirp / "quadmask_0.mp4", np.repeat(quad[..., None], 3, -1),
               fps, lossless=True)
    (dirp / "prompt.json").write_text(json.dumps({"bg": prompt}))


def _pass1(ctx, env, work, video, quad, prompt, p) -> np.ndarray:
    n = len(video)
    starts = _plan_pass1(n)
    counts = [min(CHUNK_SIZE, n)] * len(starts)
    outs = []
    for ci, (s, c) in enumerate(zip(starts, counts)):
        seq = f"seq_p1c{ci:02d}"
        root = work / "pass1" / f"chunk_{ci:02d}"
        v, q = _pad_4k1(video[s:s + c]), _pad_4k1(quad[s:s + c])
        _stage(root / "data" / seq, v, q, prompt, p["fps"])
        out_dir = root / "out"
        shutil.rmtree(out_dir, ignore_errors=True)
        cmd = [
            sys.executable, ctx.repo / "inference/cogvideox_fun/predict_v2v.py",
            f"--config={ctx.repo / 'config/quadmask_cogvideox.py'}",
            f"--config.system.gpu_memory_mode={p['gpu_memory_mode']}",
            f"--config.data.data_rootdir={root / 'data'}",
            f"--config.data.sample_size={p['height']}x{p['width']}",
            f"--config.data.fps={p['fps']}",
            f"--config.data.max_video_length={len(v)}",
            f"--config.experiment.run_seqs={seq}",
            f"--config.experiment.save_path={out_dir}",
            f"--config.video_model.model_name={ctx.weight('base')}",
            "--config.video_model.transformer_path="
            f"{ctx.weight('void') / 'void_pass1.safetensors'}",
            f"--config.video_model.temporal_window_size={WINDOW}",
            f"--config.video_model.guidance_scale={p['guidance_scale']}",
            f"--config.system.seed={p['seed']}",
            f"--config.video_model.num_inference_steps={p['steps']}",
        ]
        _run(cmd, ctx, env, f"VOID pass 1 chunk {ci}")
        cands = sorted(x for x in out_dir.glob(f"{seq}-fg=-1-*.mp4")
                       if not x.name.endswith("_tuple.mp4"))
        if not cands:
            raise NodeError(f"pass 1 chunk {ci} wrote no video")
        fr = _read_mp4(cands[-1])
        if len(fr) < c:
            raise NodeError(f"pass 1 chunk {ci}: {len(fr)} frames < {c}")
        outs.append(fr[:c])
    if len(starts) == 1:
        return outs[0][:n]
    return _stitch(outs, starts, counts, n, warmup=0)


def _pass2(ctx, env, work, video, quad, pass1, prompt, p) -> np.ndarray:
    n = len(video)
    starts = _plan_pass2(n)
    outs = []
    for ci, s in enumerate(starts):
        root = work / "pass2" / f"chunk_{ci:02d}"
        name = "seq"
        sl = slice(s, s + WINDOW)
        _stage(root / "data" / name, video[sl], quad[sl], prompt, p["fps"])
        (root / "pass1").mkdir(parents=True, exist_ok=True)
        _write_mp4(root / "pass1" / f"{name}-fg=-1-0001.mp4", pass1[sl],
                   p["fps"], lossless=False)
        out_dir = root / "out"
        cmd = [
            sys.executable,
            ctx.repo / "inference/cogvideox_fun/"
            "inference_with_pass1_warped_noise.py",
            "--video_name", name,
            "--data_rootdir", root / "data",
            "--pass1_dir", root / "pass1",
            "--output_dir", out_dir,
            "--warped_noise_cache_dir", root / "noise_cache",
            "--model_name", ctx.weight("base"),
            "--model_checkpoint", ctx.weight("void") / "void_pass2.safetensors",
            "--max_video_length", WINDOW,
            "--temporal_window_size", WINDOW,
            "--height", p["height"], "--width", p["width"],
            "--guidance_scale", p["guidance_scale"],
            "--num_inference_steps", p["steps"],
            "--seed", p["seed"],
            "--use_quadmask",
        ]
        _run(cmd, ctx, env, f"VOID pass 2 chunk {ci}")
        mp4 = out_dir / f"{name}_warped_noise_inference.mp4"
        if not mp4.exists():
            raise NodeError(f"pass 2 chunk {ci} produced no video",
                            hint="upstream logs errors and continues; see "
                            "log.txt (warped noise / RAFT / model load)")
        outs.append(_read_mp4(mp4))
    return _stitch(outs, starts, [WINDOW] * len(starts), n, warmup=WARMUP)


def inpaint(ctx: Context) -> None:
    import cv2
    from PIL import Image

    # ffmpeg (linked into the venv bin by setup) for mediapy in this process
    os.environ["PATH"] = (str(Path(sys.executable).parent) + os.pathsep
                          + os.environ.get("PATH", ""))
    files = io.list_frames(ctx.input("frames"))
    frames = np.stack([io.read_image(f) for f in files])
    n, H, W = frames.shape[:3]
    mfiles = io.list_frames(ctx.input("masks"), (".png",))
    if len(mfiles) != n:
        raise NodeError(f"masks has {len(mfiles)} frames, frames has {n}")
    remove = np.stack([np.asarray(Image.open(f)) != 0 for f in mfiles])
    if remove.shape[1:] != (H, W):
        remove = _resize(remove.astype(np.uint8), H, W, nearest=True) > 0
    if not remove.any():
        raise NodeError("masks are empty: nothing to remove")
    affected = np.zeros_like(remove)
    if ctx.has_input("affected"):
        afiles = io.list_frames(ctx.input("affected"), (".png",))
        if len(afiles) != n:
            raise NodeError(f"affected has {len(afiles)} frames, need {n}")
        affected = np.stack([np.asarray(Image.open(f)) != 0 for f in afiles])
        if affected.shape[1:] != (H, W):
            affected = _resize(affected.astype(np.uint8), H, W,
                               nearest=True) > 0
    d = int(ctx.param("dilate_px"))
    if d > 0:  # SimFoundry dilates before encoding (VOID skips it)
        k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
        remove = np.stack([cv2.dilate(m.astype(np.uint8), k) > 0
                           for m in remove])
    p = {"height": int(ctx.param("height")), "width": int(ctx.param("width")),
         "fps": int(ctx.param("fps")), "seed": int(ctx.param("seed")),
         "guidance_scale": float(ctx.param("guidance_scale")),
         "steps": int(ctx.param("num_inference_steps")),
         "gpu_memory_mode": ctx.param("gpu_memory_mode")}
    if p["height"] % 16 or p["width"] % 16:
        raise NodeError("height/width must be multiples of 16")
    # quadmask: 0 remove, 63 remove & affected, 127 affected, 255 keep
    quad = np.full(remove.shape, 255, np.uint8)
    quad[affected] = 127
    quad[remove] = 0
    quad[remove & affected] = 63
    # work at the model resolution (as SimFoundry, whose frames are 672x384)
    video = _resize(frames, p["height"], p["width"])
    quad = _resize(quad, p["height"], p["width"], nearest=True)

    work = ctx.output_path("_work")
    env = _env(ctx, work)
    prompt = ctx.param("prompt")
    out = _pass1(ctx, env, work, video, quad, prompt, p)
    if int(ctx.param("passes")) == 2:
        out = _pass2(ctx, env, work, video, quad, out, prompt, p)
    out = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
                    for f in out]) if out.shape[1:3] != (H, W) else out
    out_dir = ctx.output_path("frames")
    io.write_image_seq(out_dir, out, fps=p["fps"])
    ctx.set_output("frames", out_dir)
    qdir = work / "quadmask"
    qdir.mkdir(exist_ok=True)
    for i, q in enumerate(quad):
        Image.fromarray(q).save(qdir / io.frame_name(i))
    ctx.metadata.update({"passes": int(ctx.param("passes")),
                         "pass1_chunks": _plan_pass1(n),
                         "pass2_chunks": _plan_pass2(n)})
    shutil.rmtree(work / "torch_home", ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main({"inpaint": inpaint}))
