"""Run one task of a node in its own venv as a one-shot subprocess."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from verdi.core import doctor, envelope, install, paths
from verdi.core.manifest import Manifest
from verdi.types.io import VIDEO_EXT, video_to_image_seq


def _git_commit(path: Path) -> Optional[str]:
    out = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "rev-parse", "HEAD"],
        cwd=path, capture_output=True, text=True)
    return out.stdout.strip() or None


def _prepare_inputs(task, inputs: Dict[str, str], run_dir: Path):
    """Convenience conversions done before validation (video -> frames)."""
    prepared = dict(inputs)
    for name, path in inputs.items():
        port = task.inputs.get(name)
        if port and port.type == "image_seq" and Path(path).is_file() \
                and Path(path).suffix.lower() in VIDEO_EXT:
            prepared[name] = str(video_to_image_seq(
                Path(path), run_dir / "inputs" / name))
    return prepared


def _device_env(device: str) -> Dict[str, str]:
    if device == "cpu":
        return {"CUDA_VISIBLE_DEVICES": ""}
    if device.startswith("cuda:"):
        return {"CUDA_VISIBLE_DEVICES": device.split(":", 1)[1]}
    return {}


def run(
    m: Manifest,
    task_name: Optional[str],
    inputs: Dict[str, str],
    params: Dict[str, Any],
    device: str = "cuda",
    output_dir: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    stream_log: bool = True,
) -> Dict[str, Any]:
    """Execute a task and return the (validated) result dict."""
    task = m.task(task_name)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = paths.runs_dir(m.name) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(output_dir).resolve() if output_dir else run_dir / "out"

    not_ready = install.is_ready(m)
    if not_ready:
        return envelope.error_result(
            "setup", f"node {m.name}: {not_ready}",
            hint=f"verdi setup {m.name}")
    try:
        inputs = _prepare_inputs(task, inputs, run_dir)
        request = envelope.build_request(m, task, inputs, params, device,
                                         out_dir)
    except (envelope.RequestError, ValueError) as exc:
        return envelope.error_result(
            "request", str(exc), hint=f"verdi describe {m.name}")

    req_path, res_path = run_dir / "request.json", run_dir / "result.json"
    log_path = run_dir / "log.txt"
    envelope.write_json(req_path, request)
    env = install.base_env(m)
    env.update(_device_env(device))
    python = paths.venv_dir(m.name) / "bin" / "python"
    cmd = [str(python), str(m.entry), str(req_path), str(res_path)]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    timeout = timeout_sec or m.timeout_sec
    state = {"timed_out": False}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, bufsize=1)

        def _kill():
            state["timed_out"] = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout, _kill)
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if stream_log:
                    sys.stderr.write(line)
            proc.wait()
        finally:
            timer.cancel()
    timed_out = state["timed_out"]
    duration = round(time.time() - t0, 2)

    tail = log_path.read_text()[-4000:]
    if timed_out:
        result = envelope.error_result(
            "timeout", f"exceeded {timeout}s", log_tail=tail,
            hint="raise --timeout or use a smaller input")
    elif not res_path.exists():
        result = envelope.error_result(
            "crash", f"node exited with code {proc.returncode} without "
            "writing result.json", log_tail=tail)
    else:
        result = json.loads(res_path.read_text())
        problems = envelope.check_result(task, result)
        if problems:
            result = envelope.error_result(
                "contract", "; ".join(problems), log_tail=tail,
                hint="node output violates its manifest; this is a node bug")
        elif result.get("status") != "ok":
            result.setdefault("error", {})["log_tail"] = tail

    rec = install.read_record(m.name)
    gpus = doctor.gpu_info() if device != "cpu" else []
    result["provenance"] = {
        "node": m.name, "node_version": m.version, "task": task.name,
        "upstream_repo": rec.get("upstream_repo"),
        "upstream_commit": rec.get("upstream_commit"),
        "weights": rec.get("weights"), "env_lock": rec.get("env_lock"),
        "device": device, "gpu": gpus[0]["name"] if gpus else None,
        "host": socket.gethostname(), "started_at": started,
        "duration_sec": duration, "verdi_commit": _git_commit(
            paths.REPO_ROOT),
        "run_dir": str(run_dir),
    }
    envelope.write_json(res_path, result)
    return result
