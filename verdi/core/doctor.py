"""Check the system requirements declared in ``[system]``.

uv owns Python dependencies; everything uv cannot install (driver, CUDA
toolkit, compilers, shared libraries, binaries, disk, tokens) is declared
in the manifest and checked here. Failing checks carry a fix command.

``[system]`` keys::

    gpu       = { min_vram_gb = 24, arch = ["sm_90"], required = true }
    cuda      = { toolkit = ">=12.4", driver = ">=12.4" }
    compilers = { gcc = ">=9,<13" }
    binaries  = ["ffmpeg", { name = "blenderproc", fix = "..." }]
    libs      = ["libGL.so.1", { name = "libEGL.so.1", apt = "libegl1" }]
    disk_gb   = 30
    env       = ["GEMINI_API_KEY"]
    hf_token  = true
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from verdi.core import paths
from verdi.core.manifest import Manifest

APT_HINTS = {
    "ffmpeg": "ffmpeg", "ffprobe": "ffmpeg", "git": "git",
    "git-lfs": "git-lfs", "cmake": "cmake", "ninja": "ninja-build",
    "libGL.so.1": "libgl1", "libEGL.so.1": "libegl1",
    "libglib-2.0.so.0": "libglib2.0-0", "libSM.so.6": "libsm6",
    "libXext.so.6": "libxext6", "libXrender.so.1": "libxrender1",
    "libOpenGL.so.0": "libopengl0", "libgomp.so.1": "libgomp1",
    "libosmesa.so": "libosmesa6-dev", "libOSMesa.so.8": "libosmesa6",
    "libeigen3": "libeigen3-dev", "libboost_system.so": "libboost-all-dev",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _cmd(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(args, capture_output=True, text=True,
                             timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _spec_ok(found: str, spec: str) -> bool:
    return Version(found) in SpecifierSet(spec)


def gpu_info() -> List[Dict[str, Any]]:
    out = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        name, mem, cap = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "vram_gb": float(mem) / 1024,
                     "arch": "sm_" + cap.replace(".", "")})
    return gpus


def driver_cuda() -> Optional[str]:
    out = _cmd(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def toolkit_cuda(cuda_home: Optional[str]) -> Optional[str]:
    candidates = []
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    candidates += [shutil.which("nvcc") or "", "/usr/local/cuda/bin/nvcc"]
    for nvcc in candidates:
        if nvcc and os.path.exists(nvcc):
            out = _cmd([nvcc, "--version"]) or ""
            m = re.search(r"release ([\d.]+)", out)
            if m:
                return m.group(1)
    return None


def _named(item: Any) -> Dict[str, str]:
    return {"name": item} if isinstance(item, str) else dict(item)


def run_checks(manifest: Manifest) -> List[Check]:
    sysreq = manifest.system
    checks: List[Check] = []
    try:
        checks.append(Check("uv", True, paths.uv_bin()))
    except FileNotFoundError as exc:
        checks.append(Check("uv", False, "not found", str(exc)))

    gpu = sysreq.get("gpu")
    if gpu:
        gpus = gpu_info()
        if not gpus:
            checks.append(Check("gpu", not gpu.get("required", True),
                                "no NVIDIA GPU visible",
                                "check nvidia-smi / driver"))
        else:
            g = max(gpus, key=lambda x: x["vram_gb"])
            need = float(gpu.get("min_vram_gb", 0))
            checks.append(Check(
                "gpu.vram", g["vram_gb"] >= need,
                f"{g['name']} {g['vram_gb']:.0f} GB (need {need:.0f})",
                "use a larger GPU or a smaller checkpoint"))
            archs = gpu.get("arch")
            if archs:
                checks.append(Check(
                    "gpu.arch", g["arch"] in archs,
                    f"{g['arch']} (supported {archs})",
                    "node has not been verified on this GPU arch"))

    cuda = sysreq.get("cuda", {})
    if "driver" in cuda:
        found = driver_cuda()
        checks.append(Check(
            "cuda.driver", bool(found) and _spec_ok(found, cuda["driver"]),
            f"driver supports CUDA {found} (need {cuda['driver']})",
            "upgrade the NVIDIA driver or install cuda-compat"))
    if "toolkit" in cuda:
        from verdi.core.install import expand

        cuda_home = manifest.build_env.get("CUDA_HOME")
        found = toolkit_cuda(expand(cuda_home, manifest) if cuda_home
                             else None)
        checks.append(Check(
            "cuda.toolkit", bool(found) and _spec_ok(found, cuda["toolkit"]),
            f"nvcc {found} (need {cuda['toolkit']})",
            "install a matching CUDA toolkit and set [build.env].CUDA_HOME"))

    for comp, spec in sysreq.get("compilers", {}).items():
        out = _cmd([comp, "-dumpfullversion"]) or _cmd([comp, "-dumpversion"])
        found = (out or "").strip()
        checks.append(Check(
            f"compiler.{comp}", bool(found) and _spec_ok(found, spec),
            f"{comp} {found or 'missing'} (need {spec})",
            f"apt-get install -y {comp}"))

    for item in sysreq.get("binaries", []):
        item = _named(item)
        name = item["name"]
        found = shutil.which(name)
        fix = item.get("fix") or (
            f"apt-get install -y {item.get('apt') or APT_HINTS.get(name, name)}")
        checks.append(Check(f"bin.{name}", bool(found), found or "missing",
                            fix))

    ldcache = _cmd(["ldconfig", "-p"]) or ""
    for item in sysreq.get("libs", []):
        item = _named(item)
        name = item["name"]
        found = name in ldcache
        apt = item.get("apt") or APT_HINTS.get(name, "<package providing "
                                               f"{name}>")
        checks.append(Check(f"lib.{name}", found,
                            "found" if found else "missing",
                            f"apt-get install -y {apt}"))

    if "disk_gb" in sysreq:
        home = paths.home()
        home.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(home).free / 1e9
        checks.append(Check(
            "disk", free >= float(sysreq["disk_gb"]),
            f"{free:.0f} GB free at {home} (need {sysreq['disk_gb']})",
            "free space or point VERDI_HOME elsewhere"))

    for var in sysreq.get("env", []):
        checks.append(Check(f"env.{var}", bool(os.environ.get(var)),
                            "set" if os.environ.get(var) else "unset",
                            f"export {var}=..."))

    if sysreq.get("hf_token"):
        env = paths.node_env(manifest.name)
        ok = bool(os.environ.get("HF_TOKEN") or env.get("HF_TOKEN_PATH"))
        checks.append(Check("hf_token", ok, "found" if ok else "missing",
                            "huggingface-cli login (and accept the model "
                            "licence on its HF page)"))
    return checks


def format_checks(manifest: Manifest, checks: List[Check]) -> str:
    lines = [f"doctor {manifest.name}:"]
    for c in checks:
        mark = "OK  " if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"         fix: {c.fix}")
    return "\n".join(lines)
