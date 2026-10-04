"""``verdi setup``: upstream checkout, uv environment, weights.

``[upstream]``::

    repo = "https://github.com/facebookresearch/sam2.git"
    commit = "<sha>"            # required, never a branch
    submodules = false
    lfs = false
    patches = ["patches/fix.patch"]   # applied with `git apply`

``[weights.<key>]`` (downloaded into VERDI_HOME/weights/<node>/<key>)::

    hf = "facebook/sam2.1-hiera-large"  revision = "..."  files = ["*.pt"]
    url = "https://..."  sha256 = "..."  filename = "model.pt"
    url = "https://.../ckpt.zip"  archive = "zip"   # or "tar"; extracted
        sha256 = "<of the archive>"  files_sha256 = { "a/b.pt" = "..." }
    lazy = true      # upstream code downloads itself into HF_HOME

``[setup]`` (runs after uv sync and weights)::

    commands = ["python setup.py build_ext --inplace"]  # cwd = repo,
                                                         # node venv on PATH
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from verdi.core import config, doctor, paths
from verdi.core.manifest import Manifest

RECORD = ".verdi_setup.json"


class SetupError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[setup] {msg}", file=sys.stderr, flush=True)


def _run(args: List[str], cwd: Optional[Path] = None,
         env: Optional[Dict[str, str]] = None) -> None:
    _log("$ " + " ".join(str(a) for a in args))
    proc = subprocess.run([str(a) for a in args], cwd=cwd, env=env)
    if proc.returncode != 0:
        raise SetupError(f"command failed ({proc.returncode}): "
                         + " ".join(str(a) for a in args))


def base_env(m: Manifest, build: bool = False) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(paths.node_env(m.name))
    env["UV_PYTHON_INSTALL_DIR"] = str(paths.home() / "python")
    extra = dict(m.run_env)
    if build:
        extra.update(m.build_env)
    for k, v in extra.items():
        env[k] = expand(v, m)
    cuda_home = env.get("CUDA_HOME")
    if build and cuda_home and os.path.isdir(os.path.join(cuda_home, "bin")):
        env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    return env


def expand(value: str, m: Manifest) -> str:
    build = config.build_cfg(str(paths.home()))
    return value.format(
        repo=paths.repos_dir(m.name), weights=paths.weights_dir(m.name),
        home=paths.home(), node_dir=m.node_dir, venv=paths.venv_dir(m.name),
        cuda_home=build.get("cuda_home", "/usr/local/cuda"),
        cuda_arch=build.get("cuda_arch", "9.0"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_hash(m: Manifest) -> Optional[str]:
    lock = m.node_dir / "uv.lock"
    return "sha256:" + sha256(lock) if lock.exists() else None


LEGACY = {".verdi_setup.json": ".pdebug_setup.json",
          ".verdi_complete": ".pdebug_complete",
          ".verdi_extracted": ".pdebug_extracted",
          ".verdi_patches": ".pdebug_patches"}


def _marker(path: Path) -> Path:
    """Return ``path`` or its pre-rename (pdebug) equivalent if only that
    exists, so installs and weights made before the rename stay valid."""
    if not path.exists() and path.name in LEGACY:
        old = path.with_name(LEGACY[path.name])
        if old.exists():
            return old
    return path


def read_record(node: str) -> Dict[str, Any]:
    path = _marker(paths.venv_dir(node) / RECORD)
    return json.loads(path.read_text()) if path.exists() else {}


def is_ready(m: Manifest) -> Optional[str]:
    """Return None when the node can run, else the reason."""
    rec = read_record(m.name)
    if not rec:
        return "not set up"
    if rec.get("env_lock") != lock_hash(m):
        return "uv.lock changed since setup"
    if m.upstream and not str(rec.get("upstream_commit") or "").startswith(
            str(m.upstream.get("commit"))):
        return "upstream commit changed since setup"
    return None


# ---------------------------------------------------------------- steps
def checkout(m: Manifest) -> Optional[str]:
    up = m.upstream
    if not up:
        return None
    if "repo" not in up or "commit" not in up:
        raise SetupError("[upstream] needs repo and commit")
    dest = paths.repos_dir(m.name)
    env = dict(os.environ)
    env.update(paths.node_env(m.name))
    if not up.get("lfs"):
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "-c", "safe.directory=*", "clone", "--filter=blob:none", up["repo"], dest],
             env=env)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest,
                          capture_output=True, text=True).stdout.strip()
    patches = [m.node_dir / p for p in up.get("patches", [])]
    patch_sig = ",".join(sha256(p)[:12] for p in patches)
    marker = _marker(dest / ".verdi_patches")
    applied = marker.read_text() if marker.exists() else ""
    if head.startswith(up["commit"]) and applied == patch_sig:
        _log(f"upstream already at {up['commit'][:12]}")
        return head
    _run(["git", "fetch", "--quiet", "origin", up["commit"]], cwd=dest,
         env=env)
    _run(["git", "checkout", "--force", "--quiet", up["commit"]], cwd=dest,
         env=env)
    _run(["git", "clean", "-fdq"], cwd=dest)
    if up.get("submodules"):
        _run(["git", "submodule", "update", "--init", "--recursive"],
             cwd=dest, env=env)
    if up.get("lfs"):
        _run(["git", "lfs", "pull"], cwd=dest, env=env)
    for p in patches:
        _run(["git", "apply", "--whitespace=nowarn", p], cwd=dest)
    marker.write_text(patch_sig)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest,
                          capture_output=True, text=True).stdout.strip()


def uv_sync(m: Manifest) -> None:
    env = base_env(m, build=True)
    uv = paths.uv_bin()
    if not (m.node_dir / "uv.lock").exists():
        _run([uv, "lock", "--project", m.node_dir], env=env)
    _run([uv, "sync", "--frozen", "--project", m.node_dir], env=env)


def setup_commands(m: Manifest) -> None:
    cmds = m.raw.get("setup", {}).get("commands", [])
    if not cmds:
        return
    env = base_env(m, build=True)
    venv_bin = paths.venv_dir(m.name) / "bin"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(paths.venv_dir(m.name))
    cwd = paths.repos_dir(m.name) if m.upstream else m.node_dir
    for cmd in cmds:
        _run(["bash", "-lc", expand(cmd, m)], cwd=cwd, env=env)


def _rename_ok(root: Path) -> bool:
    """False when ``[paths] weights_no_rename = true`` (write-only bucket
    filesystems that refuse rename/unlink; probing would leave debris)."""
    cfg = config.paths_cfg(str(paths.home()))
    return not cfg.get("weights_no_rename", False)


def _copy_into(src: Path, dest: Path) -> None:
    """Copy a staged download into its final (no-rename) location."""
    if src.is_dir():
        for f in src.rglob("*"):
            rel = f.relative_to(src)
            if rel.parts and rel.parts[0] == ".cache":
                continue
            target = dest / rel
            if f.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists() or target.stat().st_size != \
                    f.stat().st_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


HF_MARKER = ".verdi_complete"


def _hf_id(w: Dict[str, Any]) -> str:
    return f"hf:{w['hf']}@{w.get('revision', 'main')}"


def _already_done(key: str, w: Dict[str, Any], dest: Path) -> Optional[str]:
    """Record string if the final location already holds this weight."""
    if w.get("lazy"):
        return f"lazy:{w.get('hf') or w.get('url')}"
    if "hf" in w:
        marker = _marker(dest / HF_MARKER)
        if marker.exists() and marker.read_text().strip() == _hf_id(w):
            return _hf_id(w)
        return None
    if "url" in w and w.get("archive"):
        done = _marker(dest / ".verdi_extracted")
        files = w.get("files_sha256", {})
        if done.exists() and all((dest / f).exists() and
                                 sha256(dest / f) == h
                                 for f, h in files.items()):
            return done.read_text().strip()
        return None
    if "url" in w:
        target = dest / w.get("filename", w["url"].rsplit("/", 1)[-1])
        if target.exists():
            digest = sha256(target)
            if not w.get("sha256") or digest == w["sha256"]:
                return f"sha256:{digest}"
    return None


def fetch_weights(m: Manifest) -> Dict[str, str]:
    """Fetch missing weights; keys already complete in place are skipped.

    HF snapshots get a ``.verdi_complete`` marker (repo@revision); url
    weights are verified by sha256; archives by ``files_sha256``.
    """
    root = paths.weights_dir(m.name)
    record: Dict[str, str] = {}
    todo = {}
    for key, w in m.weights.items():
        done = _already_done(key, w, root / key)
        if done:
            record[key] = done
        else:
            todo[key] = w
    if not todo:
        return record
    sub = Manifest(**{**m.__dict__, "weights": todo})
    if not _rename_ok(root.parent):
        # download locally (atomic renames work there), then copy over
        staging = paths.home() / "staging" / m.name
        _log(f"weights root forbids rename; staging in {staging}")
        orig = paths.weights_dir
        paths.weights_dir = lambda node: staging  # type: ignore
        try:
            record.update(_fetch_weights(sub))
        finally:
            paths.weights_dir = orig  # type: ignore
        for key in todo:
            if (staging / key).exists():
                _log(f"weights {key}: copy -> {root / key}")
                _copy_into(staging / key, root / key)
        shutil.rmtree(staging, ignore_errors=True)
    else:
        record.update(_fetch_weights(sub))
    for key, w in todo.items():
        if "hf" in w and not w.get("lazy"):
            (root / key).mkdir(parents=True, exist_ok=True)
            (root / key / HF_MARKER).write_text(_hf_id(w))
    return record


def _fetch_weights(m: Manifest) -> Dict[str, str]:
    record = {}
    for key, w in m.weights.items():
        dest = paths.weights_dir(m.name) / key
        if w.get("lazy"):
            record[key] = f"lazy:{w.get('hf') or w.get('url')}"
            continue
        if "hf" in w:
            env = paths.node_env(m.name)
            for k in ("HF_HOME", "HF_TOKEN_PATH", "HF_ENDPOINT",
                      "HF_HUB_DISABLE_XET"):
                if k in env:
                    os.environ.setdefault(k, env[k])
            from huggingface_hub import snapshot_download

            _log(f"weights {key}: hf {w['hf']}")
            snapshot_download(
                repo_id=w["hf"], revision=w.get("revision"),
                allow_patterns=w.get("files"), local_dir=str(dest),
                repo_type=w.get("repo_type", "model"),
                endpoint=env.get("HF_ENDPOINT"))
            record[key] = f"hf:{w['hf']}@{w.get('revision', 'main')}"
        elif "url" in w and w.get("archive"):
            record[key] = _fetch_archive(key, w, dest)
        elif "url" in w:
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / w.get("filename", w["url"].rsplit("/", 1)[-1])
            if not target.exists() or (
                    w.get("sha256") and sha256(target) != w["sha256"]):
                _log(f"weights {key}: {w['url']}")
                tmp = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(
                    config.rewrite_url(w["url"], str(paths.home())), tmp)
                tmp.rename(target)
            digest = sha256(target)
            if w.get("sha256") and digest != w["sha256"]:
                raise SetupError(f"weights {key}: sha256 mismatch {digest}")
            record[key] = f"sha256:{digest}"
        else:
            raise SetupError(f"weights.{key}: needs hf, url or lazy")
    return record


def _fetch_archive(key: str, w: Dict[str, Any], dest: Path) -> str:
    """Download + extract an archive; verify archive and per-file hashes."""
    import tarfile
    import zipfile

    files = w.get("files_sha256", {})
    done = _marker(dest / ".verdi_extracted")
    if done.exists() and all(
            (dest / f).exists() and sha256(dest / f) == h
            for f, h in files.items()):
        return done.read_text()
    dest.mkdir(parents=True, exist_ok=True)
    arc = dest.parent / f".{key}.{w['archive']}"
    if not arc.exists() or (w.get("sha256") and sha256(arc) != w["sha256"]):
        _log(f"weights {key}: {w['url']}")
        tmp = arc.with_suffix(arc.suffix + ".part")
        urllib.request.urlretrieve(
            config.rewrite_url(w["url"], str(paths.home())), tmp)
        tmp.rename(arc)
    digest = sha256(arc)
    if w.get("sha256") and digest != w["sha256"]:
        raise SetupError(f"weights {key}: archive sha256 mismatch {digest}")
    if w["archive"] == "zip":
        with zipfile.ZipFile(arc) as z:
            z.extractall(dest)
    else:
        with tarfile.open(arc) as t:
            t.extractall(dest)
    for f, h in files.items():
        if not (dest / f).exists() or sha256(dest / f) != h:
            raise SetupError(f"weights {key}: {f} missing or sha256 mismatch")
    done.write_text(f"sha256:{digest}")
    arc.unlink()
    return f"sha256:{digest}"


def setup(m: Manifest, force: bool = False,
          skip_weights: bool = False) -> Dict[str, Any]:
    checks = doctor.run_checks(m)
    print(doctor.format_checks(m, checks), file=sys.stderr)
    if not all(c.ok for c in checks) and not force:
        raise SetupError("system checks failed (fix them or pass --force)")
    t0 = time.time()
    commit = checkout(m)
    uv_sync(m)
    weights = {} if skip_weights else fetch_weights(m)
    setup_commands(m)  # after weights: commands may use them
    record = {
        "node": m.name, "node_version": m.version,
        "upstream_repo": m.upstream.get("repo"), "upstream_commit": commit,
        "env_lock": lock_hash(m), "weights": weights,
        "setup_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "setup_sec": round(time.time() - t0, 1),
    }
    venv = paths.venv_dir(m.name)
    venv.mkdir(parents=True, exist_ok=True)
    (venv / RECORD).write_text(json.dumps(record, indent=2))
    return record


def remove(m: Manifest) -> None:
    for d in (paths.venv_dir(m.name),):
        if d.exists():
            shutil.rmtree(d)
