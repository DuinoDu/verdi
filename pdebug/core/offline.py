"""Prepare everything an offline GPU host cannot download itself.

Run on a machine WITH internet (e.g. the ubuntu laptop)::

    otn-cli offline-fetch --all --out /data/pdebug-offline

then copy ``<out>/git-mirror`` and ``<out>/weights`` to the offline host and
point its ``$PDEBUG_HOME/config.toml`` at them::

    [network]
    github_mirror = "<copied>/git-mirror"
    [paths]
    weights = "<copied>/weights"

Collected:
- every ``https://github.com/<org>/<repo>`` referenced by a node's
  manifest.toml / pyproject.toml / uv.lock / patches (uv git deps, the
  ``[upstream]`` repo, repos cloned by ``[setup].commands``), mirrored with
  ``git clone --mirror`` (+ their submodules, recursively);
- every ``[weights]`` entry with a ``url`` (HF weights are skipped: offline
  hosts are expected to reach an HF mirror; pass ``--hf`` to fetch them too).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set

from pdebug.core import paths
from pdebug.core.manifest import Manifest

GH = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def _log(msg: str) -> None:
    print(f"[offline] {msg}", file=sys.stderr, flush=True)


def github_repos(m: Manifest) -> Set[str]:
    """``org/repo`` names referenced by a node's files."""
    found: Set[str] = set()
    files = [m.node_dir / n for n in ("manifest.toml", "pyproject.toml",
                                      "uv.lock")]
    files += sorted((m.node_dir / "patches").glob("*")) \
        if (m.node_dir / "patches").exists() else []
    for f in files:
        if f.is_file():
            for org, repo in GH.findall(f.read_text(errors="ignore")):
                repo = re.sub(r"(\.git)?([?#].*)?$", "", repo)
                if repo:
                    found.add(f"{org}/{repo}")
    return found


def _git(args: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1",
               GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def mirror_repo(name: str, root: Path, seen: Set[str]) -> None:
    if name in seen:
        return
    seen.add(name)
    org, repo = name.split("/", 1)
    dest = root / "github.com" / org / f"{repo}.git"
    if dest.exists():
        _log(f"update {name}")
        r = _git(["remote", "update", "--prune"], cwd=dest)
    else:
        _log(f"mirror {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = _git(["clone", "--mirror", f"https://github.com/{name}.git",
                  str(dest)])
    if r.returncode != 0:
        _log(f"FAILED {name}: {r.stderr.strip()[-300:]}")
        return
    link = dest.parent / repo
    if not link.exists():
        link.symlink_to(dest.name)
    # submodules at any branch head / tag are best effort: scan all refs
    refs = _git(["for-each-ref", "--format=%(objectname)",
                 "refs/heads", "refs/tags"], cwd=dest).stdout.split()
    subs: Set[str] = set()
    for ref in refs[:50]:
        txt = _git(["show", f"{ref}:.gitmodules"], cwd=dest).stdout
        for o, rp in GH.findall(txt):
            subs.add(f"{o}/{re.sub(r'(.git)?$', '', rp)}")
    for sub in sorted(subs):
        mirror_repo(sub, root, seen)


def fetch_url_weights(m: Manifest, root: Path, hf: bool) -> None:
    """Download url (and optionally hf) weights into root/<node>/<key>."""
    from pdebug.core import install

    os.environ["PDEBUG_OFFLINE_WEIGHTS"] = str(root)
    weights = {k: w for k, w in m.weights.items()
               if not w.get("lazy") and (hf or "hf" not in w)}
    if not weights:
        return
    orig = paths.weights_dir
    paths.weights_dir = lambda node: root / node  # type: ignore
    try:
        sub = Manifest(**{**m.__dict__, "weights": weights})
        install.fetch_weights(sub)
    finally:
        paths.weights_dir = orig  # type: ignore


def offline_fetch(manifests: Iterable[Manifest], out: Path,
                  weights: bool = True, hf: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    failed: List[str] = []
    for m in manifests:
        _log(f"== {m.name}")
        for name in sorted(github_repos(m)):
            mirror_repo(name, out / "git-mirror", seen)
        if weights:
            try:
                fetch_url_weights(m, out / "weights", hf)
            except Exception as exc:  # noqa: BLE001  keep going
                failed.append(f"{m.name}: {exc}")
                _log(f"FAILED weights {m.name}: {exc}")
    _log(f"done: {len(seen)} repos mirrored under {out / 'git-mirror'}")
    for f in failed:
        _log(f"weights failure: {f}")
