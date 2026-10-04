"""Filesystem layout of the repository and of ``VERDI_HOME``."""
from __future__ import annotations

import os
from pathlib import Path

from verdi.core import config

REPO_ROOT = Path(__file__).resolve().parents[2]
NODES_DIR = REPO_ROOT / "nodes"

_DEFAULT_HOME = "~/.cache/verdi"


def home() -> Path:
    """Return ``VERDI_HOME`` (repos, venvs, weights, caches, runs)."""
    # VERDI_HOME; PDEBUG_HOME is accepted for installs made before the rename
    value = (os.environ.get("VERDI_HOME") or os.environ.get("PDEBUG_HOME")
             or _DEFAULT_HOME)
    return Path(value).expanduser()


def repos_dir(node: str) -> Path:
    return home() / "repos" / node


def venv_dir(node: str) -> Path:
    return home() / "venvs" / node


def _override(key: str, default: Path) -> Path:
    value = config.paths_cfg(str(home())).get(key)
    return Path(value).expanduser() if value else default


def weights_root() -> Path:
    return _override("weights", home() / "weights")


def weights_dir(node: str) -> Path:
    return weights_root() / node


def runs_dir(node: str) -> Path:
    return _override("runs", home() / "runs") / node


def uv_cache_dir() -> Path:
    return home() / "uv-cache"


def hf_home() -> Path:
    return _override("hf_cache", home() / "hf-cache")


def uv_bin() -> str:
    """Locate the ``uv`` executable (PATH first, then VERDI_HOME/bin)."""
    from shutil import which

    found = which("uv")
    if found:
        return found
    candidate = home() / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh "
        f"| env UV_INSTALL_DIR={home() / 'bin'} UV_NO_MODIFY_PATH=1 sh"
    )


def node_env(node: str) -> dict:
    """Environment variables shared by setup/run for one node."""
    return {
        "VERDI_HOME": str(home()),
        "VERDI_NODE": node,
        "VERDI_NODE_REPO": str(repos_dir(node)),
        "VERDI_NODE_WEIGHTS": str(weights_dir(node)),
        "UV_CACHE_DIR": str(uv_cache_dir()),
        "UV_PROJECT_ENVIRONMENT": str(venv_dir(node)),
        "HF_HOME": str(hf_home()),
        **_hf_token_env(),
        **config.network_env(str(home())),
    }


def _hf_token_env() -> dict:
    """Keep using the user's HF token although HF_HOME is relocated."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_PATH"):
        return {}
    token = Path("~/.cache/huggingface/token").expanduser()
    return {"HF_TOKEN_PATH": str(token)} if token.exists() else {}
