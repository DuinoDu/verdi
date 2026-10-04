"""Request context and entry-point dispatcher for node processes."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class NodeError(RuntimeError):
    """Expected failure with an actionable hint for the caller."""

    def __init__(self, message: str, hint: str = "", kind: str = "node"):
        super().__init__(message)
        self.hint = hint
        self.kind = kind


def prefetch_files(path: Path, min_bytes: int = 1 << 26) -> None:
    """Sequentially read large files under ``path`` into the page cache."""
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file())
    for f in files:
        if f.stat().st_size < min_bytes:
            continue
        with open(f, "rb") as fh:
            while fh.read(1 << 24):
                pass


class Context:
    """Typed access to one request inside a node process."""

    def __init__(self, request: Dict[str, Any]) -> None:
        self.request = request
        self.task: str = request["task"]
        self.params: Dict[str, Any] = request.get("params", {})
        self.output_dir = Path(request["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.outputs: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

    # ---- environment
    @property
    def device(self) -> str:
        """``cuda`` or ``cpu`` (GPU index is applied via CUDA_VISIBLE_DEVICES)."""
        dev = self.request.get("device", "cuda")
        return "cpu" if dev == "cpu" else "cuda"

    @property
    def repo(self) -> Path:
        """Checkout of the pinned upstream repository."""
        return Path(os.environ["VERDI_NODE_REPO"])

    @property
    def weights(self) -> Path:
        """Directory holding weights declared in the manifest."""
        return Path(os.environ["VERDI_NODE_WEIGHTS"])

    def weight(self, key: str, prefetch: bool = False) -> Path:
        """Path of a manifest weight; ``prefetch`` warms the page cache.

        Network filesystems (vepfs) make mmap-based safetensors loading very
        slow on a cold cache; a sequential read first is ~10x faster.
        """
        path = self.weights / key
        if not path.exists():
            raise NodeError(f"weight {key!r} missing at {path}",
                            hint=f"run `verdi setup {self.node}`",
                            kind="setup")
        if prefetch:
            prefetch_files(path)
        return path

    @property
    def node(self) -> str:
        return os.environ.get("VERDI_NODE", self.request.get("node", ""))

    # ---- inputs / params
    def has_input(self, name: str) -> bool:
        return name in self.request.get("inputs", {})

    def input(self, name: str) -> Path:
        try:
            return Path(self.request["inputs"][name]["path"])
        except KeyError:
            raise NodeError(f"input {name!r} not provided") from None

    def input_summary(self, name: str) -> Dict[str, Any]:
        return self.request["inputs"][name].get("summary", {})

    def param(self, name: str, default: Any = None) -> Any:
        value = self.params.get(name)
        return default if value is None else value

    # ---- outputs
    def output_path(self, name: str, filename: Optional[str] = None) -> Path:
        """Path inside output_dir for an output (a dir if no filename)."""
        path = self.output_dir / (filename or name)
        if filename is None:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def set_output(self, name: str, path: Path, type_: Optional[str] = None,
                   **extra: Any) -> None:
        """Register an output; type defaults to the manifest declaration."""
        ref = {"type": type_ or self._declared_type(name),
               "path": str(Path(path).resolve())}
        ref.update(extra)
        self.outputs[name] = ref

    def _declared_type(self, name: str) -> str:
        declared = self.request.get("output_types", {})
        if name not in declared:
            raise NodeError(f"output {name!r} is not declared by the task")
        return declared[name]

    @staticmethod
    def log(*args: Any) -> None:
        print(*args, file=sys.stderr, flush=True)


def main(tasks: Dict[str, Callable[[Context], None]],
         argv: Optional[list] = None) -> int:
    """Entry point: ``python entry.py request.json result.json``."""
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("usage: entry.py request.json result.json", file=sys.stderr)
        return 2
    req_path, res_path = Path(argv[0]), Path(argv[1])
    request = json.loads(req_path.read_text())
    result: Dict[str, Any]
    try:
        fn = tasks.get(request["task"])
        if fn is None:
            raise NodeError(f"entry has no task {request['task']!r}",
                            kind="manifest")
        ctx = Context(request)
        fn(ctx)
        result = {"status": "ok", "outputs": ctx.outputs,
                  "metadata": ctx.metadata, "error": None}
    except NodeError as exc:
        result = {"status": "error", "outputs": {}, "error": {
            "kind": exc.kind, "message": str(exc), "hint": exc.hint}}
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        result = {"status": "error", "outputs": {}, "error": {
            "kind": type(exc).__name__, "message": str(exc),
            "hint": "", "traceback_tail": tb[-3000:]}}
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1
