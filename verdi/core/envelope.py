"""request.json / result.json construction and validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from verdi import types
from verdi.core.manifest import Manifest, Task


class RequestError(ValueError):
    """The caller passed invalid inputs or params."""


SCALED_TYPES = ("depth", "depth_seq", "pointcloud", "mesh")


def build_request(
    manifest: Manifest,
    task: Task,
    inputs: Dict[str, str],
    params: Dict[str, Any],
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Validate caller inputs/params and return a request dict."""
    unknown = set(inputs) - set(task.inputs)
    if unknown:
        raise RequestError(
            f"unknown inputs {sorted(unknown)}; task {task.name} takes "
            f"{sorted(task.inputs)}"
        )
    req_inputs = {}
    for name, port in task.inputs.items():
        if name not in inputs:
            if port.optional:
                continue
            raise RequestError(
                f"missing input {name!r} ({port.type}: "
                f"{types.TYPES[port.type].format})"
            )
        path = Path(inputs[name]).expanduser().resolve()
        try:
            summary = types.validate(port.type, path)
        except Exception as exc:  # noqa: BLE001
            raise RequestError(f"input {name}: {exc}") from exc
        if port.type in SCALED_TYPES and port.scale == "metric":
            st = summary.get("scale_status", "unspecified")
            if st not in types.registry.METRIC_SCALE_STATES + ("unspecified",):
                raise RequestError(
                    f"input {name}: {path} is declared scale_status={st!r}"
                    f" (source {summary.get('scale_source', '?')}); this input "
                    "needs metric data. Align it first (e.g. aruco_scale "
                    "scale_align) or use a metric producer; see the "
                    "<file>.scale.json / scale.json sidecar")
        req_inputs[name] = {"type": port.type, "path": str(path),
                            "summary": summary}
    unknown = set(params) - set(task.params)
    if unknown:
        raise RequestError(
            f"unknown params {sorted(unknown)}; task {task.name} takes "
            f"{sorted(task.params)}"
        )
    req_params = {}
    for name, spec in task.params.items():
        if name in params:
            req_params[name] = spec.coerce(params[name])
        elif spec.required:
            raise RequestError(f"missing required param {name!r}")
        else:
            req_params[name] = spec.default
    return {
        "node": manifest.name,
        "task": task.name,
        "inputs": req_inputs,
        "params": req_params,
        "output_types": {k: v.type for k, v in task.outputs.items()},
        "device": device,
        "output_dir": str(output_dir),
    }


def check_result(task: Task, result: Dict[str, Any]) -> List[str]:
    """Return a list of problems with a node's result (empty = valid)."""
    problems = []
    if result.get("status") != "ok":
        return problems
    outputs = result.get("outputs", {})
    for name, port in task.outputs.items():
        ref = outputs.get(name)
        if ref is None:
            if not port.optional:
                problems.append(f"missing output {name!r}")
            continue
        if ref.get("type") != port.type:
            problems.append(
                f"output {name}: type {ref.get('type')!r} != {port.type!r}"
            )
            continue
        try:
            ref["summary"] = types.validate(port.type, ref["path"])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"output {name}: {exc}")
    extra = set(outputs) - set(task.outputs)
    if extra:
        problems.append(f"undeclared outputs {sorted(extra)}")
    return problems


def error_result(kind: str, message: str, hint: str = "",
                 log_tail: Optional[str] = None) -> Dict[str, Any]:
    err = {"kind": kind, "message": message, "hint": hint}
    if log_tail:
        err["log_tail"] = log_tail
    return {"status": "error", "outputs": {}, "error": err}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
