"""``verdi test``: run the manifest fixtures and record STATUS.toml.

One ``[[tests]]`` entry per case (cover every task)::

    [[tests]]
    name = "image"                # optional, defaults to the task
    task = "segment_image"
    timeout_sec = 900
    inputs = { image = "tests/fixture/cups.jpg" }  # relative to node dir
    params = { text = "cup" }
    [[tests.checks]]              # summary field checks
    output = "mask"
    field = "instances"
    min = 1
    [[tests.checks]]              # value inside a json output
    output = "result"
    path = "objects.0.azimuth_deg"  # dotted path, list indices allowed
    min = 200
    [[tests]]                     # negative case: the run must FAIL
    task = "segment_image"        # with this text in error message/hint
    inputs = { image = "{repo}/assets/x.png" }   # {repo} = upstream checkout
    expect_error = "distortion"
    [[tests.checks]]              # comparison with a reference output
    output = "mask"
    metric = "mask_iou"           # see verdi.types.registry.COMPARATORS
    expected = "tests/expected/cups_mask.png"
    min = 0.8
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from verdi import types
from verdi.core import runner
from verdi.core.manifest import Manifest


def _resolve(data: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(data, list):
            data = data[int(part)]
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            return None
        if data is None:
            return None
    return data


def _check(chk: Dict[str, Any], result: Dict[str, Any],
           node_dir: Path) -> Tuple[bool, str]:
    out = result["outputs"].get(chk["output"])
    if out is None:
        return False, f"{chk['output']}: missing"
    if "metric" in chk:
        value = types.compare(chk["metric"], out["path"],
                              node_dir / chk["expected"])
        label = f"{chk['output']}.{chk['metric']}"
    elif "path" in chk:  # value inside a json output, e.g. "objects.0.x"
        import json

        value = _resolve(json.loads(Path(out["path"]).read_text()),
                         chk["path"])
        label = f"{chk['output']}:{chk['path']}"
    else:
        value = out.get("summary", {}).get(chk["field"])
        label = f"{chk['output']}.{chk['field']}"
    ok = True
    if "eq" in chk:
        ok &= value == chk["eq"]
    if "min" in chk:
        ok &= value is not None and value >= chk["min"]
    if "max" in chk:
        ok &= value is not None and value <= chk["max"]
    if "contains" in chk:
        ok &= value is not None and str(chk["contains"]).lower() in \
            str(value).lower()
    bounds = {k: chk[k] for k in ("eq", "min", "max", "contains") if k in chk}
    if isinstance(value, float):
        value = round(value, 4)
    return bool(ok), f"{label}={value!r} {bounds}"


def run_case(m: Manifest, case: Dict[str, Any],
             device: str) -> Dict[str, Any]:
    from verdi.core import paths

    def _where(v: str) -> str:  # "{repo}/..." = file of the upstream checkout
        if v.startswith("{repo}/"):
            return str(paths.repos_dir(m.name) / v[len("{repo}/"):])
        return str(m.node_dir / v)

    inputs = {k: _where(v) for k, v in case.get("inputs", {}).items()}
    result = runner.run(m, case["task"], inputs,
                        dict(case.get("params", {})), device=device,
                        timeout_sec=case.get("timeout_sec"))
    prov = result.get("provenance", {})
    details: List[str] = []
    passed = result.get("status") == "ok"
    if "expect_error" in case:  # negative case: the node must refuse
        err = result.get("error") or {}
        text = f"{err.get('message')} {err.get('hint', '')}"
        want = str(case["expect_error"])
        passed = result.get("status") == "error" and want.lower() in text.lower()
        details.append(("PASS " if passed else "FAIL ")
                       + f"expected error containing {want!r}: status="
                       + f"{result.get('status')} {text[:200]!r}")
    elif not passed:
        err = result.get("error") or {}
        details.append(f"run failed: {err.get('kind')}: "
                       f"{err.get('message')}")
    else:
        if not case.get("checks"):
            passed = False
            details.append("no checks declared; a test must check outputs")
        for chk in case.get("checks", []):
            try:
                ok, msg = _check(chk, result, m.node_dir)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{chk}: {exc}"
            passed &= ok
            details.append(("PASS " if ok else "FAIL ") + msg)
    return {"name": case["name"], "task": case["task"], "passed": passed,
            "duration_sec": prov.get("duration_sec"),
            "run_dir": prov.get("run_dir"), "detail": details,
            "_prov": prov}


def run_test(m: Manifest, device: str = "cuda") -> Dict[str, Any]:
    if not m.tests:
        return {"status": "untested", "detail": ["manifest has no tests"]}
    from verdi.core import install

    not_ready = install.is_ready(m)
    if not_ready:  # do not overwrite STATUS.toml with a setup problem
        return {"status": "untested",
                "detail": [f"{not_ready}: run `verdi setup {m.name}`"]}
    cases = [run_case(m, c, device) for c in m.tests]
    prov = cases[-1].pop("_prov")
    for c in cases:
        c.pop("_prov", None)
    untested = sorted(set(m.tasks) - {c["task"] for c in cases})
    status = {
        "status": "pass" if all(c["passed"] for c in cases) else "fail",
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": prov.get("gpu"),
        "verdi_commit": prov.get("verdi_commit"),
        "upstream_commit": prov.get("upstream_commit"),
        "env_lock": prov.get("env_lock"),
        "untested_tasks": untested,
        "cases": [
            f"{'PASS' if c['passed'] else 'FAIL'} {c['name']} "
            f"({c['duration_sec']}s): " + "; ".join(c["detail"])
            for c in cases],
    }
    write_status(m.node_dir / "STATUS.toml", status)
    status["case_runs"] = {c["name"]: c["run_dir"] for c in cases}
    return status


def _toml_value(v: Any) -> str:
    if v is None:
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[\n" + "".join(f"  {_toml_value(x)},\n" for x in v) + "]"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{s}"'


def write_status(path: Path, status: Dict[str, Any]) -> None:
    lines = ["# Written by `verdi test`. Do not edit by hand."]
    lines += [f"{k} = {_toml_value(v)}" for k, v in status.items()]
    path.write_text("\n".join(lines) + "\n")


def read_status(node_dir: Path) -> Dict[str, Any]:
    path = node_dir / "STATUS.toml"
    if not path.exists():
        return {}
    from verdi.core.manifest import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)
