"""verdi: list / describe / doctor / setup / run / test / types.

stdout carries machine-readable output (JSON for ``run``/``test``);
progress and model logs go to stderr.
"""
from __future__ import annotations

import json
import sys
from typing import List, Optional

import typer

from verdi import types
from verdi.core import describe as describe_mod
from verdi.core import doctor as doctor_mod
from verdi.core import install, manifest, runner, testing

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Run verified model inference nodes (verdi).")


def _err(msg: str) -> None:
    typer.echo(msg, err=True)


def _load(node: str) -> manifest.Manifest:
    try:
        return manifest.load(node)
    except (FileNotFoundError, manifest.ManifestError) as exc:
        _err(str(exc))
        raise typer.Exit(2)


def _kv(items: Optional[List[str]], what: str) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            _err(f"{what} must be key=value, got {item!r}")
            raise typer.Exit(2)
        k, v = item.split("=", 1)
        out[k] = v
    return out


@app.command("list")
def list_cmd(as_json: bool = typer.Option(False, "--json")):
    """List nodes with tasks, setup state and last test result."""
    rows = []
    for name in manifest.all_nodes():
        m = manifest.load(name)
        st = testing.read_status(m.node_dir)
        rows.append({
            "node": name, "tasks": sorted(m.tasks),
            "setup": install.is_ready(m) or "ready",
            "test": st.get("status", "never"),
            "tested_at": st.get("tested_at", ""),
            "description": m.description.strip().splitlines()[0],
        })
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        typer.echo(f"{r['node']:<22} test={r['test']:<6} "
                   f"setup={r['setup']:<12} tasks={','.join(r['tasks'])}"
                   f"\n    {r['description']}")


@app.command()
def describe(node: str, task: Optional[str] = typer.Option(None, "--task")):
    """Show tasks, typed inputs/outputs, params and an example command."""
    typer.echo(describe_mod.describe(_load(node), task))


@app.command("types")
def types_cmd():
    """List the unified data types and their on-disk formats."""
    typer.echo(types.describe_types())


@app.command()
def doctor(node: Optional[str] = typer.Argument(None),
           all_: bool = typer.Option(False, "--all")):
    """Check system requirements (driver, CUDA, compilers, libs, ...)."""
    names = manifest.all_nodes() if all_ or node is None else [node]
    bad = 0
    for name in names:
        m = _load(name)
        checks = doctor_mod.run_checks(m)
        bad += sum(not c.ok for c in checks)
        typer.echo(doctor_mod.format_checks(m, checks))
    raise typer.Exit(1 if bad else 0)


@app.command()
def setup(node: str, force: bool = typer.Option(False, "--force"),
          skip_weights: bool = typer.Option(False, "--skip-weights")):
    """doctor -> pinned upstream checkout -> uv sync -> weights."""
    m = _load(node)
    try:
        rec = install.setup(m, force=force, skip_weights=skip_weights)
    except install.SetupError as exc:
        _err(f"setup failed: {exc}")
        raise typer.Exit(1)
    typer.echo(json.dumps(rec, indent=2))


@app.command()
def run(
    node: str,
    task: Optional[str] = typer.Option(None, "--task", "-t"),
    inp: Optional[List[str]] = typer.Option(None, "--input", "-i",
                                            help="name=path"),
    param: Optional[List[str]] = typer.Option(None, "--param", "-p",
                                              help="name=value"),
    request: Optional[str] = typer.Option(
        None, "--request", help="json file {task, inputs, params}"),
    device: str = typer.Option("cuda", "--device"),
    out: Optional[str] = typer.Option(None, "--out",
                                      help="output directory"),
    timeout: Optional[int] = typer.Option(None, "--timeout"),
    quiet: bool = typer.Option(False, "--quiet", "-q",
                               help="do not stream model logs"),
):
    """Run one task; prints result JSON (outputs + provenance) to stdout."""
    m = _load(node)
    inputs, params = _kv(inp, "--input"), _kv(param, "--param")
    if request:
        with open(request) as f:
            req = json.load(f)
        task = task or req.get("task")
        inputs = {**{k: (v["path"] if isinstance(v, dict) else v)
                     for k, v in req.get("inputs", {}).items()}, **inputs}
        params = {**req.get("params", {}), **params}
    try:
        result = runner.run(m, task, inputs, params, device=device,
                            output_dir=out, timeout_sec=timeout,
                            stream_log=not quiet)
    except ValueError as exc:
        _err(str(exc))
        raise typer.Exit(2)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    raise typer.Exit(0 if result.get("status") == "ok" else 1)


@app.command()
def test(node: Optional[str] = typer.Argument(None),
         all_: bool = typer.Option(False, "--all"),
         device: str = typer.Option("cuda", "--device")):
    """Run the node fixture test on this machine and write STATUS.toml."""
    names = manifest.all_nodes() if all_ else [node] if node else []
    if not names:
        _err("give a node or --all")
        raise typer.Exit(2)
    failed = 0
    summary = {}
    for name in names:
        _err(f"=== test {name}")
        st = testing.run_test(_load(name), device=device)
        summary[name] = st
        failed += st["status"] != "pass"
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    raise typer.Exit(1 if failed else 0)


@app.command("offline-fetch")
def offline_fetch_cmd(
    nodes: Optional[List[str]] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all"),
    out: str = typer.Option(..., "--out", help="output directory"),
    skip_weights: bool = typer.Option(False, "--skip-weights"),
    hf: bool = typer.Option(False, "--hf", help="also fetch HF weights"),
):
    """On an online machine: mirror GitHub repos + url weights for offline
    hosts (see verdi/core/offline.py)."""
    from pathlib import Path

    from verdi.core import offline

    names = manifest.all_nodes() if all_ else (nodes or [])
    if not names:
        _err("give nodes or --all")
        raise typer.Exit(2)
    offline.offline_fetch([_load(n) for n in names], Path(out),
                          weights=not skip_weights, hf=hf)


def cli() -> None:
    app()


if __name__ == "__main__":
    sys.exit(cli())
