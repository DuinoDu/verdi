"""Render a manifest as agent-readable markdown (``otn-cli describe``)."""
from __future__ import annotations

import shlex
from typing import Optional

from pdebug import types
from pdebug.core import install, testing
from pdebug.core.manifest import Manifest, Task


def _example(m: Manifest, t: Task) -> str:
    parts = ["otn-cli", "run", m.name]
    if len(m.tasks) > 1:
        parts += ["--task", t.name]
    for name, port in t.inputs.items():
        if not port.optional:
            parts += ["-i", f"{name}=<{port.type}>"]
    for name, p in t.params.items():
        if p.required:
            parts += ["-p", f"{name}=<{p.type}>"]
    return " ".join(shlex.quote(x) if " " in x else x for x in parts)


def describe_task(m: Manifest, t: Task) -> str:
    out = [f"### task `{t.name}`", "", t.description.strip(), ""]
    if t.inputs:
        out.append("inputs:")
        for n, p in t.inputs.items():
            opt = " (optional)" if p.optional else ""
            out.append(f"- `{n}`: {p.type}{opt}: {p.description} "
                       f"[{types.TYPES[p.type].format}]")
    out.append("outputs:")
    for n, p in t.outputs.items():
        out.append(f"- `{n}`: {p.type}: {p.description} "
                   f"[{types.TYPES[p.type].format}]")
    if t.params:
        out.append("params:")
        for n, p in t.params.items():
            dflt = "required" if p.required else f"default {p.default!r}"
            ch = f", choices {p.choices}" if p.choices else ""
            out.append(f"- `{n}` ({p.type}, {dflt}{ch}): {p.description}")
    out += ["", "```bash", _example(m, t), "```", ""]
    return "\n".join(out)


def describe(m: Manifest, task: Optional[str] = None) -> str:
    status = testing.read_status(m.node_dir)
    ready = install.is_ready(m)
    out = [f"## {m.name} (v{m.version})", "", m.description.strip(), ""]
    out.append(f"- setup: {'ready' if not ready else ready}"
               + ("" if not ready else f" -> `otn-cli setup {m.name}`"))
    if status:
        out.append(f"- last test: {status.get('status')} at "
                   f"{status.get('tested_at')} on {status.get('gpu')}")
        if status.get("untested_tasks"):
            out.append(f"- untested tasks: {status['untested_tasks']}")
    else:
        out.append("- last test: never (treat as unverified)")
    if m.upstream:
        out.append(f"- upstream: {m.upstream.get('repo')} @ "
                   f"{str(m.upstream.get('commit'))[:12]}")
    out.append("")
    tasks = [m.task(task)] if task else list(m.tasks.values())
    for t in tasks:
        out.append(describe_task(m, t))
    if m.known_issues:
        out.append("known issues:")
        for k in m.known_issues:
            out.append(f"- {k.get('summary')} -> {k.get('fix', '')}")
    notes = m.node_dir / "NOTES.md"
    if notes.exists():
        out.append(f"\nmore: {notes}")
    return "\n".join(out)
