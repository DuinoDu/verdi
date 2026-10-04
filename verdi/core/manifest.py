"""Parse and validate ``nodes/<node>/manifest.toml``."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from verdi.core import paths
from verdi.types import TYPES

PARAM_TYPES = {"str": str, "int": int, "float": float, "bool": bool,
               "list": list, "dict": dict}


class ManifestError(ValueError):
    """Raised when a manifest is malformed."""


@dataclass
class Port:
    """A typed task input or output."""

    name: str
    type: str
    optional: bool = False
    description: str = ""

    @classmethod
    def parse(cls, name: str, spec: Any, where: str) -> "Port":
        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, dict) or "type" not in spec:
            raise ManifestError(f"{where}.{name}: needs a type")
        if spec["type"] not in TYPES:
            raise ManifestError(
                f"{where}.{name}: unknown type {spec['type']!r}; "
                f"known: {sorted(TYPES)}"
            )
        return cls(name, spec["type"], bool(spec.get("optional", False)),
                   spec.get("description", ""))


@dataclass
class Param:
    """A scalar task parameter."""

    name: str
    type: str
    default: Any = None
    required: bool = False
    choices: Optional[List[Any]] = None
    description: str = ""

    @classmethod
    def parse(cls, name: str, spec: Dict[str, Any], where: str) -> "Param":
        ptype = spec.get("type", "str")
        if ptype not in PARAM_TYPES:
            raise ManifestError(f"{where}.{name}: bad param type {ptype!r}")
        return cls(name, ptype, spec.get("default"),
                   "default" not in spec, spec.get("choices"),
                   spec.get("description", ""))

    def coerce(self, value: Any) -> Any:
        """Convert CLI strings to the declared type and check choices."""
        if isinstance(value, str) and self.type != "str":
            if self.type == "bool":
                value = value.lower() in ("1", "true", "yes", "on")
            elif self.type in ("list", "dict"):
                import json

                value = json.loads(value)
            else:
                value = PARAM_TYPES[self.type](value)
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"param {self.name}={value!r} not in {self.choices}"
            )
        return value


@dataclass
class Task:
    name: str
    description: str
    inputs: Dict[str, Port]
    outputs: Dict[str, Port]
    params: Dict[str, Param]


@dataclass
class Manifest:
    """In-memory view of one node manifest."""

    name: str
    version: str
    description: str
    node_dir: Path
    upstream: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    build_env: Dict[str, str] = field(default_factory=dict)
    run_env: Dict[str, str] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)
    tests: List[Dict[str, Any]] = field(default_factory=list)
    known_issues: List[Dict[str, str]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def entry(self) -> Path:
        return self.node_dir / self.run.get("entry", "entry.py")

    @property
    def timeout_sec(self) -> int:
        return int(self.run.get("timeout_sec", 3600))

    def task(self, name: Optional[str]) -> Task:
        if name is None:
            if len(self.tasks) == 1:
                return next(iter(self.tasks.values()))
            raise ValueError(
                f"node {self.name} has several tasks, pass --task one of "
                f"{sorted(self.tasks)}"
            )
        if name not in self.tasks:
            raise ValueError(
                f"node {self.name} has no task {name!r}; "
                f"available: {sorted(self.tasks)}"
            )
        return self.tasks[name]


def load(node_or_path: str) -> Manifest:
    """Load a manifest by node name or by path to a node directory."""
    path = Path(node_or_path)
    if not path.exists():
        path = paths.NODES_DIR / node_or_path
    if path.is_dir():
        path = path / "manifest.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"no node {node_or_path!r}; run `verdi list` to see nodes"
        )
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse(raw, path.parent)


def parse(raw: Dict[str, Any], node_dir: Path) -> Manifest:
    for key in ("name", "version", "description", "tasks"):
        if key not in raw:
            raise ManifestError(f"{node_dir}/manifest.toml: missing {key}")
    if raw["name"] != node_dir.name:
        raise ManifestError(
            f"manifest name {raw['name']!r} != directory {node_dir.name!r}"
        )
    tasks = {}
    for tname, tspec in raw["tasks"].items():
        where = f"tasks.{tname}"
        tasks[tname] = Task(
            name=tname,
            description=tspec.get("description", ""),
            inputs={k: Port.parse(k, v, where + ".inputs")
                    for k, v in tspec.get("inputs", {}).items()},
            outputs={k: Port.parse(k, v, where + ".outputs")
                     for k, v in tspec.get("outputs", {}).items()},
            params={k: Param.parse(k, v, where + ".params")
                    for k, v in tspec.get("params", {}).items()},
        )
        if not tasks[tname].outputs:
            raise ManifestError(f"{where}: declares no outputs")
    build = raw.get("build", {})
    run = dict(raw.get("run", {}))
    run_env = run.pop("env", {})
    tests = list(raw.get("tests", []))
    if raw.get("test"):
        tests.insert(0, raw["test"])
    for i, t in enumerate(tests):
        if t.get("task") not in tasks:
            raise ManifestError(f"tests[{i}].task {t.get('task')!r} is not "
                                "a task")
        t.setdefault("name", t["task"])
    return Manifest(
        name=raw["name"],
        version=str(raw["version"]),
        description=raw["description"],
        node_dir=node_dir,
        upstream=raw.get("upstream", {}),
        weights=raw.get("weights", {}),
        system=raw.get("system", {}),
        build_env={k: str(v) for k, v in build.get("env", {}).items()},
        run_env={k: str(v) for k, v in run_env.items()},
        run=run,
        tasks=tasks,
        tests=tests,
        known_issues=raw.get("known_issues", []),
        raw=raw,
    )


def all_nodes() -> List[str]:
    """Names of every node directory with a manifest."""
    if not paths.NODES_DIR.exists():
        return []
    return sorted(
        p.name for p in paths.NODES_DIR.iterdir()
        if (p / "manifest.toml").exists()
    )
