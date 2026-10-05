"""Content identity of node inputs (recorded in result provenance).

Algorithm (also written into every record as `hash_algorithm`):
* file: content_sha256 = sha256 of the file bytes, streamed in 1 MiB chunks.
* directory: manifest = one line per regular file (symlinks followed), sorted by
  POSIX relative path: "<relpath>\\t<size>\\t<sha256(content)>\\n"; manifest_sha256 =
  sha256 of the UTF-8 manifest. Sidecars inside the directory (e.g. scale.json) are
  part of the manifest.
* file sidecar: "<file>.scale.json" next to a file input is hashed separately
  (sidecars: {name: content_sha256}).
* weak fingerprint: only if the caller sets VERDI_HASH_MAX_BYTES and the input is
  larger: sha256 over "<relpath>\\t<size>\\t<mtime_ns>" lines, hash_verified = false.
  A weak fingerprint is NOT a content check and must not be used as a cache hit.
This is a pure content identity; it differs from name+content signatures that a
caller may compute (e.g. real2sim's bridge), which are not produced here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

ALGORITHM = ("file: sha256(content), 1 MiB streaming; dir: sha256 of sorted lines "
             "'relpath\\tsize\\tsha256(content)\\n' over all regular files (sidecars "
             "included); file sidecar <file>.scale.json hashed separately; weak "
             "fingerprint (hash_verified=false) only above VERDI_HASH_MAX_BYTES")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path):
    out = []
    for dirpath, _, names in os.walk(root, followlinks=True):
        for n in names:
            p = Path(dirpath) / n
            if p.is_file():
                out.append(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def digest(path: str | Path, max_bytes: Optional[int] = None) -> Dict[str, Any]:
    p = Path(path)
    if max_bytes is None and os.environ.get("VERDI_HASH_MAX_BYTES"):
        max_bytes = int(os.environ["VERDI_HASH_MAX_BYTES"])
    rec: Dict[str, Any] = {"hash_algorithm": ALGORITHM}
    if p.is_file():
        size = p.stat().st_size
        rec.update(kind="file", size=size)
        if max_bytes is not None and size > max_bytes:
            st = p.stat()
            rec.update(weak_fingerprint=hashlib.sha256(
                f"{p.name}\t{size}\t{st.st_mtime_ns}\n".encode()).hexdigest(), hash_verified=False)
        else:
            rec.update(content_sha256=sha256_file(p), hash_verified=True)
        side = p.with_name(p.name + ".scale.json")
        if side.is_file():
            rec["sidecars"] = {side.name: sha256_file(side)}
        return rec
    if p.is_dir():
        files = _files(p)
        total = sum(f.stat().st_size for f in files)
        rec.update(kind="dir", files=len(files), size=total)
        if max_bytes is not None and total > max_bytes:
            lines = "".join(f"{f.relative_to(p).as_posix()}\t{f.stat().st_size}\t{f.stat().st_mtime_ns}\n" for f in files)
            rec.update(weak_fingerprint=hashlib.sha256(lines.encode()).hexdigest(), hash_verified=False)
        else:
            lines = "".join(f"{f.relative_to(p).as_posix()}\t{f.stat().st_size}\t{sha256_file(f)}\n" for f in files)
            rec.update(manifest_sha256=hashlib.sha256(lines.encode("utf-8")).hexdigest(), hash_verified=True)
        return rec
    return {"hash_algorithm": ALGORITHM, "kind": "missing", "hash_verified": False}
