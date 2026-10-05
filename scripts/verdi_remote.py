#!/usr/bin/env python3
"""Run a verdi node on the 063 GPU host from apex (or any LAN host that can
ssh to the ubuntu relay), with input upload and output download.

    python3 scripts/verdi_remote.py <node> --task <task> -i name=PATH ... \
        -p key=value ... --out LOCAL_OUT_DIR [--device cuda:N]

Route: this host --(ssh/rsync, LAN)--> ubuntu relay --(ssh/rsync via the
jump host)--> 063. Nothing but ssh + rsync; no service to run. Inputs (files
or directories, plus their *.scale.json / scale.json sidecars) are copied
to 063 under ~/verdi_remote/<run_id>/in/, `verdi run` executes there, the
output directory and the remote log come back into --out, and the result
JSON (stdout, same contract as `verdi run`) has its output paths rewritten
to local paths. Exit code 0 = status ok, 1 = node error (error.kind,
message, hint, log_tail as from `verdi run`), 2 = transport failure.

Configuration (environment variables, defaults = the current setup):
  VERDI_RELAY        ssh target of the relay        duino@10.103.75.12
  VERDI_RELAY_KEY    ssh key ON THE RELAY for 063   ~/code/scripts/envrc/ssh_keys/id_rsa_ubuntu
  VERDI_063          ssh target of 063 (jump host)  min01.du-labs@min01.du-labs@10.36.14.82@blj.horizon.cc
  VERDI_063_PORT     2222
  VERDI_063_REPO     ~/ws/pdebug   (verdi checkout on 063)
063 is shared: --device auto (default) picks the GPU with the most free memory
and refuses (exit 2, kind busy) below --min-free-gb.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

RELAY = os.environ.get("VERDI_RELAY", "duino@10.103.75.12")
KEY = os.environ.get("VERDI_RELAY_KEY", "~/code/scripts/envrc/ssh_keys/id_rsa_ubuntu")
H063 = os.environ.get("VERDI_063", "min01.du-labs@min01.du-labs@10.36.14.82@blj.horizon.cc")
PORT = os.environ.get("VERDI_063_PORT", "2222")
REPO = os.environ.get("VERDI_063_REPO", "~/ws/pdebug")
SSH_OPTS = "-o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SSH063 = f"ssh -p {PORT} -i {KEY} {SSH_OPTS}"


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True,
                       capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def relay(cmd: str) -> str:
    return sh(["ssh", *SSH_OPTS.split(), RELAY, cmd])


def on063(cmd: str) -> str:
    return relay(f"{SSH063} {shlex.quote(H063)} {shlex.quote(cmd)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("node")
    ap.add_argument("--task", required=True)
    ap.add_argument("-i", "--input", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", required=True, help="local output directory")
    ap.add_argument("--device", default="auto",
                    help="cuda:N, cpu, or auto = the 063 GPU with the most free memory")
    ap.add_argument("--min-free-gb", type=float, default=16.0,
                    help="with --device auto: refuse if no GPU has this much free memory")
    ap.add_argument("--timeout", type=int, default=0, help="seconds (0 = node default)")
    ap.add_argument("--keep-remote", action="store_true", help="keep files on 063 / relay")
    a = ap.parse_args(argv)

    rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    stage = f"/tmp/verdi_remote/{rid}"
    rdir = f"~/verdi_remote/{rid}"
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu_info = None
    try:
        if a.device == "auto":
            rows = on063("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
                         "--format=csv,noheader,nounits").strip().splitlines()
            gpus = []
            for r in rows:
                i, used, tot, util = (x.strip() for x in r.split(","))
                gpus.append({"index": int(i), "free_gb": (float(tot) - float(used)) / 1024,
                             "util": float(util)})
            best = max(gpus, key=lambda g: (g["free_gb"], -g["util"]))
            gpu_info = {"chosen": best, "all": gpus}
            if best["free_gb"] < a.min_free_gb:
                print(json.dumps({"status": "error", "error": {
                    "kind": "busy", "message": f"no 063 GPU has {a.min_free_gb} GB free "
                    f"(best: cuda:{best['index']} {best['free_gb']:.1f} GB)", "gpus": gpus}}, indent=2))
                return 2
            a.device = f"cuda:{best['index']}"
            print(f"[verdi_remote] 063 device {a.device} ({best['free_gb']:.1f} GB free)", file=sys.stderr)
        # ---- upload inputs (+ scale sidecars) to relay, then to 063
        relay(f"mkdir -p {stage}/in")
        remote_inputs = []
        for spec in a.input:
            name, _, path = spec.partition("=")
            p = Path(path).expanduser().resolve()
            if not p.exists():
                print(json.dumps({"status": "error", "error": {
                    "kind": "request", "message": f"input {name}: {p} not found"}}))
                return 1
            files = [str(p)]
            side = p.with_name(p.name + ".scale.json")
            if p.is_file() and side.exists():
                files.append(str(side))
            sh(["rsync", "-a", "-e", f"ssh {SSH_OPTS}", *files, f"{RELAY}:{stage}/in/{name}/"])
            remote_inputs.append(f"-i {name}={rdir}/in/{name}/{p.name}")
        on063(f"mkdir -p {rdir}")
        relay(f"rsync -a -e {shlex.quote(SSH063)} {stage}/in {shlex.quote(H063)}:{rdir}/")
        # ---- run
        params = " ".join(f"-p {shlex.quote(x)}" for x in a.param)
        tmo = f"--timeout {a.timeout}" if a.timeout else ""
        cmd = (f"source ~/verdi_env.sh >/dev/null 2>&1; cd {REPO} && "
               f".venv/bin/verdi run {a.node} --task {a.task} {' '.join(remote_inputs)} "
               f"{params} --out {rdir}/out --device {a.device} {tmo} -q "
               f"> {rdir}/result.json 2> {rdir}/stderr.txt; "
               f"rd=$(python3 -c \"import json;print(json.load(open('$HOME/verdi_remote/{rid}/result.json'))"
               f".get('provenance',{{}}).get('run_dir',''))\" 2>/dev/null); "
               f"[ -n \"$rd\" ] && cp $rd/log.txt {rdir}/log.txt; true")
        on063(cmd)
        # ---- download outputs + result + log
        relay(f"mkdir -p {stage}/back && rsync -a -e {shlex.quote(SSH063)} "
              f"{shlex.quote(H063)}:{rdir}/out {shlex.quote(H063)}:{rdir}/result.json "
              f"{shlex.quote(H063)}:{rdir}/log.txt {stage}/back/ 2>/dev/null || "
              f"rsync -a -e {shlex.quote(SSH063)} {shlex.quote(H063)}:{rdir}/result.json {stage}/back/")
        sh(["rsync", "-a", "-e", f"ssh {SSH_OPTS}", f"{RELAY}:{stage}/back/", str(out) + "/"])
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "error": {
            "kind": "transport", "message": str(exc)[-3000:],
            "hint": "check ssh to the relay / 063 and free disk; nothing ran or results stayed on 063 "
                    f"under ~/verdi_remote/{rid}"}}, indent=2))
        return 2
    finally:
        if not a.keep_remote:
            try:
                relay(f"rm -rf {stage}")
            except RuntimeError:
                pass
    res_file = out / "result.json"
    text = res_file.read_text() if res_file.exists() else ""
    if not text.strip():
        print(json.dumps({"status": "error", "error": {
            "kind": "transport", "message": "no result.json came back",
            "hint": f"see 063 ~/verdi_remote/{rid}/stderr.txt"}}, indent=2))
        return 2
    res = json.loads(text)
    remote_out_prefix = None
    for ref in res.get("outputs", {}).values():
        pth = ref.get("path", "")
        if "/verdi_remote/" in pth and "/out" in pth:
            remote_out_prefix = pth[:pth.index(f"/verdi_remote/{rid}/out") + len(f"/verdi_remote/{rid}/out")]
            ref["path"] = str(out / "out") + pth[len(remote_out_prefix):]
    res.setdefault("provenance", {})["remote"] = {
        "host": "063", "run_id": rid, "remote_dir": f"~/verdi_remote/{rid}",
        "local_log": str(out / "log.txt"), "device": a.device, "gpu_check": gpu_info}
    res_file.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not a.keep_remote:
        try:
            on063(f"rm -rf {rdir}")
        except RuntimeError:
            pass
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
