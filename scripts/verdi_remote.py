#!/usr/bin/env python3
"""Run a verdi node on the 063 GPU host from apex (or any LAN host that can
ssh to the ubuntu relay), with input upload and output download.

    python3 scripts/verdi_remote.py <node> --task <task> -i name=PATH ... \
        -p key=value ... --out LOCAL_OUT_DIR [--device auto|cuda:N]
    python3 scripts/verdi_remote.py --resume RUN_ID --out LOCAL_OUT_DIR
    python3 scripts/verdi_remote.py --list            # remote runs + state
    python3 scripts/verdi_remote.py --delete RUN_ID   # delete ONE named, finished run of yours

Route: this host --(ssh/rsync, LAN)--> ubuntu relay --(ssh/rsync via the
jump host)--> 063. Nothing but ssh + rsync; no service to run. Inputs (files
or directories, plus their *.scale.json / scale.json sidecars) are copied
to 063 under ~/verdi_remote/<run_id>/in/, `verdi run` executes there, the
output directory, result.json, log.txt and stderr.txt come back into --out,
and the result JSON (stdout, same contract as `verdi run`) has its output
paths rewritten to local paths. Exit code 0 = status ok, 1 = node error
(error.kind, message, hint, log_tail as from `verdi run`), 2 = transport /
busy failure.

Isolation: every call uses its own run id (timestamp + random suffix):
~/verdi_remote/<run_id> on 063 and /tmp/verdi_remote/<run_id> on the
relay. A call only ever reads, writes or deletes paths of its own run id;
there is no shared / wildcard cleanup. A run dir carries ACTIVE while the
node runs and DONE when it finished. The remote dir is removed only after a
complete, verified download of that run (or by an explicit --delete RUN_ID,
which refuses ACTIVE runs). On any download failure it is kept and the exact --resume command
is printed, so no GPU work is repeated.

Configuration (environment variables, defaults = the current setup):
  VERDI_RELAY        ssh target of the relay        duino@10.103.75.12
  VERDI_RELAY_KEY    ssh key ON THE RELAY for 063   ~/code/scripts/envrc/ssh_keys/id_rsa_ubuntu
  VERDI_063          ssh target of 063 (jump host)  min01.du-labs@min01.du-labs@10.36.14.82@blj.horizon.cc
  VERDI_063_PORT     2222
  VERDI_063_REPO     ~/ws/pdebug   (verdi checkout on 063)
--host 063|017|018 (default 063) selects the GPU host; homes are separate per host,
so --resume / --list / --delete need the same --host as the run (run ids are per host).
An explicit VERDI_063 overrides --host.
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
HOSTS = {  # blj GPU hosts with the same account / key / layout (~/ws/pdebug, ~/verdi_env.sh)
    "063": "min01.du-labs@min01.du-labs@10.36.14.82@blj.horizon.cc",
    "017": "min01.du-labs@min01.du-labs@10.36.14.20@blj.horizon.cc",
    "018": "min01.du-labs@min01.du-labs@10.36.14.21@blj.horizon.cc",
}
HOST = "063"
H063 = os.environ.get("VERDI_063", HOSTS["063"])
PORT = os.environ.get("VERDI_063_PORT", "2222")
REPO = os.environ.get("VERDI_063_REPO", "~/ws/pdebug")
SSH_OPTS = "-o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SSH063 = f"ssh -p {PORT} -i {KEY} {SSH_OPTS}"
ROOT = "verdi_remote"  # under $HOME on 063


class Transport(RuntimeError):
    pass


def sh(cmd, retries: int = 1):
    last = None
    for k in range(retries):
        r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True,
                           stdin=subprocess.DEVNULL)
        if r.returncode == 0:
            return r.stdout
        last = r
        time.sleep(3 * (k + 1))
    raise Transport(f"command failed ({last.returncode}) after {retries} tries: "
                    f"{cmd if isinstance(cmd, str) else ' '.join(cmd)}\n--- stderr ---\n{last.stderr[-3000:]}")


def relay(cmd: str, retries: int = 1) -> str:
    return sh(["ssh", *SSH_OPTS.split(), RELAY, cmd], retries)


def on063(cmd: str, retries: int = 1) -> str:
    # the jump host keeps only the first line of a remote command: ship the
    # script base64-encoded on stdin to `bash -s` (newlines / quotes safe)
    import base64

    b64 = base64.b64encode(cmd.encode()).decode()
    return relay(f"echo {b64} | base64 -d | {SSH063} {shlex.quote(H063)} 'bash -s'", retries)


def pick_device(min_free_gb: float):
    rows = on063("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
                 "--format=csv,noheader,nounits", 3).strip().splitlines()
    gpus = []
    for r in rows:
        i, used, tot, util = (x.strip() for x in r.split(","))
        gpus.append({"index": int(i), "free_gb": (float(tot) - float(used)) / 1024, "util": float(util)})
    best = max(gpus, key=lambda g: (g["free_gb"], -g["util"]))
    return best, gpus


def download(rid: str, out: Path) -> dict:
    """Fetch result.json, log.txt, stderr.txt and out/ of a remote run, each
    item separately with retries; returns {item: ok|missing|error}."""
    stage = f"/tmp/verdi_remote/{rid}/back"
    rdir = f"~/{ROOT}/{rid}"
    state = on063(f"cd {rdir} 2>/dev/null && ls -1A || echo __NO_DIR__", 3).split()
    if "__NO_DIR__" in state:
        raise Transport(f"remote run dir {rdir} does not exist on 063 (deleted?)")
    if "ACTIVE" in state and "DONE" not in state:
        raise Transport(f"remote run {rid} is still ACTIVE; wait and --host {HOST} --resume {rid}")
    relay(f"mkdir -p {stage}", 3)
    report = {}
    for item in ("result.json", "log.txt", "stderr.txt", "out"):
        if item not in state:
            report[item] = "missing"
            continue
        src = f"{shlex.quote(H063)}:{rdir}/{item}"
        relay(f"rsync -a --partial -e {shlex.quote(SSH063)} {src} {stage}/", 4)
        report[item] = "ok"
    out.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-a", "--partial", "-e", f"ssh {SSH_OPTS}", f"{RELAY}:{stage}/", str(out) + "/"], 4)
    # verify: every output path listed in result.json exists locally
    res_file = out / "result.json"
    if res_file.exists() and res_file.read_text().strip():
        res = json.loads(res_file.read_text())
        missing = []
        for ref in res.get("outputs", {}).values():
            pth = ref.get("path", "")
            key = f"/{ROOT}/{rid}/out"
            if key in pth and not (out / "out" / pth.split(key, 1)[1].lstrip("/")).exists():
                missing.append(pth)
        if missing:
            raise Transport(f"downloaded result lists outputs that did not arrive: {missing[:3]}")
    relay(f"rm -rf /tmp/verdi_remote/{rid}")
    return report


def finish(rid: str, out: Path, device, gpu_info, keep: bool) -> int:
    res_file = out / "result.json"
    text = res_file.read_text() if res_file.exists() else ""
    if not text.strip():
        err = (out / "stderr.txt").read_text()[-3000:] if (out / "stderr.txt").exists() else ""
        print(json.dumps({"status": "error", "error": {
            "kind": "transport", "message": "remote run produced no result.json",
            "hint": f"remote stderr tail: {err}" if err else f"see {HOST} ~/{ROOT}/{rid}/stderr.txt",
            "run_id": rid}}, indent=2))
        return 2
    res = json.loads(text)
    key = f"/{ROOT}/{rid}/out"
    for ref in res.get("outputs", {}).values():
        pth = ref.get("path", "")
        if key in pth:
            ref["path"] = str(out / "out") + pth.split(key, 1)[1]
    res.setdefault("provenance", {})["remote"] = {
        "host": "063", "run_id": rid, "remote_dir": f"~/{ROOT}/{rid}",
        "local_log": str(out / "log.txt"), "device": device, "gpu_check": gpu_info}
    res_file.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not keep:
        on063(f"rm -rf ~/{ROOT}/{rid}")  # only after a complete, verified download
    return 0 if res.get("status") == "ok" else 1


def transport_error(exc: Exception, rid: str, out: Path) -> int:
    print(json.dumps({"status": "error", "error": {
        "kind": "transport", "message": str(exc)[-4000:], "run_id": rid,
        "hint": f"nothing was deleted on {HOST} (~/{ROOT}/{rid}); recover without rerunning: "
                f"python3 scripts/verdi_remote.py --host {HOST} --resume {rid} --out {out}"}}, indent=2))
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("node", nargs="?")
    ap.add_argument("--task")
    ap.add_argument("-i", "--input", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", help="local output directory")
    ap.add_argument("--device", default="auto",
                    help="cuda:N, cpu, or auto = the 063 GPU with the most free memory")
    ap.add_argument("--min-free-gb", type=float, default=16.0,
                    help="with --device auto: refuse if no GPU has this much free memory")
    ap.add_argument("--timeout", type=int, default=0, help="seconds (0 = node default)")
    ap.add_argument("--keep-remote", action="store_true", help="keep the run dir on 063")
    ap.add_argument("--resume", metavar="RUN_ID", help="download an existing remote run (no rerun)")
    ap.add_argument("--list", action="store_true", help="list remote runs and their state")
    ap.add_argument("--delete", metavar="RUN_ID",
                    help="delete exactly this finished remote run (refuses ACTIVE)")
    ap.add_argument("--host", default="063", choices=sorted(HOSTS),
                    help="GPU host (same layout on each); run ids are per host")
    a = ap.parse_args(argv)
    global H063, HOST
    HOST = a.host
    if "VERDI_063" not in os.environ:
        H063 = HOSTS[a.host]

    if a.list:
        cmd = (f"cd ~/{ROOT} 2>/dev/null || exit 0; shopt -s nullglob; for d in */; do d=${{d%/}}; "
               f"s=unmarked; [ -f $d/ACTIVE ] && s=active; [ -f $d/DONE ] && s=done; "
               f"age=$(( ($(date +%s) - $(stat -c %Y $d)) / 60 )); echo \"$d $s ${{age}}min\"; done")
        rows = [r for r in on063(cmd, 3).split("\n") if r.strip()]
        print("\n".join(rows) or "(no remote runs)")
        return 0
    if a.delete:
        import re

        if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", a.delete):
            ap.error("--delete takes one exact run id (YYYYMMDD-HHMMSS-xxxxxx)")
        st = on063(f"cd ~/{ROOT}/{a.delete} 2>/dev/null && ls -A || echo __NO_DIR__", 3).split()
        if "__NO_DIR__" in st:
            print(f"run {a.delete} does not exist"); return 2
        if "ACTIVE" in st or "DONE" not in st:
            print(f"refused: run {a.delete} is not marked DONE (active or made by an old runner)"); return 2
        on063(f"rm -rf ~/{ROOT}/{a.delete}")
        print(f"deleted ~/{ROOT}/{a.delete}")
        return 0
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out).expanduser().resolve()
    if a.resume:
        try:
            download(a.resume, out)
        except Transport as exc:
            return transport_error(exc, a.resume, out)
        return finish(a.resume, out, None, None, a.keep_remote)
    if not (a.node and a.task):
        ap.error("node and --task are required (or --resume / --list / --delete)")

    rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    stage = f"/tmp/verdi_remote/{rid}"
    rdir = f"~/{ROOT}/{rid}"
    out.mkdir(parents=True, exist_ok=True)
    gpu_info = None
    try:
        if a.device == "auto":
            best, gpus = pick_device(a.min_free_gb)
            gpu_info = {"chosen": best, "all": gpus}
            if best["free_gb"] < a.min_free_gb:
                print(json.dumps({"status": "error", "error": {
                    "kind": "busy", "message": f"no 063 GPU has {a.min_free_gb} GB free "
                    f"(best: cuda:{best['index']} {best['free_gb']:.1f} GB)", "gpus": gpus}}, indent=2))
                return 2
            a.device = f"cuda:{best['index']}"
            print(f"[verdi_remote] run {rid} on 063 {a.device} ({best['free_gb']:.1f} GB free)", file=sys.stderr)
        # ---- upload inputs (+ scale sidecars) to relay, then to 063
        relay(f"mkdir -p {stage}/in", 3)
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
            sh(["rsync", "-a", "-e", f"ssh {SSH_OPTS}", *files, f"{RELAY}:{stage}/in/{name}/"], 3)
            remote_inputs.append(f"-i {name}={rdir}/in/{name}/{p.name}")
        on063(f"mkdir -p {rdir} && touch {rdir}/ACTIVE", 3)
        relay(f"rsync -a -e {shlex.quote(SSH063)} {stage}/in {shlex.quote(H063)}:{rdir}/", 3)
        relay(f"rm -rf {stage}")
        # ---- run (ACTIVE while running, DONE afterwards, whatever the status)
        params = " ".join(f"-p {shlex.quote(x)}" for x in a.param)
        tmo = f"--timeout {a.timeout}" if a.timeout else ""
        cmd = (f"source ~/verdi_env.sh >/dev/null 2>&1; cd {REPO} && "
               f".venv/bin/verdi run {a.node} --task {a.task} {' '.join(remote_inputs)} "
               f"{params} --out {rdir}/out --device {a.device} {tmo} -q "
               f"> {rdir}/result.json 2> {rdir}/stderr.txt; "
               f"rd=$(python3 -c \"import json;print(json.load(open('$HOME/{ROOT}/{rid}/result.json'))"
               f".get('provenance',{{}}).get('run_dir',''))\" 2>/dev/null); "
               f"[ -n \"$rd\" ] && cp $rd/log.txt {rdir}/log.txt; touch {rdir}/DONE; rm -f {rdir}/ACTIVE; true")
        # detached on 063: a dropped ssh connection does not kill the job
        on063(f"cat > {rdir}/run.sh <<'VERDI_EOF'\n{cmd}\nVERDI_EOF\n"
              f"setsid nohup bash {rdir}/run.sh < /dev/null > {rdir}/runner.txt 2>&1 &", 3)
        t0 = time.time()
        limit = (a.timeout or 6 * 3600) + 600
        while True:
            time.sleep(10)
            try:
                if on063(f"test -f {rdir}/DONE && echo done || echo wait", 2).strip() == "done":
                    break
            except Transport:
                pass  # transient; keep polling
            if time.time() - t0 > limit:
                raise Transport(f"remote run {rid} not DONE after {limit} s (still running on 063?)")
    except Transport as exc:
        return transport_error(exc, rid, out)
    # ---- download (separate items, retries, verified); keep remote on failure
    try:
        download(rid, out)
    except Transport as exc:
        return transport_error(exc, rid, out)
    return finish(rid, out, a.device, gpu_info, a.keep_remote)


if __name__ == "__main__":
    sys.exit(main())
