"""CPU state simulation of scripts/verdi_remote.py (no model, no network).

The relay and 063 are simulated by local temp dirs: ssh/rsync command strings
are rewritten to local commands; the remote `verdi run` is a fake script that
counts its invocations. Real run directories are never touched.
"""
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verdi_remote.py"


def _load():
    spec = importlib.util.spec_from_file_location("verdi_remote_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def sim(tmp_path, monkeypatch):
    vr = _load()
    remote_home = tmp_path / "home063"; remote_home.mkdir()
    relay_tmp = tmp_path / "relay_tmp"; relay_tmp.mkdir()
    (remote_home / "verdi_env.sh").write_text("")
    repo = tmp_path / "repo"; (repo / ".venv" / "bin").mkdir(parents=True)
    counter = tmp_path / "node_calls.txt"; counter.write_text("0")
    fake = repo / ".venv" / "bin" / "verdi"
    fake.write_text(f"""#!/usr/bin/env bash
# fake `verdi run NODE --task T ... --out DIR ...`: counts calls, writes one output
n=$(cat {counter}); echo $((n+1)) > {counter}
out=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
mkdir -p "$out"; echo payload > "$out/x.txt"
printf '{{"status": "ok", "outputs": {{"x": {{"type": "file", "path": "%s/x.txt"}}}}, "provenance": {{}}}}\\n' "$out"
""")
    fake.chmod(0o755)
    monkeypatch.setattr(vr, "REPO", str(repo))
    env = dict(os.environ, HOME=str(remote_home))
    state = {"fail_out_download": 0}

    def local(cmd):
        cmd = cmd.replace("/tmp/verdi_remote", str(relay_tmp))
        cmd = cmd.replace(f"-e {shlex.quote(vr.SSH063)} ", "").replace(f"{shlex.quote(vr.H063)}:", "")
        return cmd

    def fake_relay(cmd, retries=1):
        c = local(cmd)
        if state["fail_out_download"] and "rsync" in c and c.rstrip().split()[-2].endswith("/out"):
            raise vr.Transport(f"simulated rsync failure (code 23) for: {c}")
        r = subprocess.run(["bash", "-c", c], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise vr.Transport(f"local relay cmd failed: {c}\n{r.stderr}")
        return r.stdout

    def fake_on063(cmd, retries=1):
        r = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise vr.Transport(f"local 063 cmd failed:\n{r.stderr}")
        return r.stdout

    def fake_sh(cmd, retries=1):
        if isinstance(cmd, list):
            cmd = [a for a in cmd]
            if cmd[:1] == ["rsync"]:
                if "-e" in cmd:
                    i = cmd.index("-e"); del cmd[i:i + 2]
                cmd = [a.replace(f"{vr.RELAY}:", "").replace("/tmp/verdi_remote", str(relay_tmp)) for a in cmd]
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise vr.Transport(f"local cmd failed: {cmd}\n{r.stderr}")
        return r.stdout

    monkeypatch.setattr(vr, "relay", fake_relay)
    monkeypatch.setattr(vr, "on063", fake_on063)
    monkeypatch.setattr(vr, "sh", fake_sh)
    real_sleep = vr.time.sleep
    monkeypatch.setattr(vr.time, "sleep", lambda s: real_sleep(min(s, 0.2)))
    inp = tmp_path / "input.txt"; inp.write_text("hello")
    return vr, remote_home / "verdi_remote", counter, state, inp, tmp_path


def _tree_hash(p: Path):
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob("*")) if f.is_file()}


def test_normal_run_verified_download_then_own_dir_removed(sim, capsys):
    vr, root, counter, state, inp, tmp = sim
    rc = vr.main(["fake_node", "--task", "t", "-i", f"a={inp}", "--out", str(tmp / "o1"), "--device", "cpu"])
    res = json.loads(capsys.readouterr().out)
    assert rc == 0 and res["status"] == "ok"
    assert Path(res["outputs"]["x"]["path"]).read_text().strip() == "payload"
    assert counter.read_text().strip() == "1"
    assert list(root.iterdir()) == []            # own run dir removed only after verification


def test_failure_keeps_remote_and_resume_downloads_without_rerun(sim, capsys):
    vr, root, counter, state, inp, tmp = sim
    state["fail_out_download"] = 1
    rc = vr.main(["fake_node", "--task", "t", "-i", f"a={inp}", "--out", str(tmp / "o2"), "--device", "cpu"])
    err = json.loads(capsys.readouterr().out)["error"]
    assert rc == 2 and err["kind"] == "transport"
    rid = err["run_id"]
    assert "--resume " + rid in err["hint"]
    kept = root / rid
    assert (kept / "DONE").exists() and (kept / "out" / "x.txt").exists() and (kept / "result.json").exists()
    assert counter.read_text().strip() == "1"
    before = _tree_hash(kept)
    state["fail_out_download"] = 0
    rc = vr.main(["--resume", rid, "--out", str(tmp / "o2r")])
    res = json.loads(capsys.readouterr().out)
    assert rc == 0 and res["status"] == "ok"
    assert Path(res["outputs"]["x"]["path"]).read_text().strip() == "payload"
    assert counter.read_text().strip() == "1"     # resume did not run the node again
    assert before and not kept.exists()           # removed only after the verified resume


def test_delete_refuses_active_and_unmarked_and_wildcards(sim, capsys):
    vr, root, counter, state, inp, tmp = sim
    root.mkdir(exist_ok=True)
    act = root / "20991231-000000-aaaaaa"; act.mkdir(); (act / "ACTIVE").write_text(""); (act / "data.bin").write_bytes(b"x" * 1000)
    unm = root / "20991231-000001-bbbbbb"; unm.mkdir(); (unm / "result.json").write_text("{}")
    h_act, h_unm = _tree_hash(act), _tree_hash(unm)
    assert vr.main(["--delete", act.name]) == 2
    assert vr.main(["--delete", unm.name]) == 2
    with pytest.raises(SystemExit):
        vr.main(["--delete", "*"])
    with pytest.raises(SystemExit):
        vr.main(["--delete", "20991231-*"])
    assert _tree_hash(act) == h_act and _tree_hash(unm) == h_unm
    done = root / "20991231-000002-cccccc"; done.mkdir(); (done / "DONE").write_text("")
    assert vr.main(["--delete", done.name]) == 0 and not done.exists()
    assert act.exists() and unm.exists()          # other runs untouched
