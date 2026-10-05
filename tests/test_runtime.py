"""End-to-end: setup (uv venv) -> run -> test for a CPU-only node."""
import shutil

import numpy as np
import pytest

from verdi.core import install, manifest, runner, testing
from verdi.types import io

FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures/nodes"


@pytest.fixture()
def node(tmp_path, monkeypatch):
    monkeypatch.setenv("VERDI_HOME", str(tmp_path / "home"))
    node_dir = FIXTURE / "echo_node"
    io.write_image(node_dir / "tests" / "in.png",
                   np.zeros((4, 6, 3), np.uint8))
    m = manifest.load(str(node_dir))
    install.setup(m, force=True)
    yield m
    (node_dir / "STATUS.toml").unlink(missing_ok=True)
    shutil.rmtree(node_dir / ".venv", ignore_errors=True)


def test_manifest_parse():
    m = manifest.load(str(FIXTURE / "echo_node"))
    assert m.task(None).name == "invert"
    with pytest.raises(ValueError):
        m.task("nope")


def test_run_ok_error_and_contract(node, tmp_path):
    img = io.write_image(tmp_path / "x.png", np.zeros((4, 6, 3), np.uint8))
    res = runner.run(node, None, {"image": str(img)}, {}, device="cpu",
                     stream_log=False)
    assert res["status"] == "ok", res
    out = io.read_image(res["outputs"]["image"]["path"])
    assert out.min() == 255
    assert res["provenance"]["node"] == "echo_node"

    res = runner.run(node, None, {"image": str(img)}, {"fail": "true"},
                     device="cpu", stream_log=False)
    assert res["status"] == "error" and res["error"]["hint"]

    res = runner.run(node, None, {}, {}, device="cpu", stream_log=False)
    assert res["error"]["kind"] == "request"


def test_fixture_test(node):
    st = testing.run_test(node, device="cpu")
    assert st["status"] == "pass", st
    assert testing.read_status(node.node_dir)["status"] == "pass"


def test_expect_error_case(node):
    base = {"name": "neg", "task": "invert",
            "inputs": {"image": "tests/in.png"}}
    ok = testing.run_case(node, {**base, "params": {"fail": True},
                                 "expect_error": "asked to fail"}, "cpu")
    assert ok["passed"], ok
    wrong = testing.run_case(node, {**base, "params": {"fail": True},
                                    "expect_error": "other text"}, "cpu")
    assert not wrong["passed"]
    no_fail = testing.run_case(node, {**base, "expect_error": "asked"}, "cpu")
    assert not no_fail["passed"]
