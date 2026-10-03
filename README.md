# pdebug

`pdebug` is not for humans. It is for AI coding.

It packages existing models as **nodes** that an agent can run with one
command and trust: each node has its own uv venv, a pinned upstream
commit, pinned weights, declared system requirements and a fixture test
that has passed on a real GPU (`STATUS.toml`).

```bash
make env                                   # main env with otn-cli
.venv/bin/otn-cli list                     # nodes + last test status
.venv/bin/otn-cli describe sam2            # typed IO, params, example
.venv/bin/otn-cli setup sam2               # doctor + uv sync + weights
.venv/bin/otn-cli run sam2 --task segment_image \
    -i image=truck.jpg -i prompts=prompts.json
.venv/bin/otn-cli test sam2                # fixture test -> STATUS.toml
```

- Agent entry point: [`skills/pdebug/SKILL.md`](skills/pdebug/SKILL.md)
- Adding a node: [`skills/pdebug/references/add-node.md`](skills/pdebug/references/add-node.md)
- Pipelines that use nodes: [`pipelines/`](pipelines/)

The pre-refactor toolkit is tagged `legacy-v0.0.2`.
