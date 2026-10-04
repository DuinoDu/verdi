# verdi

**Verified, one-command model inference for AI agents.**

verdi packages existing vision / 3D models as **nodes** that an agent can
run with one command and trust: each node has its own uv venv, a pinned
upstream commit, pinned weights, declared system requirements, unified
typed inputs/outputs (metres, OpenCV camera frame) and a fixture test that
has passed on a real GPU (`STATUS.toml`). It is built for AI coding agents
first, humans second.

```bash
make env                                 # main env with the verdi CLI
.venv/bin/verdi list                     # nodes + last test status
.venv/bin/verdi describe sam2            # typed IO, params, example
.venv/bin/verdi setup sam2               # doctor + uv sync + weights
.venv/bin/verdi run sam2 --task segment_image \
    -i image=truck.jpg -i prompts=prompts.json
.venv/bin/verdi test sam2                # fixture test -> STATUS.toml
```

- Agent entry point: [`skills/verdi/SKILL.md`](skills/verdi/SKILL.md)
- Adding a node: [`skills/verdi/references/add-node.md`](skills/verdi/references/add-node.md)
- Pipelines that use nodes: [`pipelines/`](pipelines/)

## License

verdi itself (core, CLI, node wrappers, tests) is released under the
[MIT License](LICENSE).

Each node downloads and runs third-party code and model weights that keep
their **own licences**, several of which are research / non-commercial only
(for example FoundationStereo, MASt3R-SLAM, nvdiffrast, the SAM License,
Meta DINOv3, CC-BY-NC weights). Using a node means accepting the terms of its
upstream project; the restrictions are listed in each node's
`manifest.toml` (`[[known_issues]]`) and `NOTES.md`. Test fixtures under
`nodes/*/tests/` come from the respective upstream projects or public
datasets and follow their licences.
