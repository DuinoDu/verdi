# SimFoundry with verdi nodes

SimFoundry (NVlabs, https://github.com/NVlabs/SimFoundry) is used **unmodified**
as a git submodule in `third_party/SimFoundry`. It keeps its own orchestration
and conda environments; verdi does not reimplement it.

What verdi provides: every learned model SimFoundry calls is also available as
a verified, one-command node (same upstream commit and patches as SimFoundry's
install scripts where possible), so each stage can be run, inspected and
debugged on its own with unified inputs/outputs.

| SimFoundry stage / env | verdi node(s) |
|---|---|
| depth (`da3`, depth backends) | `depth_anything_3`, `depth_pro`, `prior_depth_anything`, `foundation_stereo` |
| segmentation (SAM 3, Grounded SAM) | `sam3`, `langsam`, `groundingdino`, `sam2` |
| mesh generation (`hunyuan`, `trellis2`) | `hunyuan3d`, `trellis2` |
| pose (`any6d`, FoundationPose) | `any6d`, `foundationpose` |
| auto-background (`void`, `nerfstudio_simfoundry`, `3dgrut`) | `void`, `splatfacto`, `threedgrut` |
| articulation (stage 9 part segmentation) | `p3sam`, `partfield` |

Not nodes (pipeline logic or API calls): VLM/GPT prompting, digital-cousin
matching, OmniGibson scene compilation, physics packaging.

## Running SimFoundry itself

```bash
git submodule update --init third_party/SimFoundry   # via the local git mirror on offline hosts
cd third_party/SimFoundry && cat docs/INSTALL.md       # official conda-based install
```

On an offline host, its install scripts need the same treatment as our
nodes (PyPI/HF mirrors, GitHub mirror, prefetched checkpoints); start from the
node NOTES.md files, which record every pitfall met for the same models.
