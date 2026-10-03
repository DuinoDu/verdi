# threedgrut: notes

## Upstream use
- nv-tlabs/3dgrut @ a37ef72 (`[upstream]`, put on sys.path), SimFoundry
  `patches/3dgrut.patch` (NuRecExporter(export_cameras=False) for PLY-only
  export). Runs `threedgrut/export/scripts/ply_to_usd.py` like SimFoundry
  auto_bg step 7.
- Only the export path is installed: the 3DGUT CUDA tracer (slangc,
  tiny-cuda-nn), kaolin etc. are not built; entry.py replaces
  `threedgut_tracer.Tracer` with a no-op stub (export never renders).

## Test
- Fixture `tests/fixture/splat.ply`: 8000 random Gaussians (opacity > 0.5)
  of the splatfacto node's bedroom splat (8000 steps on the RTX 5090 host), SH degree 3.
- Checks: gaussian count / SH degree / Z-up / Volume prim / file size. The
  NuRec volume cannot be rendered without Omniverse, so no image check.

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.8.0 already; `sm_120` added; nvidia-ncore is on the aliyun mirror,
  so the explicit pypi.org index was dropped; re-locked.
