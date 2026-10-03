"""3DGRUT node: ply_to_usdz (3DGS ply -> NuRec USDZ), as SimFoundry step 7."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


class _NoRenderTracer:
    """Stand-in for threedgut_tracer.Tracer.

    MixtureOfGaussians.__init__ JIT-builds the 3DGUT CUDA rasteriser (slangc,
    tiny-cuda-nn headers) although PLY -> USDZ export never renders.
    """

    def __init__(self, conf):
        self.conf = conf

    def build_acc(self, gaussians, rebuild=True):
        pass

    def render(self, *a, **kw):
        raise NodeError("3DGUT rendering is not built in this node")


def ply_to_usdz(ctx: Context) -> None:
    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    src = Path(ctx.input("gaussians"))
    if src.suffix.lower() != ".ply":
        raise NodeError(f"{src} is not a .ply file")

    import threedgut_tracer
    import threedgrut.model.model as mog

    threedgut_tracer.Tracer = _NoRenderTracer
    mog.threedgut_tracer.Tracer = _NoRenderTracer
    from threedgrut.export.scripts import ply_to_usd

    # capture the model the upstream script builds, for the info output
    built = {}
    orig_init = mog.MixtureOfGaussians.init_from_ply

    def _init_from_ply(self, *a, **kw):
        out = orig_init(self, *a, **kw)
        built["model"] = self
        return out

    mog.MixtureOfGaussians.init_from_ply = _init_from_ply
    out = ctx.output_path("usdz", "gaussians.usdz")
    argv = sys.argv
    sys.argv = ["ply_to_usd.py", str(src), "--output_file", str(out)]
    try:
        ply_to_usd.main()
    except SystemExit as exc:
        raise NodeError(f"ply_to_usd failed (exit {exc.code})",
                        hint="see log.txt: is the ply in INRIA 3DGS layout "
                        "(f_dc_*, opacity, scale_*, rot_*)?") from None
    finally:
        sys.argv = argv
    if not out.exists() or out.stat().st_size == 0:
        raise NodeError("ply_to_usd wrote no usdz")
    ctx.set_output("usdz", out)

    from pxr import Usd, UsdGeom

    with zipfile.ZipFile(out) as z:
        members = z.namelist()
    usda = next((m for m in members if m.endswith((".usda", ".usdc",
                                                   ".usd"))), None)
    stage = Usd.Stage.Open(str(out))
    if stage is None:
        raise NodeError("written usdz cannot be opened by USD")
    prims = [{"path": str(p.GetPath()), "type": str(p.GetTypeName())}
             for p in stage.Traverse()]
    model = built.get("model")
    info = {
        "num_gaussians": int(model.num_gaussians) if model is not None
        else None,
        "sh_degree": int(model.max_n_features) if model is not None
        else None,
        "usdz_members": members,
        "root_layer": usda,
        "prims": prims,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "bytes": out.stat().st_size,
    }
    ipath = ctx.output_path("info", "info.json")
    io.write_json(ipath, info)
    ctx.set_output("info", ipath)


if __name__ == "__main__":
    raise SystemExit(main({"ply_to_usdz": ply_to_usdz}))
