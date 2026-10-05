"""mustard/mesh_x1.5.glb: the mustard mesh scaled by 1.5 (a wrong-size
model) - FoundationPose still returns a pose, the diagnostics must flag it."""
from pathlib import Path
import trimesh
here = Path(__file__).parent / "mustard"
s = trimesh.load(str(here / "mesh.glb"))
s.apply_scale(1.5)
s.export(str(here / "mesh_x1.5.glb"))
print(s.extents)
