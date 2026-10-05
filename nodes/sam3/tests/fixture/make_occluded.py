"""bedroom_occluded/: bedroom frames with child id 1 (left child) hidden
behind a flat grey rectangle on frames 2 and 3 (x 136..306, y 100..420), to
test that track_* report it as NOT visible there (and re-acquired after)."""
from pathlib import Path
import numpy as np
from PIL import Image
here = Path(__file__).parent
out = here / "bedroom_occluded"
out.mkdir(exist_ok=True)
for i in range(6):
    im = np.asarray(Image.open(here / "bedroom" / f"{i:06d}.jpg").convert("RGB")).copy()
    if i in (2, 3):
        im[100:420, 136:306] = 128
    Image.fromarray(im).save(out / f"{i:06d}.jpg", quality=95)
