# Run by `verdi setup sam6d` through `blenderproc run`: installs
# blenderproc's own pip packages into Blender's python once, so the first
# real run does not need network access.
import blenderproc as bproc  # noqa: F401

bproc.init()
print("blenderproc warm-up ok")
