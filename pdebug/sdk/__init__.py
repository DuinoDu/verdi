"""Node-side SDK: imported by ``nodes/<node>/entry.py`` inside the node venv.

Usage::

    from pdebug.sdk import Context, main

    def segment_image(ctx: Context) -> None:
        image = ctx.input("image")              # Path
        out = ctx.output_path("mask", "mask.png")
        ...                                     # run the model, write out
        ctx.set_output("mask", out)

    if __name__ == "__main__":
        main({"segment_image": segment_image})

Rules: never fabricate outputs. If the model fails, raise; ``main``
turns the exception into ``status=error``.
"""
from pdebug.sdk.context import (  # noqa: F401
    Context,
    NodeError,
    main,
    prefetch_files,
)
