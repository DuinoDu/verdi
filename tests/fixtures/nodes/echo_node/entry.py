from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def invert(ctx: Context) -> None:
    if ctx.param("fail"):
        raise NodeError("asked to fail", hint="set fail=false")
    rgb = io.read_image(ctx.input("image"))
    out = ctx.output_path("image", "inverted.png")
    io.write_image(out, 255 - rgb)
    ctx.set_output("image", out)


if __name__ == "__main__":
    raise SystemExit(main({"invert": invert}))
