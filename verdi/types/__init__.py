"""Unified data types exchanged between nodes.

Every input/output is a file or directory on disk referenced from JSON as
``{"type": <name>, "path": <abs path>}``. Conventions (fixed, never per
node): RGB uint8 images, metres for every length, OpenCV camera frame
(x right, y down, z forward), poses are 4x4 ``T_cam_obj`` (object to
camera), boxes are pixel ``xyxy``, frames are named ``%06d.<ext>``.
"""
from verdi.types.registry import (  # noqa: F401
    TYPES,
    TypeSpec,
    compare,
    describe_types,
    validate,
)
