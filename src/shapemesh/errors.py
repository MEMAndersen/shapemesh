"""Exception hierarchy for shapemesh.

Every structured library error derives from :class:`ShapeMeshError`. The
geometry/label/format errors additionally subclass the builtin ``ValueError``
(and :class:`BuildError` subclasses ``RuntimeError``) so callers can catch them
with the usual builtins while still having a single library-wide base.
"""

from __future__ import annotations

__all__ = [
    "BuildError",
    "GeometryError",
    "LabelError",
    "MeshFormatError",
    "ShapeMeshError",
]


class ShapeMeshError(Exception):
    """Base class for all shapemesh errors."""


class GeometryError(ShapeMeshError, ValueError):
    """Invalid shape or geometry, e.g. an empty boolean result."""


class LabelError(ShapeMeshError, ValueError):
    """Invalid or conflicting physical-group labels."""


class MeshFormatError(ShapeMeshError, ValueError):
    """Corrupt or unsupported MSH file."""


class BuildError(ShapeMeshError, RuntimeError):
    """A Gmsh call failed during a build.

    Attributes:
        log: The last Gmsh log text, captured before Gmsh was finalized.
    """

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log
