# Copyright (C) 2026 Mads Kruse
#
# This file is part of shapemesh.
#
# shapemesh is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# shapemesh is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with shapemesh. If not, see <https://www.gnu.org/licenses/>.

"""Exception hierarchy for shapemesh.

Every structured library error derives from `ShapeMeshError`. The
geometry/label/format errors additionally subclass the builtin ``ValueError``
(and `BuildError` subclasses ``RuntimeError``) so callers can catch them
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
