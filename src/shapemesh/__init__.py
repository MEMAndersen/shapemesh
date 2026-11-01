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

"""shapemesh — a Python geometry and meshing library powered by Gmsh."""

from __future__ import annotations

from .builder import build
from .errors import (
    BuildError,
    GeometryError,
    LabelError,
    MeshFormatError,
    ShapeMeshError,
)
from .mesh import (
    BlockSlice,
    ElementBlock,
    ElementType,
    EntityRecord,
    Mesh,
    MeshMetadata,
    NodeTable,
    PhysicalGroup,
    TypeBlockView,
)
from .settings import Algorithm2D, Algorithm3D, MeshSettings
from .shapes import (
    Box,
    Circle,
    Cylinder,
    Difference,
    Disk,
    Ellipse,
    EllipticalDisk,
    Intersection,
    Line,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    Shape,
    Sphere,
    Union,
    Vec3,
    Vec3Like,
)

# Hatchling reads this value; keep it as the single source of truth.
__version__ = "0.2.0"

__all__ = [
    "Algorithm2D",
    "Algorithm3D",
    "BlockSlice",
    "Box",
    "BuildError",
    "Circle",
    "Cylinder",
    "Difference",
    "Disk",
    "ElementBlock",
    "ElementType",
    "Ellipse",
    "EllipticalDisk",
    "EntityRecord",
    "GeometryError",
    "Intersection",
    "LabelError",
    "Line",
    "Mesh",
    "MeshFormatError",
    "MeshMetadata",
    "MeshSettings",
    "NodeTable",
    "PhysicalGroup",
    "Point",
    "Polygon",
    "Polyline",
    "Rectangle",
    "Shape",
    "ShapeMeshError",
    "Sphere",
    "TypeBlockView",
    "Union",
    "Vec3",
    "Vec3Like",
    "build",
]
