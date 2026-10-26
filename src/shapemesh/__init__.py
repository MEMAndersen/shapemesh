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
    ElementBlock,
    ElementType,
    EntityRecord,
    Mesh,
    MeshMetadata,
    NodeTable,
    NumpyElementBlock,
    NumpyMesh,
    PhysicalGroup,
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

__version__ = "0.1.0"

__all__ = [
    "Algorithm2D",
    "Algorithm3D",
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
    "NumpyElementBlock",
    "NumpyMesh",
    "PhysicalGroup",
    "Point",
    "Polygon",
    "Polyline",
    "Rectangle",
    "Shape",
    "ShapeMeshError",
    "Sphere",
    "Union",
    "Vec3",
    "Vec3Like",
    "build",
]
