"""shapemesh — a Python geometry and meshing library powered by Gmsh."""

from __future__ import annotations

from .errors import (
    BuildError,
    GeometryError,
    LabelError,
    MeshFormatError,
    ShapeMeshError,
)
from .settings import Algorithm2D, Algorithm3D, MeshSettings
from .shapes import (
    Cube,
    Cylinder,
    Difference,
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
    "BuildError",
    "Cube",
    "Cylinder",
    "Difference",
    "GeometryError",
    "Intersection",
    "LabelError",
    "Line",
    "MeshFormatError",
    "MeshSettings",
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
]
