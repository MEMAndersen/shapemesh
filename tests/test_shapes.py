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

import warnings

import numpy as np
import pytest

from shapemesh import (
    Box,
    BuildError,
    Circle,
    Cylinder,
    Difference,
    Disk,
    Ellipse,
    EllipticalDisk,
    GeometryError,
    Intersection,
    LabelError,
    Line,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    Shape,
    ShapeMeshError,
    Sphere,
    Union,
    Vec3,
)

# --- Vec3 -----------------------------------------------------------------


def test_vec3_construction_and_z_padding():
    v = Vec3(1, 2)
    assert (v.x, v.y, v.z) == (1.0, 2.0, 0.0)


def test_vec3_rejects_non_finite():
    with pytest.raises(GeometryError):
        Vec3(float("nan"), 0.0, 0.0)
    with pytest.raises(GeometryError):
        Vec3(0.0, float("inf"), 0.0)


def test_vec3_rejects_non_numbers():
    with pytest.raises(GeometryError):
        Vec3("1", 2, 3)
    with pytest.raises(GeometryError):
        Vec3(True, 2, 3)


def test_vec3_from_any_variants():
    assert Vec3.from_any(Vec3(1, 2, 3)) == Vec3(1, 2, 3)
    assert Vec3.from_any((1, 2, 3)) == Vec3(1, 2, 3)
    assert Vec3.from_any([1, 2]) == Vec3(1, 2, 0)
    assert Vec3.from_any(np.array([1.0, 2.0, 3.0])) == Vec3(1, 2, 3)


def test_vec3_from_any_rejects_shapes_and_strings():
    with pytest.raises(GeometryError):
        Vec3.from_any(Point((0, 0, 0)))
    with pytest.raises(GeometryError):
        Vec3.from_any("ab")
    with pytest.raises(GeometryError):
        Vec3.from_any([1, 2, 3, 4])


def test_vec3_arithmetic():
    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    assert a + b == Vec3(5, 7, 9)
    assert b - a == Vec3(3, 3, 3)
    assert -a == Vec3(-1, -2, -3)
    assert a.dot(b) == 32.0
    assert a.cross(b) == Vec3(-3, 6, -3)
    assert Vec3(3, 4, 0).length() == 5.0
    assert Vec3(3, 4, 0).normalized().length() == pytest.approx(1.0)


# --- labels ---------------------------------------------------------------


def test_label_is_stripped():
    with pytest.warns(UserWarning):
        p = Point((0, 0, 0), label="  wall  ")
    assert p.label == "wall"


def test_already_stripped_label_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Point((0, 0, 0), label="wall")
    assert not caught


def test_label_validation_errors():
    for bad in ("", "   ", "a" * 128, 'has "quote"', "line\nbreak"):
        with pytest.raises(LabelError):
            Point((0, 0, 0), label=bad)
    with pytest.raises(LabelError):
        Point((0, 0, 0), label=42)  # non-str


# --- shape base -----------------------------------------------------------


def test_shape_base_and_frozen():
    p = Point((0, 0, 0))
    assert isinstance(p, Shape)
    assert hash(p) == hash(Point((0, 0, 0)))
    with pytest.raises(AttributeError):
        p.coord = Vec3(1, 1, 1)


# --- primitives -----------------------------------------------------------


def test_point():
    p = Point((1, 2, 3))
    assert p.coord == Vec3(1, 2, 3)
    assert p.dimension == 0


def test_line_rejects_identical_endpoints():
    with pytest.raises(GeometryError):
        Line((0, 0, 0), (0, 0, 0))


def test_polyline_closed():
    assert Polyline([(0, 0, 0), (1, 0, 0)]).closed is False
    closed = Polyline([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)])
    assert closed.closed is True
    with pytest.raises(GeometryError):
        Polyline([(0, 0, 0), (1, 0, 0), (0, 0, 0)])  # closed but only 2 distinct points


def test_polygon_validation():
    assert Polygon([(0, 0, 0), (1, 0, 0), (0, 1, 0)]).dimension == 2
    with pytest.raises(GeometryError):
        Polygon([(0, 0, 0), (1, 0, 0), (0, 0, 0)])  # repeated final point
    with pytest.raises(GeometryError):
        Polygon([(0, 0, 0), (1, 0, 0), (2, 0, 0)])  # collinear
    with pytest.raises(GeometryError):
        Polygon([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])  # not coplanar


def test_rectangle_validation():
    assert Rectangle((0, 0, 0), 1, 2).dimension == 2
    assert Rectangle((0, 0, 1), 1, 2).origin == Vec3(0, 0, 1)  # any z is allowed
    with pytest.raises(GeometryError):
        Rectangle((0, 0, 0), -1, 2)  # dx <= 0


def test_curve_primitives():
    assert Circle((0, 0, 0), 1).dimension == 1
    assert Ellipse((0, 0, 0), 2, 1).dimension == 1
    with pytest.raises(GeometryError):
        Circle((0, 0, 0), 0)
    with pytest.raises(GeometryError):
        Ellipse((0, 0, 0), 1, -1)


def test_disk_primitives():
    assert Disk((0, 0, 0), 1).dimension == 2
    assert Disk((0, 0, 2), 1).center == Vec3(0, 0, 2)  # any z is allowed
    assert EllipticalDisk((0, 0, 0), 2, 1).dimension == 2
    with pytest.raises(GeometryError):
        Disk((0, 0, 0), 0)
    with pytest.raises(GeometryError):
        EllipticalDisk((0, 0, 0), 1, 0)


def test_volume_primitives():
    assert Sphere((0, 0, 0), 1).dimension == 3
    with pytest.raises(GeometryError):
        Sphere((0, 0, 0), 0)
    assert Box((0, 0, 0), 1, 1, 1).dimension == 3
    with pytest.raises(GeometryError):
        Box((0, 0, 0), 1, 1, 0)
    assert Cylinder((0, 0, 0), (0, 0, 1), 0.5).dimension == 3
    with pytest.raises(GeometryError):
        Cylinder((0, 0, 0), (0, 0, 0), 0.5)


# --- booleans -------------------------------------------------------------


def test_boolean_construction_and_dimension():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0.5, 0.5), 1, 1, 1)
    assert Union(a, b).dimension == 3
    assert Union(a, b, label="x").label == "x"
    assert isinstance(a | b, Union)
    assert isinstance(a & b, Intersection)
    assert isinstance(a - b, Difference)


def test_boolean_validation_errors():
    a = Box((0, 0, 0), 1, 1, 1)
    with pytest.raises(GeometryError):
        Union(a)  # fewer than 2 operands
    with pytest.raises(GeometryError):
        Union(a, Point((0, 0, 0)))  # mixed dimensions
    with pytest.raises(GeometryError):
        Union(a, "not a shape")


def test_boolean_allows_reusing_same_object():
    a = Box((0, 0, 0), 1, 1, 1)
    assert len(Union(a, a).operands) == 2


def test_boolean_distinct_equal_objects_allowed():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0, 0, 0), 1, 1, 1)  # equal value, distinct object
    assert len(Union(a, b).operands) == 2


# --- errors ---------------------------------------------------------------


def test_error_hierarchy():
    assert issubclass(GeometryError, ValueError)
    assert issubclass(GeometryError, ShapeMeshError)
    assert issubclass(LabelError, ValueError)
    assert issubclass(BuildError, RuntimeError)
    assert BuildError("boom", log="some log").log == "some log"
