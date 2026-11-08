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

import math
import warnings
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from shapemesh import (
    Box,
    BuildError,
    Circle,
    CircleArc,
    Cylinder,
    Difference,
    Disk,
    Ellipse,
    EllipticalDisk,
    GeometryError,
    Intersection,
    LabelError,
    Line,
    Path,
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


def test_vec3_rejects_float_overflow():
    with pytest.raises(GeometryError):
        Vec3(10**1000, 0)


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


def test_vec3_from_any_rejects_undeclared_containers():
    with pytest.raises(GeometryError):
        Vec3.from_any({1, 2})
    with pytest.raises(GeometryError):
        Vec3.from_any(component for component in (1, 2))


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


def test_polygon_rejects_repeated_nonconsecutive_vertices():
    with pytest.raises(GeometryError, match="distinct"):
        Polygon([(0, 0), (1, 0), (0, 1), (1, 0)])


@pytest.mark.parametrize("shape_type", [Polyline, Polygon])
def test_point_sequences_reject_non_sequence_containers(shape_type):
    with pytest.raises(GeometryError):
        shape_type(None)
    with pytest.raises(GeometryError):
        shape_type({(0, 0), (1, 0), (0, 1)})
    with pytest.raises(GeometryError):
        shape_type(point for point in [(0, 0), (1, 0), (0, 1)])


def test_rectangle_validation():
    assert Rectangle((0, 0, 0), 1, 2).dimension == 2
    assert Rectangle((0, 0, 1), 1, 2).origin == Vec3(0, 0, 1)  # any z is allowed
    with pytest.raises(GeometryError):
        Rectangle((0, 0, 0), -1, 2)  # dx <= 0


def test_curve_primitives():
    assert Circle((0, 0, 0), 1).dimension == 1
    assert Circle((0, 0), Fraction(1, 2)).radius == 0.5
    assert Ellipse((0, 0, 0), 2, 1).dimension == 1
    with pytest.raises(GeometryError):
        Circle((0, 0, 0), 0)
    with pytest.raises(GeometryError):
        Ellipse((0, 0, 0), 1, -1)


@pytest.mark.parametrize(
    "radius",
    [Fraction(1, 10**400), 10**1000],
    ids=["underflow", "overflow"],
)
def test_positive_shape_dimensions_validate_converted_float(radius):
    with pytest.raises(GeometryError):
        Circle((0, 0), radius)


def test_disk_primitives():
    assert Disk((0, 0, 0), 1).dimension == 2
    assert Disk((0, 0, 2), 1).center == Vec3(0, 0, 2)  # any z is allowed
    assert EllipticalDisk((0, 0, 0), 2, 1).dimension == 2
    with pytest.raises(GeometryError):
        Disk((0, 0, 0), 0)
    with pytest.raises(GeometryError):
        EllipticalDisk((0, 0, 0), 1, 0)


def test_planar_shape_directions_are_normalized():
    for shape in (
        Circle((0, 0), 1, normal=(0, 0, 10)),
        Disk((0, 0), 1, normal=(0, 0, 10)),
    ):
        assert shape.normal == Vec3(0, 0, 1)

    ellipse = Ellipse((0, 0), 2, 1, normal=(0, 0, 10), x_axis=(0, 5, 0))
    assert ellipse.normal == Vec3(0, 0, 1)
    assert ellipse.x_axis == Vec3(0, 1, 0)
    assert Ellipse((0, 0), 2, 1).x_axis is None
    assert EllipticalDisk((0, 0), 2, 1).x_axis is None


def test_direction_normalization_is_scale_safe():
    shape = Circle((0, 0), 1, normal=(1e308, 1e308, 1e308))
    assert shape.normal.x == pytest.approx(3**-0.5)
    assert shape.normal.y == pytest.approx(3**-0.5)
    assert shape.normal.z == pytest.approx(3**-0.5)

    tiny = Disk((0, 0), 1, normal=(5e-324, 0, 0))
    assert tiny.normal == Vec3(1, 0, 0)


@pytest.mark.parametrize("shape_type", [Circle, Disk])
def test_circular_shapes_reject_zero_normal(shape_type):
    with pytest.raises(GeometryError, match="normal must be nonzero"):
        shape_type((0, 0), 1, normal=(0, 0, 0))


@pytest.mark.parametrize("shape_type", [Ellipse, EllipticalDisk])
def test_elliptical_shapes_reject_zero_normal(shape_type):
    with pytest.raises(GeometryError, match="normal must be nonzero"):
        shape_type((0, 0), 2, 1, normal=(0, 0, 0))


@pytest.mark.parametrize("shape_type", [Ellipse, EllipticalDisk])
@pytest.mark.parametrize("x_axis", [(0, 0, 0), (0, 0, 1), (1, 0, 1)])
def test_elliptical_shapes_reject_invalid_x_axis(shape_type, x_axis):
    with pytest.raises(GeometryError):
        shape_type((0, 0), 2, 1, x_axis=x_axis)


def test_replace_recomputes_automatic_ellipse_axis():
    ellipse = Ellipse((0, 0), 2, 1)
    replaced = replace(ellipse, normal=(1, 0, 0))
    assert replaced.normal == Vec3(1, 0, 0)
    assert replaced.x_axis is None


# --- arcs and paths -------------------------------------------------------


def _assert_vec3(actual, expected):
    expected = Vec3.from_any(expected)
    assert (actual.x, actual.y, actual.z) == pytest.approx((expected.x, expected.y, expected.z))


def test_circle_arc_geometry():
    arc = CircleArc((0, 0, 0), (1, 0, 0), math.pi / 2)
    assert arc.dimension == 1
    assert arc.radius == pytest.approx(1.0)
    _assert_vec3(arc.end, (0, 1, 0))
    _assert_vec3(arc.start_tangent, (0, 1, 0))
    _assert_vec3(arc.end_tangent, (-1, 0, 0))

    reverse = CircleArc((0, 0, 0), (1, 0, 0), -math.pi / 2)
    _assert_vec3(reverse.end, (0, -1, 0))
    _assert_vec3(reverse.start_tangent, (0, -1, 0))
    _assert_vec3(reverse.end_tangent, (-1, 0, 0))

    semicircle = CircleArc((0, 0, 0), (1, 0, 0), math.pi)
    _assert_vec3(semicircle.end, (-1, 0, 0))
    _assert_vec3(semicircle.end_tangent, (0, -1, 0))


def test_circle_arc_normalizes_normal_and_supports_replace():
    arc = CircleArc((0, 0, 0), (1, 0, 0), math.pi / 2, normal=(0, 0, 10), label="arc")
    assert arc.normal == Vec3(0, 0, 1)
    assert arc.label == "arc"

    replaced = replace(arc, angle=math.pi, label="replaced")
    assert replaced.angle == math.pi
    assert replaced.label == "replaced"
    _assert_vec3(replaced.end, (-1, 0, 0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": (0, 0, 0), "start": (0, 0, 0), "angle": math.pi / 2},
        {"center": (0, 0, 0), "start": (1, 0, 0), "angle": 0.0},
        {"center": (0, 0, 0), "start": (1, 0, 0), "angle": 2 * math.pi},
        {"center": (0, 0, 0), "start": (1, 0, 0), "angle": -2 * math.pi},
        {"center": (0, 0, 0), "start": (1, 0, 0), "angle": math.nan},
        {"center": (0, 0, 0), "start": (1, 0, 0), "angle": math.inf},
        {
            "center": (0, 0, 0),
            "start": (1, 0, 0),
            "angle": math.pi / 2,
            "normal": (0, 0, 0),
        },
        {
            "center": (0, 0, 0),
            "start": (1, 0, 0),
            "angle": math.pi / 2,
            "normal": (1, 0, 0),
        },
    ],
)
def test_circle_arc_rejects_invalid_geometry(kwargs):
    with pytest.raises(GeometryError):
        CircleArc(**kwargs)


def test_path_connects_line_and_arc():
    line = Line((0, 0, 0), (0, 0, 2))
    arc = CircleArc((0.5, 0, 2), (0, 0, 2), math.pi / 2, normal=(0, 1, 0))
    tail = Line((0.5, 0, 2.5), (1.0, 0, 2.5))
    path = Path(line, arc, tail, label="guide")

    assert path.dimension == 1
    assert path.segments == (line, arc, tail)
    assert path.label == "guide"
    assert path.start == Vec3(0, 0, 0)
    _assert_vec3(path.end, (1.0, 0, 2.5))
    assert path.closed is False


def test_path_closed_and_closure_tangency():
    first = CircleArc((0, 0, 0), (1, 0, 0), math.pi, normal=(0, 0, 1))
    second = CircleArc((0, 0, 0), (-1, 0, 0), math.pi, normal=(0, 0, 1))
    assert Path(first, second).closed is True

    line = Line((0, 0, 0), (1, 0, 0))
    bend = CircleArc((1, 1, 0), (1, 0, 0), math.pi / 2)
    return_arc = CircleArc((0.75, 1, 0), (2, 1, 0), 2 * math.pi - math.acos(-0.6))
    with pytest.raises(GeometryError, match="closure"):
        Path(line, bend, return_arc)


def test_path_rejects_invalid_segments():
    line = Line((0, 0, 0), (0, 0, 1))
    with pytest.raises(GeometryError, match="at least 2"):
        Path(line)
    with pytest.raises(GeometryError, match="Shape"):
        Path(line, "not a shape")
    with pytest.raises(GeometryError, match="Line or CircleArc"):
        Path(line, Point((0, 0, 1)))
    with pytest.raises(GeometryError, match="Line or CircleArc"):
        Path(line, Polyline([(0, 0, 1), (1, 0, 1)]))
    with pytest.raises(GeometryError, match="connect"):
        Path(line, Line((0.1, 0, 1), (0.1, 0, 2)))
    with pytest.raises(GeometryError, match="tangent-continuous"):
        Path(line, Line((0, 0, 1), (1, 0, 1)))
    with pytest.raises(GeometryError, match="tangent-continuous"):
        Path(line, Line((0, 0, 1), (0, 0, 0)))


def test_path_dataclass_replace():
    line = Line((0, 0, 0), (0, 0, 1))
    arc = CircleArc((1, 0, 1), (0, 0, 1), math.pi / 2, normal=(0, 1, 0))
    path = Path(line, arc, label="original")

    renamed = replace(path, label="renamed")
    assert renamed.segments == (line, arc)
    assert renamed.label == "renamed"

    rebuilt = replace(path, segments=(line, arc))
    assert rebuilt.segments == (line, arc)

    with pytest.raises(TypeError, match="both"):
        Path(line, segments=(arc,))


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


def test_boolean_accepts_sequence():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    c = Box((2, 0, 0), 1, 1, 1)
    assert Union([a, b, c]).operands == (a, b, c)
    assert Intersection([a, b]).operands == (a, b)
    assert Difference([a, b]).operands == (a, b)


def test_boolean_dataclass_replace():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    original = Union(a, b, label="original")

    replaced = replace(original, operands=(a, a), label="replaced")
    assert isinstance(replaced, Union)
    assert replaced.operands == (a, a)
    assert replaced.label == "replaced"

    renamed = replace(replaced, label="renamed")
    assert renamed.operands == (a, a)
    assert renamed.label == "renamed"


def test_boolean_validation_errors():
    a = Box((0, 0, 0), 1, 1, 1)
    with pytest.raises(GeometryError):
        Union(a)  # fewer than 2 operands
    with pytest.raises(GeometryError):
        Union(a, Point((0, 0, 0)))  # mixed dimensions
    with pytest.raises(GeometryError):
        Union(a, "not a shape")
    with pytest.raises(TypeError):
        Union(a, a, operands=(a, a))


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
