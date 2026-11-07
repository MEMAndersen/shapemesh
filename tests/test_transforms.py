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
from dataclasses import replace

import pytest

from shapemesh import (
    Box,
    Difference,
    GeometryError,
    Line,
    MeshSettings,
    Point,
    Polygon,
    Polyline,
    Rotate,
    Translate,
    Union,
    Vec3,
    build,
)


def test_translate_construction_and_replace():
    operand = Line((0, 0), (1, 0), label="inner")
    transform = Translate(operand, (1, 2), label="outer")

    assert transform.dimension == 1
    assert transform.offset == Vec3(1, 2, 0)
    assert transform.label == "outer"
    assert hash(transform) == hash(Translate(operand, (1, 2), label="outer"))
    with pytest.raises(AttributeError):
        transform.offset = Vec3(3, 4, 0)

    replaced = replace(transform, offset=(3, 4), label="replaced")
    assert replaced.operand is operand
    assert replaced.offset == Vec3(3, 4, 0)
    assert replaced.label == "replaced"


def test_rotate_construction_and_replace():
    operand = Line((0, 0), (1, 0))
    transform = Rotate(
        operand,
        math.pi / 2,
        origin=(1, 0),
        axis=(0, 0, 10),
        label="rotated",
    )

    assert transform.dimension == 1
    assert transform.angle == math.pi / 2
    assert transform.origin == Vec3(1, 0, 0)
    assert transform.axis == Vec3(0, 0, 1)
    assert transform.label == "rotated"

    replaced = replace(transform, angle=-math.pi / 2, axis=(1, 0, 0))
    assert replaced.angle == -math.pi / 2
    assert replaced.axis == Vec3(1, 0, 0)


def test_rotation_axis_normalization_is_scale_safe():
    large = Rotate(Point((0, 0)), 1, axis=(1e308, 1e308, 1e308))
    assert large.axis.x == pytest.approx(3**-0.5)
    assert large.axis.y == pytest.approx(3**-0.5)
    assert large.axis.z == pytest.approx(3**-0.5)

    tiny = Rotate(Point((0, 0)), 1, axis=(5e-324, 0, 0))
    assert tiny.axis == Vec3(1, 0, 0)


def test_transforms_reject_invalid_operands():
    with pytest.raises(GeometryError, match="operand"):
        Translate("not a shape", (1, 0))
    with pytest.raises(GeometryError, match="operand"):
        Rotate("not a shape", 1)


@pytest.mark.parametrize("angle", [True, "1", float("nan"), float("inf"), 10**1000])
def test_rotate_rejects_invalid_angle(angle):
    with pytest.raises(GeometryError):
        Rotate(Point((0, 0)), angle)


def test_transforms_reject_invalid_vectors():
    with pytest.raises(GeometryError):
        Translate(Point((0, 0)), (float("inf"), 0))
    with pytest.raises(GeometryError):
        Rotate(Point((0, 0)), 1, origin=(float("nan"), 0))
    with pytest.raises(GeometryError, match="axis must be nonzero"):
        Rotate(Point((0, 0)), 1, axis=(0, 0, 0))


@pytest.mark.parametrize(
    ("operand", "expected_bbox"),
    [
        (Point((1, 2, 3)), (2, 4, 6, 2, 4, 6)),
        (Line((0, 0, 0), (1, 0, 0)), (1, 2, 3, 2, 2, 3)),
        (Polygon([(0, 0), (1, 0), (0, 1)]), (1, 2, 3, 2, 3, 3)),
        (Box((0, 0, 0), 1, 2, 3), (1, 2, 3, 2, 4, 6)),
    ],
    ids=["point", "line", "surface", "volume"],
)
def test_translate_across_dimensions(operand, expected_bbox):
    mesh = build(Translate(operand, (1, 2, 3)), MeshSettings())
    assert mesh.bbox == pytest.approx(expected_bbox, abs=1e-8)


@pytest.mark.parametrize(
    ("operand", "angle", "rotation_kwargs", "expected"),
    [
        (Point((1, 0, 0)), math.pi / 2, {}, (0, 1, 0)),
        (Point((1, 0, 0)), -math.pi / 2, {}, (0, -1, 0)),
        (Point((2, 0, 0)), math.pi / 2, {"origin": (1, 0, 0)}, (1, 1, 0)),
        (Point((0, 1, 0)), math.pi / 2, {"axis": (1, 0, 0)}, (0, 0, 1)),
        (
            Point((1, 0, 0)),
            math.pi,
            {"axis": (1e308, 1e308, 1e308)},
            (-1 / 3, 2 / 3, 2 / 3),
        ),
    ],
    ids=["z-axis", "negative-angle", "offset-origin", "x-axis", "diagonal-axis"],
)
def test_rotate_point(operand, angle, rotation_kwargs, expected):
    mesh = build(Rotate(operand, angle, **rotation_kwargs), MeshSettings())
    assert mesh.nodes.coords[0] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    ("operand", "expected_bbox"),
    [
        (Polygon([(0, 0), (1, 0), (0, 1)]), (-1, 0, 0, 0, 1, 0)),
        (Box((0, 0, 0), 1, 2, 3), (-2, 0, 0, 0, 1, 3)),
    ],
    ids=["surface", "volume"],
)
def test_rotate_surface_and_volume(operand, expected_bbox):
    mesh = build(Rotate(operand, math.pi / 2), MeshSettings())
    assert mesh.bbox == pytest.approx(expected_bbox, abs=1e-8)


@pytest.mark.parametrize(
    "transform",
    [
        Translate(Point((1, 2, 3)), (0, 0, 0)),
        Rotate(Point((1, 2, 3)), 0),
    ],
    ids=["translate", "rotate"],
)
def test_identity_transforms(transform):
    mesh = build(transform, MeshSettings())
    assert mesh.nodes.coords[0] == pytest.approx((1, 2, 3))


def test_nested_transform_order():
    point = Point((1, 0, 0))
    rotate_then_translate = Translate(Rotate(point, math.pi / 2), (1, 0, 0))
    translate_then_rotate = Rotate(Translate(point, (1, 0, 0)), math.pi / 2)

    first = build(rotate_then_translate, MeshSettings())
    second = build(translate_then_rotate, MeshSettings())

    assert first.nodes.coords[0] == pytest.approx((1, 1, 0))
    assert second.nodes.coords[0] == pytest.approx((0, 2, 0))


@pytest.mark.parametrize(
    ("transform", "expected_bbox"),
    [
        (
            Translate(Polyline([(0, 0), (1, 0), (1, 1)]), (1, 2, 3)),
            (1, 2, 3, 2, 3, 3),
        ),
        (
            Rotate(Polyline([(0, 0), (1, 0), (1, 1)]), math.pi / 2),
            (-1, 0, 0, 0, 1, 0),
        ),
    ],
    ids=["translate", "rotate"],
)
def test_transform_multi_entity_polyline_preserves_shared_topology(transform, expected_bbox):
    mesh = build(transform, MeshSettings())
    assert mesh.bbox == pytest.approx(expected_bbox, abs=1e-8)

    curves = mesh.entities(dim=1)
    assert len(curves) == 2
    point_keys = [(0, abs(tag)) for curve in curves for _, tag in curve.boundary]
    assert len(set(point_keys)) == 3
    assert sorted(point_keys.count(key) for key in set(point_keys)) == [1, 1, 2]


def test_transform_around_multi_entity_boolean_result():
    combined = Union(
        Box((0, 0, 0), 1, 1, 1),
        Box((2, 0, 0), 1, 1, 1),
    )
    mesh = build(Translate(combined, (5, 0, 0), label="moved"), MeshSettings())

    assert mesh.bbox == pytest.approx((5, 0, 0, 8, 1, 1), abs=1e-8)
    assert len(mesh.physical_group("moved", 3).entities) == 2


def test_transform_inside_boolean():
    moved_tool = Translate(Box((0.5, 0, 0), 1, 1, 1), (-0.25, 0, 0))
    combined = Union(Box((0, 0, 0), 1, 1, 1), moved_tool, label="combined")
    mesh = build(combined, MeshSettings())

    assert mesh.bbox == pytest.approx((0, 0, 0, 1.25, 1, 1), abs=1e-8)
    assert len(mesh.physical_group("combined", 3).entities) == 1


def test_rotate_around_boolean_difference():
    difference = Difference(
        Box((0, 0, 0), 2, 1, 1),
        Box((0.5, 0.25, 0), 0.5, 0.5, 1),
    )
    mesh = build(Rotate(difference, math.pi / 2, label="cut"), MeshSettings())

    assert mesh.bbox == pytest.approx((-1, 0, 0, 0, 2, 1), abs=1e-8)
    assert len(mesh.physical_group("cut", 3).entities) == 1


def test_shape_reuse_and_transform_labels():
    base = Box((0, 0, 0), 1, 1, 1, label="base")
    moved = Translate(base, (2, 0, 0), label="moved")
    mesh = build([base, moved], MeshSettings())
    entities = {(entity.dim, entity.tag): entity for entity in mesh.entities()}

    assert mesh.bbox == pytest.approx((0, 0, 0, 3, 1, 1), abs=1e-8)
    base_group = mesh.physical_group("base", 3)
    moved_group = mesh.physical_group("moved", 3)
    assert len(base_group.entities) == 1
    assert len(moved_group.entities) == 1
    assert entities[base_group.entities[0]].bbox == pytest.approx((0, 0, 0, 1, 1, 1), abs=1e-6)
    assert entities[moved_group.entities[0]].bbox == pytest.approx((2, 0, 0, 3, 1, 1), abs=1e-6)


def test_transform_consumes_operand_label():
    operand = Box((0, 0, 0), 1, 1, 1, label="inner")
    mesh = build(Translate(operand, (1, 0, 0)), MeshSettings())
    assert mesh.groups == ()


def test_transformed_groups_survive_fragmentation():
    moved = Translate(Box((0, 0, 0), 2, 1, 1), (1, 0, 0), label="moved")
    fixed = Box((2, 0, 0), 2, 1, 1, label="fixed")
    mesh = build([moved, fixed], MeshSettings())

    moved_entities = set(mesh.physical_group("moved", 3).entities)
    fixed_entities = set(mesh.physical_group("fixed", 3).entities)
    assert len(mesh.entities(dim=3)) == 3
    assert len(moved_entities & fixed_entities) == 1
