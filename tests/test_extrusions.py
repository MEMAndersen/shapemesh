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

import numpy as np
import pytest

from shapemesh import (
    Box,
    Circle,
    Disk,
    ElementType,
    Ellipse,
    Extrude,
    ExtrudeAlong,
    GeometryError,
    Line,
    MeshSettings,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    Revolve,
    Translate,
    Union,
    Vec3,
    build,
)


def _mesh_length(mesh):
    view = mesh.blocks_by_type().get(ElementType.LINE2)
    if view is None:
        return 0.0
    lengths = 0.0
    for row in view.connectivity:
        points = mesh.nodes.coords[row]
        lengths += np.linalg.norm(points[1] - points[0])
    return lengths


def _mesh_area(mesh):
    view = mesh.blocks_by_type().get(ElementType.TRI3)
    if view is None:
        return 0.0
    area = 0.0
    for row in view.connectivity:
        points = mesh.nodes.coords[row]
        area += np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2
    return area


def _mesh_volume(mesh):
    view = mesh.blocks_by_type().get(ElementType.TET4)
    if view is None:
        return 0.0
    volume = 0.0
    for row in view.connectivity:
        points = mesh.nodes.coords[row]
        matrix = [points[1] - points[0], points[2] - points[0], points[3] - points[0]]
        volume += abs(np.linalg.det(matrix)) / 6
    return volume


def test_extrude_construction_and_replace():
    operand = Rectangle((0, 0), 1, 1)
    operation = Extrude(operand, (0, 0, 2), label="solid")

    assert operation.dimension == 3
    assert operation.offset == Vec3(0, 0, 2)
    assert operation.label == "solid"

    replaced = replace(operation, offset=(1, 2, 3), label="replaced")
    assert replaced.offset == Vec3(1, 2, 3)
    assert replaced.label == "replaced"


def test_revolve_construction_and_replace():
    operand = Line((1, 0), (2, 0))
    operation = Revolve(
        operand,
        -math.pi / 2,
        pivot=(1, 0),
        axis=(0, 0, 10),
        label="surface",
    )

    assert operation.dimension == 2
    assert operation.angle == -math.pi / 2
    assert operation.pivot == Vec3(1, 0, 0)
    assert operation.axis == Vec3(0, 0, 1)

    replaced = replace(operation, angle=math.pi, axis=(1, 0, 0))
    assert replaced.angle == math.pi
    assert replaced.axis == Vec3(1, 0, 0)


def test_extrude_along_construction_and_replace():
    profile = Rectangle((0, 0), 1, 1)
    guide = Line((0, 0, 0), (0, 0, 1))
    operation = ExtrudeAlong(profile, guide, label="swept")

    assert operation.dimension == 3
    assert operation.operand is profile
    assert operation.guide is guide
    assert operation.label == "swept"

    replacement_guide = Line((0, 0, 0), (1, 0, 0))
    replaced = replace(operation, guide=replacement_guide, label="replaced")
    assert replaced.guide is replacement_guide
    assert replaced.label == "replaced"


def test_extrusions_reject_invalid_operands():
    guide = Line((0, 0), (0, 1))
    with pytest.raises(GeometryError, match="operand"):
        Extrude("not a shape", (0, 0, 1))
    with pytest.raises(GeometryError, match="operand"):
        Revolve("not a shape", 1)
    with pytest.raises(GeometryError, match="operand"):
        ExtrudeAlong("not a shape", guide)
    with pytest.raises(GeometryError, match="guide"):
        ExtrudeAlong(Rectangle((0, 0), 1, 1), "not a shape")


def test_extrusions_reject_volume_profiles():
    volume = Box((0, 0, 0), 1, 1, 1)
    guide = Line((0, 0), (0, 1))
    with pytest.raises(GeometryError, match="at most 2"):
        Extrude(volume, (0, 0, 1))
    with pytest.raises(GeometryError, match="at most 2"):
        Revolve(volume, 1)
    with pytest.raises(GeometryError, match="at most 2"):
        ExtrudeAlong(volume, guide)


def test_extrusion_validation():
    with pytest.raises(GeometryError, match="offset must be nonzero"):
        Extrude(Point((0, 0)), (0, 0, 0))
    with pytest.raises(GeometryError):
        Extrude(Point((0, 0)), (float("nan"), 0, 1))
    with pytest.raises(GeometryError, match="angle must be nonzero"):
        Revolve(Point((0, 0)), 0)
    with pytest.raises(GeometryError):
        Revolve(Point((0, 0)), float("nan"))
    with pytest.raises(GeometryError, match=r"2 \* math.pi"):
        Revolve(Point((0, 0)), 2 * math.pi + 0.1)
    with pytest.raises(GeometryError, match="axis must be nonzero"):
        Revolve(Point((0, 0)), 1, axis=(0, 0, 0))
    with pytest.raises(GeometryError, match="one-dimensional"):
        ExtrudeAlong(Rectangle((0, 0), 1, 1), Point((0, 0)))


def test_extrude_point_to_line():
    mesh = build(Extrude(Point((1, 2, 3)), (0, 0, 2), label="line"), MeshSettings())

    assert mesh.metadata.dim == 1
    assert mesh.bbox == pytest.approx((1, 2, 3, 1, 2, 5), abs=1e-8)
    assert _mesh_length(mesh) == pytest.approx(2)
    assert len(mesh.physical_group("line", 1).entities) == 1
    assert len(mesh.entities(dim=0)) == 2


def test_extrude_line_to_surface():
    line = Line((1, 0, 0), (2, 0, 0))
    mesh = build(Extrude(line, (0, 0, 2), label="surface"), MeshSettings())

    assert mesh.metadata.dim == 2
    assert mesh.bbox == pytest.approx((1, 0, 0, 2, 0, 2), abs=1e-8)
    assert _mesh_area(mesh) == pytest.approx(2)
    assert len(mesh.physical_group("surface", 2).entities) == 1


def test_extrude_surface_to_volume():
    mesh = build(
        Extrude(Rectangle((0, 0), 1, 1), (0, 0, 2), label="volume"),
        MeshSettings(),
    )

    assert mesh.metadata.dim == 3
    assert mesh.bbox == pytest.approx((0, 0, 0, 1, 1, 2), abs=1e-8)
    assert _mesh_volume(mesh) == pytest.approx(2)
    assert len(mesh.physical_group("volume", 3).entities) == 1


def test_extrude_multi_entity_profile():
    profile = Polyline([(0, 0), (1, 0), (1, 1)])
    mesh = build(Extrude(profile, (0, 0, 1), label="surfaces"), MeshSettings())

    assert mesh.bbox == pytest.approx((0, 0, 0, 1, 1, 1), abs=1e-8)
    assert _mesh_area(mesh) == pytest.approx(2)
    assert len(mesh.physical_group("surfaces", 2).entities) == 2


@pytest.mark.parametrize(
    ("point", "angle", "rotation_kwargs", "expected_bbox"),
    [
        (Point((1, 0, 0)), math.pi / 2, {}, (0, 0, 0, 1, 1, 0)),
        (Point((1, 0, 0)), -math.pi / 2, {}, (0, -1, 0, 1, 0, 0)),
        (
            Point((1, 1, 0)),
            math.pi / 2,
            {"pivot": (1, 0, 0), "axis": (2, 0, 0)},
            (1, 0, 0, 1, 1, 1),
        ),
    ],
    ids=["z-axis", "negative-angle", "x-axis-offset-pivot"],
)
def test_revolve_point_to_curve(point, angle, rotation_kwargs, expected_bbox):
    mesh = build(Revolve(point, angle, **rotation_kwargs), MeshSettings())

    assert mesh.metadata.dim == 1
    assert mesh.bbox == pytest.approx(expected_bbox, abs=1e-6)
    assert _mesh_length(mesh) == pytest.approx(math.pi / 2, rel=0.01)


def test_revolve_line_to_annulus():
    mesh = build(
        Revolve(Line((1, 0, 0), (2, 0, 0)), 2 * math.pi, label="annulus"),
        MeshSettings(),
    )

    assert mesh.metadata.dim == 2
    entity = mesh.entities(dim=2)[0]
    assert entity.bbox == pytest.approx((-2, -2, 0, 2, 2, 0), abs=1e-6)
    assert _mesh_area(mesh) == pytest.approx(3 * math.pi, rel=0.05)
    assert len(mesh.physical_group("annulus", 2).entities) == 1


def test_revolve_surface_to_full_annular_cylinder():
    profile = Polygon([(1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)])
    mesh = build(Revolve(profile, 2 * math.pi, label="cylinder"), MeshSettings())

    assert mesh.metadata.dim == 3
    assert mesh.bbox == pytest.approx((-2, -2, 0, 2, 2, 1), abs=1e-6)
    assert _mesh_volume(mesh) == pytest.approx(3 * math.pi, rel=0.05)
    assert len(mesh.physical_group("cylinder", 3).entities) == 1
    assert len(mesh.entities(dim=2)) == 4


def test_revolve_surface_to_partial_cylinder():
    profile = Polygon([(1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)])
    mesh = build(Revolve(profile, math.pi / 2), MeshSettings())

    assert mesh.bbox == pytest.approx((0, 0, 0, 2, 2, 1), abs=1e-6)
    assert _mesh_volume(mesh) == pytest.approx(3 * math.pi / 4, rel=0.05)
    assert len(mesh.entities(dim=2)) == 6


def test_extrude_along_point_profile():
    guide = Line((0, 0, 0), (0, 0, 2))
    mesh = build(ExtrudeAlong(Point((0, 0, 0)), guide, label="curve"), MeshSettings())

    assert mesh.metadata.dim == 1
    assert mesh.bbox == pytest.approx((0, 0, 0, 0, 0, 2), abs=1e-8)
    assert _mesh_length(mesh) == pytest.approx(2)
    assert len(mesh.entities(dim=1)) == 1


def test_extrude_along_straight_guide_matches_linear_extrusion():
    profile = Rectangle((0, 0), 1, 1)
    guide = Line((0, 0, 0), (0, 0, 2))
    mesh = build(ExtrudeAlong(profile, guide, label="swept"), MeshSettings())

    assert mesh.metadata.dim == 3
    assert mesh.bbox == pytest.approx((0, 0, 0, 1, 1, 2), abs=1e-6)
    assert _mesh_volume(mesh) == pytest.approx(2)
    assert len(mesh.physical_group("swept", 3).entities) == 1
    # The swept box has 8 vertices, 12 edges, and 6 faces. A leftover guide
    # curve would appear as an additional 1D entity.
    assert len(mesh.entities(dim=0)) == 8
    assert len(mesh.entities(dim=1)) == 12
    assert len(mesh.entities(dim=2)) == 6


def test_extrude_along_line_profile():
    profile = Circle(center=(0, 0, 0), radius=0.5)
    guide = Line((0, 0, 0), (0, 0, 2))
    mesh = build(ExtrudeAlong(profile, guide), MeshSettings(element_size=0.1))

    assert mesh.metadata.dim == 2
    entity = mesh.entities(dim=2)[0]
    assert entity.bbox == pytest.approx((-0.5, -0.5, 0, 0.5, 0.5, 2), abs=1e-6)
    assert _mesh_area(mesh) == pytest.approx(2 * math.pi * 0.5 * 2, rel=0.05)


def test_extrude_along_circular_guide_creates_torus():
    radius = 0.1
    profile = Disk(center=(1, 0, 0), radius=radius, normal=(0, 1, 0))
    guide = Circle(center=(0, 0, 0), radius=1)
    mesh = build(
        ExtrudeAlong(profile, guide, label="torus"),
        MeshSettings(element_size=0.05),
    )

    expected_volume = 2 * math.pi**2 * radius**2
    entity = mesh.entities(dim=3)[0]
    assert entity.bbox == pytest.approx((-1.1, -1.1, -0.1, 1.1, 1.1, 0.1), abs=1e-6)
    assert _mesh_volume(mesh) == pytest.approx(expected_volume, rel=0.1)
    assert len(mesh.physical_group("torus", 3).entities) == 1


def test_extrude_along_elliptical_guide():
    radius = 0.1
    profile = Disk(center=(2, 0, 0), radius=radius, normal=(0, 1, 0))
    guide = Ellipse(center=(0, 0, 0), rx=2, ry=1)
    mesh = build(ExtrudeAlong(profile, guide), MeshSettings(element_size=0.05))

    assert mesh.bbox == pytest.approx((-2.1, -1.1, -0.1, 2.1, 1.1, 0.1), abs=0.01)
    assert _mesh_volume(mesh) == pytest.approx(0.30437, rel=0.1)


def test_extrude_along_revolution_generated_arc():
    radius = 0.1
    profile = Disk(center=(1, 0, 0), radius=radius, normal=(0, 1, 0))
    guide = Revolve(Point((1, 0, 0)), math.pi / 2)
    mesh = build(ExtrudeAlong(profile, guide), MeshSettings(element_size=0.05))

    expected_volume = 2 * math.pi**2 * radius**2 / 4
    entity = mesh.entities(dim=3)[0]
    assert entity.bbox == pytest.approx((0, 0, -0.1, 1.1, 1.1, 0.1), abs=1e-6)
    assert _mesh_volume(mesh) == pytest.approx(expected_volume, rel=0.1)


def test_extrude_along_multi_entity_profile():
    profile = Union(
        Rectangle((0, 0), 1, 1),
        Rectangle((2, 0), 1, 1),
    )
    guide = Line((0, 0, 0), (0, 0, 2))
    mesh = build(ExtrudeAlong(profile, guide, label="solids"), MeshSettings())

    assert _mesh_volume(mesh) == pytest.approx(4)
    assert len(mesh.physical_group("solids", 3).entities) == 2


def test_extrude_along_rejects_multi_segment_guide():
    profile = Rectangle((0, 0), 1, 1)
    guide = Polyline([(0, 0, 0), (0, 0, 1), (1, 0, 1)])
    with pytest.raises(GeometryError, match="exactly one smooth curve"):
        build(ExtrudeAlong(profile, guide), MeshSettings())


def test_extrude_along_consumes_profile_and_guide_labels():
    profile = Rectangle((0, 0), 1, 1, label="profile")
    guide = Line((0, 0, 0), (0, 0, 1), label="guide")
    mesh = build(ExtrudeAlong(profile, guide), MeshSettings())
    assert mesh.groups == ()


def test_extrude_and_revolve_consume_operand_labels():
    extruded = Extrude(Rectangle((0, 0), 1, 1, label="profile"), (0, 0, 1))
    revolved = Revolve(Line((1, 0, 0), (2, 0, 0), label="profile"), math.pi / 2)
    mesh = build([extruded, revolved], MeshSettings())
    assert mesh.groups == ()


def test_guide_can_be_reused_independently():
    profile = Rectangle((0, 0), 1, 1)
    guide = Line((0, 0, 0), (0, 0, 1), label="guide")
    swept = ExtrudeAlong(profile, guide, label="swept")
    mesh = build([guide, swept], MeshSettings())

    assert len(mesh.physical_group("guide", 1).entities) == 1
    assert len(mesh.physical_group("swept", 3).entities) == 1
    assert len(mesh.entities(dim=3)) == 1


def test_extrusion_composes_with_boolean_profile():
    profile = Union(
        Rectangle((0, 0), 1, 1),
        Rectangle((0.5, 0), 1, 1),
    )
    mesh = build(Extrude(profile, (0, 0, 1), label="solid"), MeshSettings())

    assert _mesh_volume(mesh) == pytest.approx(1.5)
    assert len(mesh.physical_group("solid", 3).entities) == 1


def test_extrusions_compose_with_transforms():
    moved = Translate(Rectangle((0, 0), 1, 1), (2, 0, 0))
    solid = Extrude(moved, (0, 0, 1), label="solid")
    mesh = build(Translate(solid, (0, 3, 0), label="moved"), MeshSettings())

    assert mesh.bbox == pytest.approx((2, 3, 0, 3, 4, 1), abs=1e-8)
    assert _mesh_volume(mesh) == pytest.approx(1)
    assert len(mesh.physical_group("moved", 3).entities) == 1


def test_extruded_groups_survive_fragmentation():
    extruded = Extrude(Rectangle((0, 0), 2, 1), (0, 0, 1), label="extruded")
    box = Box((1, 0, 0), 2, 1, 1, label="box")
    mesh = build([extruded, box], MeshSettings())

    extruded_entities = set(mesh.physical_group("extruded", 3).entities)
    box_entities = set(mesh.physical_group("box", 3).entities)
    assert len(mesh.entities(dim=3)) == 3
    assert len(extruded_entities & box_entities) == 1
