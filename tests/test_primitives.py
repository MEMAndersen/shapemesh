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

import numpy as np
import pytest

from shapemesh import (
    Box,
    Circle,
    Cylinder,
    Disk,
    Ellipse,
    EllipticalDisk,
    Line,
    MeshSettings,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    Sphere,
    build,
)

_CASES = [
    (Point((0, 0, 0)), 0, 1),
    (Line((0, 0, 0), (1, 0, 0)), 1, 1),
    (Polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), 1, 2),
    (Circle((0, 0, 0), 1), 1, 1),
    (Ellipse((0, 0, 0), 2, 1), 1, 1),
    (Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), 2, 1),
    (Rectangle((0, 0, 0), 1, 1), 2, 1),
    (Disk((0, 0, 0), 1), 2, 1),
    (EllipticalDisk((0, 0, 0), 2, 1), 2, 1),
    (Sphere((0, 0, 0), 1), 3, 1),
    (Box((0, 0, 0), 1, 1, 1), 3, 1),
    (Cylinder((0, 0, 0), (0, 0, 1), 0.5), 3, 1),
]

_IDS = [
    "point",
    "line",
    "polyline",
    "circle",
    "ellipse",
    "polygon",
    "rectangle",
    "disk",
    "elliptical_disk",
    "sphere",
    "box",
    "cylinder",
]


@pytest.mark.parametrize(("shape", "dim", "count"), _CASES, ids=_IDS)
def test_primitive_realizes_top_dim_entities(shape, dim, count):
    mesh = build(shape, MeshSettings())
    assert mesh.metadata.dim == dim
    assert len(mesh.entities(dim=dim)) == count
    assert mesh.num_elements > 0


def test_point_only_meshing():
    mesh = build(Point((0, 0, 0)), MeshSettings())
    assert mesh.num_nodes == 1
    assert mesh.num_elements == 1
    assert mesh.metadata.dim == 0


@pytest.mark.parametrize(
    "shape",
    [Ellipse((1, 2, 3), 1, 2), EllipticalDisk((1, 2, 3), 1, 2)],
    ids=["curve", "disk"],
)
def test_ellipse_with_y_major_axis(shape):
    mesh = build(shape, MeshSettings())

    entity = mesh.entities(dim=shape.dimension)[0]
    assert entity.bbox == pytest.approx((0, 0, 3, 2, 4, 3), abs=1e-6)


@pytest.mark.parametrize("shape_type", [Circle, Disk], ids=["circle", "disk"])
@pytest.mark.parametrize("normal_sign", [1, -1], ids=["positive", "negative"])
def test_circular_shape_in_yz_plane(shape_type, normal_sign):
    shape = shape_type((1, 2, 3), 2, normal=(normal_sign, 0, 0))
    mesh = build(shape, MeshSettings())

    entity = mesh.entities(dim=shape.dimension)[0]
    assert entity.bbox == pytest.approx((1, 0, 1, 1, 4, 5), abs=1e-6)
    plane_distances = mesh.nodes.coords[:, 0] - 1
    np.testing.assert_allclose(plane_distances, 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape_type", [Ellipse, EllipticalDisk], ids=["curve", "disk"])
@pytest.mark.parametrize(("rx", "ry"), [(2, 1), (1, 2)], ids=["x-major", "y-major"])
def test_ellipse_with_explicit_frame(shape_type, rx, ry):
    shape = shape_type(
        (1, 2, 3),
        rx,
        ry,
        normal=(1, 0, 0),
        x_axis=(0, 1, 0),
    )
    mesh = build(shape, MeshSettings())

    entity = mesh.entities(dim=shape.dimension)[0]
    expected_bbox = (1, 2 - rx, 3 - ry, 1, 2 + rx, 3 + ry)
    assert entity.bbox == pytest.approx(expected_bbox, abs=1e-6)
    np.testing.assert_allclose(mesh.nodes.coords[:, 0], 1, rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape_type", [Ellipse, EllipticalDisk], ids=["curve", "disk"])
def test_ellipse_automatic_axis_fallback(shape_type):
    shape = shape_type((0, 0), 2, 1, normal=(1, 0, 0))
    mesh = build(shape, MeshSettings())

    entity = mesh.entities(dim=shape.dimension)[0]
    assert entity.bbox == pytest.approx((0, -2, -1, 0, 2, 1), abs=1e-6)


@pytest.mark.parametrize("shape_type", [Ellipse, EllipticalDisk], ids=["curve", "disk"])
def test_ellipse_in_oblique_plane(shape_type):
    center = np.array([1.0, 2.0, 3.0])
    normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    x_axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    y_axis = np.cross(normal, x_axis)
    rx, ry = 3.0, 1.0
    shape = shape_type(center, rx, ry, normal=normal, x_axis=x_axis)
    mesh = build(shape, MeshSettings())

    extents = np.sqrt((rx * x_axis) ** 2 + (ry * y_axis) ** 2)
    expected_bbox = (*(center - extents), *(center + extents))
    entity = mesh.entities(dim=shape.dimension)[0]
    assert entity.bbox == pytest.approx(expected_bbox, abs=1e-6)

    offsets = mesh.nodes.coords - center
    np.testing.assert_allclose(offsets @ normal, 0, rtol=0, atol=1e-12)
    ellipse_equation = (offsets @ x_axis / rx) ** 2 + (offsets @ y_axis / ry) ** 2
    if shape.dimension == 1:
        np.testing.assert_allclose(ellipse_equation, 1, rtol=0, atol=1e-8)
    else:
        assert np.all(ellipse_equation <= 1 + 1e-8)


def test_primitive_group_labeling():
    mesh = build(Box((0, 0, 0), 1, 1, 1, label="box"), MeshSettings())
    assert len(mesh.physical_group("box", 3).entities) == 1
