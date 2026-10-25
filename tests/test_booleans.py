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
    Difference,
    ElementType,
    Intersection,
    Line,
    MeshSettings,
    Sphere,
    Union,
    build,
)


def _volume(mesh):
    result = mesh.to_numpy()
    total = 0.0
    for etype, block in result.blocks.items():
        if etype is not ElementType.TET4:
            continue
        for row in block.connectivity:
            points = result.coords[row]
            matrix = [points[1] - points[0], points[2] - points[0], points[3] - points[0]]
            total += abs(np.linalg.det(matrix)) / 6.0
    return total


def test_intersection_of_overlapping_boxes():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    mesh = build(Intersection(a, b), MeshSettings())
    assert len(mesh.entities(dim=3)) == 1


def test_difference_of_overlapping_boxes():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    mesh = build(Difference(a, b), MeshSettings())
    assert len(mesh.entities(dim=3)) == 1


def test_box_minus_sphere_volume():
    box = Box((0, 0, 0), 1, 1, 1)
    sphere = Sphere((0.5, 0.5, 0.5), 0.2)
    mesh = build(Difference(box, sphere), MeshSettings(characteristic_length=0.05))
    assert _volume(mesh) == pytest.approx(0.966, abs=0.02)


def test_nested_difference_then_union():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    c = Box((2, 0, 0), 1, 1, 1)
    mesh = build(Union(Difference(a, b), c), MeshSettings())
    assert len(mesh.entities(dim=3)) == 2


def test_line_in_box_mixed_dimension():
    box = Box((0, 0, 0), 2, 2, 2, label="box")
    line = Line((0.5, 1, 1), (1.5, 1, 1), label="line")
    mesh = build([box, line], MeshSettings())
    assert len(mesh.entities(dim=3)) == 1
    assert mesh.physical_group("box", 3).entities
    assert mesh.physical_group("line", 1).entities
