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

import pytest

from shapemesh import (
    Cube,
    Cylinder,
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
    (Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), 2, 1),
    (Rectangle((0, 0, 0), 1, 1), 2, 1),
    (Sphere((0, 0, 0), 1), 3, 1),
    (Cube((0, 0, 0), 1, 1, 1), 3, 1),
    (Cylinder((0, 0, 0), (0, 0, 1), 0.5), 3, 1),
]

_IDS = ["point", "line", "polyline", "polygon", "rectangle", "sphere", "cube", "cylinder"]


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


def test_primitive_group_labeling():
    mesh = build(Cube((0, 0, 0), 1, 1, 1, label="box"), MeshSettings())
    assert len(mesh.physical_group("box", 3).entities) == 1
