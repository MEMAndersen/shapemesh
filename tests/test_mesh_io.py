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

from shapemesh import ElementType, Mesh

MINIMAL_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "tri"
$EndPhysicalNames
$Entities
3 3 1 0
1 0 0 0 0
2 1 0 0 0
3 0 1 0 0
1 0 0 0 1 0 0 0 2 1 2
2 1 0 0 1 1 0 0 2 2 3
3 0 0 0 0 1 0 0 2 3 1
1 0 0 0 1 1 0 1 1 3 1 2 3
$EndEntities
$Nodes
1 3 1 3
2 1 0 3
1
2
3
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
1 1 1 1
2 1 2 1
1 1 2 3
$EndElements
"""

UNKNOWN_TYPE_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 0 0
$EndEntities
$Nodes
1 4 1 4
0 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
0 0 1
$EndNodes
$Elements
1 2 1 2
0 1 999 2
1 1 2 3
2 4 1 2
$EndElements
"""


def test_read_minimal(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    assert mesh.num_nodes == 3
    assert mesh.num_elements == 1
    assert len(mesh.entities()) == 7  # 3 points + 3 curves + 1 surface
    assert mesh.bbox == (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    group = mesh.physical_group("tri", 2)
    assert group.entities == ((2, 1),)
    blocks = mesh.elements_in_group("tri", 2)
    assert len(blocks) == 1
    assert blocks[0].type == ElementType.TRI3
    assert blocks[0].connectivity.tolist() == [[1, 2, 3]]


def test_roundtrip(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    out = tmp_path / "out.msh"
    mesh.write_msh(out)
    mesh2 = Mesh.read_msh(out)

    assert mesh2.num_nodes == mesh.num_nodes
    assert mesh2.num_elements == mesh.num_elements
    assert len(mesh2.entities()) == len(mesh.entities())
    assert mesh2.physical_group("tri", 2).entities == mesh.physical_group("tri", 2).entities


def test_to_numpy(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    result = mesh.to_numpy()
    assert result.coords.shape == (3, 3)
    assert ElementType.TRI3 in result.blocks
    block = result.blocks[ElementType.TRI3]
    assert block.connectivity.tolist() == [[0, 1, 2]]
    assert block.node_tags.tolist() == [[1, 2, 3]]


def test_unknown_element_type_skipped(tmp_path):
    path = tmp_path / "unknown.msh"
    path.write_text(UNKNOWN_TYPE_MSH, encoding="utf-8")
    with pytest.warns(UserWarning, match="999"):
        mesh = Mesh.read_msh(path)
    assert mesh.num_nodes == 4
    assert mesh.num_elements == 0


def test_public_arrays_are_read_only(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)
    with pytest.raises(ValueError):
        mesh.nodes.coords[0, 0] = 999.0


def test_gmsh_roundtrip(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)  # keep the test output quiet
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        path = tmp_path / "box.msh"
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    mesh = Mesh.read_msh(path)
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert mesh.bbox == pytest.approx((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), abs=1e-6)

    out = tmp_path / "box2.msh"
    mesh.write_msh(out)
    mesh2 = Mesh.read_msh(out)
    assert mesh2.num_nodes == mesh.num_nodes
    assert mesh2.num_elements == mesh.num_elements
