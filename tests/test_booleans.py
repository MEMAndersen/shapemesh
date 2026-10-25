import numpy as np
import pytest

from shapemesh import (
    Cube,
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


def test_intersection_of_overlapping_cubes():
    a = Cube((0, 0, 0), 1, 1, 1)
    b = Cube((0.5, 0, 0), 1, 1, 1)
    mesh = build(Intersection(a, b), MeshSettings())
    assert len(mesh.entities(dim=3)) == 1


def test_difference_of_overlapping_cubes():
    a = Cube((0, 0, 0), 1, 1, 1)
    b = Cube((0.5, 0, 0), 1, 1, 1)
    mesh = build(Difference(a, b), MeshSettings())
    assert len(mesh.entities(dim=3)) == 1


def test_cube_minus_sphere_volume():
    cube = Cube((0, 0, 0), 1, 1, 1)
    sphere = Sphere((0.5, 0.5, 0.5), 0.2)
    mesh = build(Difference(cube, sphere), MeshSettings(characteristic_length=0.05))
    assert _volume(mesh) == pytest.approx(0.966, abs=0.02)


def test_nested_difference_then_union():
    a = Cube((0, 0, 0), 1, 1, 1)
    b = Cube((0.5, 0, 0), 1, 1, 1)
    c = Cube((2, 0, 0), 1, 1, 1)
    mesh = build(Union(Difference(a, b), c), MeshSettings())
    assert len(mesh.entities(dim=3)) == 2


def test_line_in_cube_mixed_dimension():
    cube = Cube((0, 0, 0), 2, 2, 2, label="cube")
    line = Line((0.5, 1, 1), (1.5, 1, 1), label="line")
    mesh = build([cube, line], MeshSettings())
    assert len(mesh.entities(dim=3)) == 1
    assert mesh.physical_group("cube", 3).entities
    assert mesh.physical_group("line", 1).entities
