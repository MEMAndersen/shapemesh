# shapemesh

[![CI](https://github.com/MEMAndersen/shapemesh/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MEMAndersen/shapemesh/actions/workflows/ci.yml)

A Gmsh-powered geometry and meshing library: a small, closed set of 0–3D
shapes, n-ary booleans, and entity-level MSH 4.1 I/O.

> **Status:** v0.1.0 — initial release.

## Why shapemesh?

shapemesh is an easy-to-use wrapper for the powerful Gmsh meshing library. It
simplifies creating complex geometries and meshing them into three steps:
create primitive 1D to 3D shapes using the `Shape` primitives — for example a
`Line`, `Polygon`, or `Box`; combine them into more complex shapes with boolean
operations such as `Union`; and label them. Labels become Gmsh physical groups,
so after calling `build()` you can query the elements belonging to each labelled
shape.

## Features

- **12 shape primitives:** `Point`, `Line`, `Polyline`, `Circle`, `Ellipse`,
  `Polygon`, `Rectangle`, `Disk`, `EllipticalDisk`, `Sphere`, `Box`, `Cylinder`.
  Missing something? Open an issue — the v1 set is intentionally small.
- **n-ary booleans:** `Union`, `Intersection`, `Difference` with variadic or
  sequence constructors.
- **Overlapping physical groups:** labels are preserved across Gmsh fragment
  remapping, so an intersection volume can belong to multiple groups.
- **MSH 4.1 ASCII I/O:** read and write entity-classified meshes, with or
  without Gmsh running.
- **Typed public API:** NumPy-based data model, PEP 561 `py.typed` marker, and
  strict Python ≥3.12 typing.

## Install

```bash
pip install shapemesh
```

Gmsh (≥4.0) is installed automatically as a runtime dependency. shapemesh is
tested against Gmsh 4.15.2.

## Quick start

```python
from shapemesh import Box, MeshSettings, build

column = Box(origin=(0, 0, 0), dx=1, dy=1, dz=1, label="column")
mesh = build(shapes=column, settings=MeshSettings())
mesh.write_msh("column.msh")
```

## How it works

shapemesh builds a small scene graph of immutable shapes, hands it to Gmsh, and
then remaps the resulting entities back to your labels.

1. **Define shapes.** Every primitive takes a label. Labels are not just
   metadata — they become Gmsh physical groups.
2. **Build.** `build()` realizes the geometry in a fresh Gmsh model, sets mesh
   settings, and generates the mesh.
3. **Fragment and remap.** When shapes overlap, Gmsh fragments them into
   conformal pieces. shapemesh tracks which output pieces came from which input
   label and rebuilds the physical groups. A piece that lies in the overlap of
   two shapes belongs to **both** groups.

```python
from shapemesh import Box, MeshSettings, build

a = Box(origin=(0, 0, 0), dx=1, dy=1, dz=1, label="a")
b = Box(origin=(0.5, 0, 0), dx=1, dy=1, dz=1, label="b")
mesh = build(shapes=[a, b], settings=MeshSettings())

# Three volume pieces: a-only, overlap, b-only.
assert len(mesh.entities(dim=3)) == 3

# The overlap piece is in both physical groups.
shared = set(mesh.physical_group(name="a", dim=3).entities) & set(
    mesh.physical_group(name="b", dim=3).entities
)
assert len(shared) == 1
```

This fragment remap is what makes overlapping physical groups work without any
manual entity bookkeeping.

## Examples

Examples covering structural workflows and element-node data access live in
[`examples/`](https://github.com/MEMAndersen/shapemesh/tree/main/examples):

- [`examples/beam_with_hole.py`](https://github.com/MEMAndersen/shapemesh/blob/main/examples/beam_with_hole.py)
- [`examples/frame.py`](https://github.com/MEMAndersen/shapemesh/blob/main/examples/frame.py)
- [`examples/pilecap.py`](https://github.com/MEMAndersen/shapemesh/blob/main/examples/pilecap.py)
- [`examples/element_coordinates.py`](https://github.com/MEMAndersen/shapemesh/blob/main/examples/element_coordinates.py)
- [`examples/block_views.py`](https://github.com/MEMAndersen/shapemesh/blob/main/examples/block_views.py)

## Documentation

- [Design baseline](https://github.com/MEMAndersen/shapemesh/blob/main/design-v1.md)
- [Labels and physical groups](https://github.com/MEMAndersen/shapemesh/blob/main/docs/labels.md)
- [Thread safety](https://github.com/MEMAndersen/shapemesh/blob/main/docs/threading.md)
- [Gmsh version policy](https://github.com/MEMAndersen/shapemesh/blob/main/docs/version-policy.md)

## License

shapemesh is licensed under the **GNU Lesser General Public License v3 or later**
(LGPL-3.0-or-later). See [`COPYING`](https://github.com/MEMAndersen/shapemesh/blob/main/COPYING) and
[`COPYING.LESSER`](https://github.com/MEMAndersen/shapemesh/blob/main/COPYING.LESSER) for the full texts.

**Note on Gmsh:** shapemesh itself is LGPL-3.0-or-later. It requires Gmsh as a
runtime dependency, and Gmsh is licensed under GPL-2.0-or-later. If you
distribute a combined work that includes Gmsh — for example, a closed-source
application shipped with shapemesh and Gmsh — Gmsh's copyleft obligations apply
to that distribution. Internal or server-side use without distribution is not
affected.
