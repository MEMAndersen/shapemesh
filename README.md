# shapemesh

A Gmsh-powered geometry and meshing library: a small, closed set of 0–3D
shapes, n-ary booleans, and entity-level MSH 4.1 I/O.

> **Status:** v0.1.0 — first release in progress.

## Why shapemesh?

shapemesh is an easy to use "wrapper" for the powerful Gmsh meshing library. The process of creating complex geometries and meshing them has been simplified into three steps. Create primitive 1d to 3d shapes using the `Shape` primitives, this could for instance be a `Line`, a `Polygon`, or a `Box`, the primitive shapes can combined into more complex shapes using boolean operations such as `Union`. When a shape is created it can be labelled, this label makes sure that when the mesh is created using the `build()` function, the specific elements that this shape consists of can be queried using the label.

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

Structural examples covering 1D, 2D, and 3D workflows live in
[`examples/`](examples/):

- [`examples/beam_with_hole.py`](examples/beam_with_hole.py)
- [`examples/frame.py`](examples/frame.py)
- [`examples/pilecap.py`](examples/pilecap.py)

## Documentation

- [Design baseline](design-v1.md)
- [Labels and physical groups](docs/labels.md)
- [Thread safety](docs/threading.md)
- [Gmsh version policy](docs/version-policy.md)

## License

shapemesh is licensed under the **GNU Lesser General Public License v3 or later**
(LGPL-3.0-or-later). See [`COPYING`](COPYING) and
[`COPYING.LESSER`](COPYING.LESSER) for the full texts.

**Note on Gmsh:** shapemesh itself is LGPL-3.0-or-later. It requires Gmsh as a
runtime dependency, and Gmsh is licensed under GPL-2.0-or-later. If you
distribute a combined work that includes Gmsh — for example, a closed-source
application shipped with shapemesh and Gmsh — Gmsh's copyleft obligations apply
to that distribution. Internal or server-side use without distribution is not
affected.
