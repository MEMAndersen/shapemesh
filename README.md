# shapemesh

A Gmsh-powered geometry and meshing library: a closed set of 0–3D shapes,
n-ary booleans, and entity-level MSH 4.1 I/O.

> **Status:** v0.1.0 in development.

## Features

- 12 shape primitives: Point, Line, Polyline, Circle, Ellipse, Polygon, Rectangle, Disk, EllipticalDisk, Sphere, Box, Cylinder
- n-ary boolean `Union` / `Intersection` / `Difference`
- Overlapping physical groups via fragment remap
- MSH 4.1 ASCII read/write (no live Gmsh session required for I/O)

## Install

```bash
pip install shapemesh
```

## Quick start

```python
from shapemesh import Box, MeshSettings, build

mesh = build(Box((0, 0, 0), 1, 1, 1, label="column"), MeshSettings())
mesh.write_msh("column.msh")
```

## Examples

Structural examples (1D, 2D, and 3D) live in [`examples/`](examples/) — coming
soon.

## Documentation

- [Labels and physical groups](docs/labels.md)
- [Thread safety](docs/threading.md)
- [Gmsh version policy](docs/version-policy.md)

## License

LGPL-3.0-or-later. Gmsh — a runtime dependency — is GPLv2-or-later.
