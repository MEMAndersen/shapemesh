# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Create solids with linear extrusion, revolution, and a guided sweep."""

import math

from shapemesh import (
    Circle,
    Disk,
    Extrude,
    ExtrudeAlong,
    MeshSettings,
    Polygon,
    Revolve,
    build,
)

# Linear extrusion: a rectangular profile becomes a prism.
prism = Extrude(
    Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)]),
    offset=(0, 0, 2),
    label="prism",
)

# Revolution: an offset profile sweeps a half turn around the z-axis.
half_ring = Revolve(
    Polygon([(1, 3, 0), (1.5, 3, 0), (1.5, 3, 0.5), (1, 3, 0.5)]),
    angle=math.pi,
    pivot=(0, 3, 0),
    label="half_ring",
)

# Guided sweep: a disk profile follows a circular guide, producing a torus.
torus = ExtrudeAlong(
    Disk(center=(1, -3, 0), radius=0.1, normal=(0, 1, 0)),
    Circle(center=(0, -3, 0), radius=1),
    label="torus",
)

mesh = build([prism, half_ring, torus], MeshSettings(element_size=0.1))

print(mesh)
for group in mesh.groups:
    print(group)

mesh.write_msh("extrusions.msh")
