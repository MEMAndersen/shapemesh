# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Sweep a rebar as one solid with a straight segment and a circular hook."""

import math

from shapemesh import CircleArc, Disk, ExtrudeAlong, Line, MeshSettings, Path, build

# The guide runs up the bar, bends 90 degrees around +y, and continues along +x.
# Each segment starts where the previous one ends with the same tangent, so the
# sweep is smooth and produces a single solid.
guide = Path(
    Line((0, 0, 0), (0, 0, 1.0)),
    CircleArc(
        center=(0.25, 0, 1.0),
        start=(0, 0, 1.0),
        angle=math.pi / 2,
        normal=(0, 1, 0),
    ),
    Line((0.25, 0, 1.25), (0.75, 0, 1.25)),
)

rebar = ExtrudeAlong(
    Disk(center=(0, 0, 0), radius=0.05, normal=(0, 0, 1)),
    guide,
    label="rebar",
)

mesh = build(rebar, MeshSettings(element_size=0.02))

print(mesh)
for group in mesh.groups:
    print(group)

mesh.write_msh("rebar_hook.msh")
