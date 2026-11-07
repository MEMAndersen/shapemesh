# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Translate and rotate the same boolean shape in the shape graph."""

import math

from shapemesh import Difference, Disk, MeshSettings, Rectangle, Rotate, Translate, build

# Create a plate with a circular hole.
plate = Difference(
    Rectangle(origin=(0, 0), dx=2, dy=1), Disk(center=(1, 0.5), radius=0.2), label="original_plate"
)

# Each transform realizes the plate independently, so the same immutable shape
# can be reused in multiple positions. Labels on the operation nodes survive.
translated_plate = Translate(plate, offset=(3, 0, 0), label="translated_plate")
rotated_plate = Rotate(
    plate,
    angle=math.pi / 2,
    pivot=(0, 0, 0),
    axis=(0, 0, 1),
    label="rotated_plate",
)

mesh = build(
    [plate, translated_plate, rotated_plate],
    settings=MeshSettings(element_size=0.15),
)

print(mesh)
for group in mesh.groups:
    print(group)

mesh.write_msh("transforms.msh")
