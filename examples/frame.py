# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
1D example simple frame
"""

from shapemesh import MeshSettings, Vec3, build, shapes

wall_height = 2.0  # m
roof_height = 0.5  # m
house_width = 5.0  # m

coordinates = [
    Vec3(0.0, 0.0),
    Vec3(0.0, wall_height),
    Vec3(house_width / 2, wall_height + roof_height),
    Vec3(house_width, wall_height),
    Vec3(house_width, 0),
]

# Create the bar geometry
left_wall = shapes.Line(coordinates[0], coordinates[1], label="left_wall")
right_wall = shapes.Line(coordinates[4], coordinates[3], label="right_wall")

## By using union we can have a common label for both the walls
walls = shapes.Union(
    left_wall,
    right_wall,
    label="walls",
)
roof = shapes.Polyline(
    points=[coordinates[n] for n in [1, 2, 3]],
    label="roof",
)

# Make point shapes for the supported nodes
left_support = shapes.Point(coordinates[0], label="left_support")
right_support = shapes.Point(coordinates[4], label="right_support")


# Build the mesh by including all the shapes with labels that should survive into the mesh
mesh = build(
    [left_wall, right_wall, walls, roof, left_support, right_support],
    MeshSettings(element_size=0.2),
)

print(mesh)
for group in mesh.groups:
    print(group)

mesh.write_msh("frame.msh")
