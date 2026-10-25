"""
3D example of a pilecap with a rectangular pier surface on the top and four
circular pile interfaces on the bottom.
"""

from shapemesh import MeshSettings, build, shapes

length = 4.0  # m
width = 3.0  # m
height = 1.5  # m

column_length = 2.0  # m
column_width = 1.2  # m

pile_diameter = 0.8
pile_spacing_x = 2.5
pile_spacing_y = 1.5

element_size = 1.0  # Element size

# Define main 3d geometry
pilecap = shapes.Box(
    origin=[-length / 2, -width / 2, 0], dx=length, dy=width, dz=height, label="pilecap"
)

# Define surfaces for pile interface and column/pier interface
column_surface = shapes.Rectangle(
    origin=[-column_length / 2, -column_width / 2, height],
    dx=column_length,
    dy=column_width,
    label="column",
)
piles = []
for sign_x, sign_y in zip([-1, 1, 1, -1], [-1, -1, 1, 1], strict=True):
    piles.append(
        shapes.Disk(
            center=[sign_x * pile_spacing_x / 2, sign_y * pile_spacing_y / 2],
            radius=pile_diameter / 2,
            label="piles",
        )
    )

# Build the mesh (default for 3D is tetrahedrons with nodes at vertices)
mesh = build(
    shapes=[pilecap, column_surface, *piles],
    settings=MeshSettings(element_size=element_size),
)

print(mesh)
for group in mesh.groups:
    print(group)

mesh.write_msh("pilecap.msh")
