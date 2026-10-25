"""
2D example of a beam with circular holes.
"""

from shapemesh import MeshSettings, build, shapes

length = 3.0  # m
height = 1.0  # m

hole_diameter = 0.5  # m
number_of_holes = 2

support_width = 0.2  # m

element_size = 0.1

# Create shapes
gross_beam = shapes.Rectangle(origin=[0, 0], dx=length, dy=height)
holes = [
    shapes.Disk(center=[(i + 0.5) * length / number_of_holes, height / 2], radius=hole_diameter / 2)
    for i in range(number_of_holes)
]
left_support = shapes.Line(start=(0.0, 0.0), end=(support_width, 0.0), label="left_support")
symmetry_line = shapes.Line(start=(length, 0.0), end=(length, height), label="symmetry_line")
load_line = shapes.Line(start=(0.0, height), end=(length, height), label="load_line")

# Create the holes in the gross beam using difference
beam = shapes.Difference([gross_beam, *holes], label="beam")

# create the mesh
mesh = build(
    [beam, left_support, symmetry_line, load_line],
    settings=MeshSettings(element_size=element_size),
)

print(mesh)
for group in mesh.groups:
    print(group)
mesh.write_msh(path="beam_with_hole.msh")
