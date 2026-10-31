# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Gather element-node coordinates from tag-based element connectivity."""

from shapemesh import Line, MeshSettings, Rectangle, build

plate = Rectangle(origin=(0, 0), dx=1, dy=1, label="plate")
diagonal = Line((0, 0), (1, 1))  # add diagonal line to cut plate into two blocks
mesh = build([plate, diagonal], MeshSettings(element_size=0.5))

plate_blocks = mesh.elements_in_group("plate", dim=2)
print(f"The 'plate' label contains {len(plate_blocks)} blocks:")
for block in plate_blocks:
    # Connectivity contains node tags, not necessarily row indices. Convert the
    # whole (M, k) block before indexing coordinates or other node-aligned data.
    node_rows = mesh.nodes.indices_for(block.connectivity)
    element_coords = mesh.nodes.coords[node_rows]

    # The one-step equivalent is:
    # element_coords = mesh.nodes.coords_for(block.connectivity)

    print(f"\nBlock for entity tag {block.entity_tag} of dimension {block.entity_dim}")
    print(f"{block.type.name}: {block.connectivity.shape[0]} elements")
    print(f"Connectivity shape (node tags): {block.connectivity.shape}")
    print(f"Row-index shape: {node_rows.shape}")
    print(f"Element-coordinate shape: {element_coords.shape}")
    print("\nFirst element:")
    print("Node tags:")
    print(block.connectivity[0])
    print("Row indices:")
    print(node_rows[0])
    print("Coordinates:")
    print(element_coords[0])
