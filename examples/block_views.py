# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Use flat element-type views and physical-group row selections."""

from shapemesh import ElementType, Line, MeshSettings, Rectangle, build

plate = Rectangle(origin=(0, 0), dx=1, dy=1, label="plate")
diagonal = Line((0, 0), (1, 1), label="diagonal")
mesh = build([plate, diagonal], MeshSettings(element_size=0.5))

# Each flat view keeps zero-based connectivity, original element tags, and the
# entity-block slices from which its rows came.
for element_type, view in mesh.blocks_by_type().items():
    print(f"\n{element_type.name}: {view.connectivity.shape[0]} elements")
    print(f"Connectivity shape (node rows): {view.connectivity.shape}")
    print(f"Element tags shape: {view.element_tags.shape}")
    for block_slice in view.slices:
        entity = (block_slice.entity_dim, block_slice.entity_tag)
        print(f"  Entity {entity}: rows {block_slice.start}:{block_slice.stop}")

# Group selections contain row indices into these cached flat views.
triangles = mesh.blocks_by_type()[ElementType.TRI3]
plate_rows = mesh.element_indices_in_group("plate", dim=2)[ElementType.TRI3]
plate_connectivity = triangles.connectivity[plate_rows]
plate_coordinates = mesh.nodes.coords[plate_connectivity]
print(f"\nPlate connectivity shape: {plate_connectivity.shape}")
print(f"Plate coordinate shape: {plate_coordinates.shape}")

# Node selections are unique, sorted row indices into the global node table.
diagonal_nodes = mesh.node_indices_in_group("diagonal", dim=1)
print(f"Diagonal node rows: {diagonal_nodes}")
print(f"Diagonal node tags: {mesh.nodes.tags[diagonal_nodes]}")
