# Labels and physical groups

A `label` on a shape becomes a Gmsh **physical group** in the output mesh.

## Keying

Labels are keyed by `(label, dimension)`:

- Two shapes with the **same label and dimension** merge into one physical group.
- The **same label at different dimensions** produces separate groups (e.g. a
  surface and a volume can both be labelled `"wall"` without clashing).

## Overlapping shapes

When labelled shapes overlap, `shapemesh` fragments them so the shared region
becomes its own entity:

1. The overlap is split into a distinct piece.
2. That piece is written into **every** group whose shape covers it.

So an entity can belong to multiple physical groups, which is what MSH 4.1's
`$Entities` physical-tag list represents.

## Booleans consume labels

A boolean result only carries its *own* label. The labels of its operands are
consumed and silently dropped:

```python
# "a" and "b" are gone; only "result" survives.
build(Union(Cube(...), Cube(...), label="result"), settings)
```

## Label rules

- Labels are stripped of surrounding whitespace (a `UserWarning` is emitted when
  stripping changes the value).
- Non-empty, at most 127 characters, and must not contain quotes or newlines.
- Invalid labels raise `LabelError`.
