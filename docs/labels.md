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

## Booleans, transforms, extrusions, and paths consume labels

A boolean, transform, or extrusion result only carries its *own* label. The
labels of its operands are consumed and silently dropped:

```python
# "a" and "b" are gone; only "result" survives.
build(Union(Box(...), Box(...), label="result"), settings)

# "inner" is gone; only "outer" survives.
build(Translate(Box(..., label="inner"), (1, 0, 0), label="outer"), settings)

# "profile" and "guide" are gone; only "swept" survives.
build(
    ExtrudeAlong(
        Rectangle(..., label="profile"),
        Line(..., label="guide"),
        label="swept",
    ),
    settings,
)
```

`Path` consumes the labels of its segments. Its own label applies to every
realized curve, so a labelled path becomes one physical group with one entity
per segment:

```python
# Segment labels are gone; all curves join the "guide" group.
guide = Path(
    Line(..., label="segment"),
    CircleArc(..., label="segment"),
    label="guide",
)
```

When a `Path` is used as an `ExtrudeAlong` guide, the path label is consumed as
well; only the extrusion's own label survives.

## Label rules

- Labels are stripped of surrounding whitespace (a `UserWarning` is emitted when
  stripping changes the value).
- Non-empty, at most 127 characters, and must not contain quotes or newlines.
- Invalid labels raise `LabelError`.
