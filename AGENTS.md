# AGENTS.md

Guidance for contributors and AI agents working in this repository.

## Design

The source of truth for the current scope is `design-v1.md`. Keep it up
to date as you implement within its scope.

## License

The project is LGPL-3.0-or-later. The license is split across two files because
the LGPL is the GPL plus additional permissions:

- `COPYING` — the GNU GPL-3.0 text.
- `COPYING.LESSER` — the LGPL-3.0 supplement.

Every new source file must start with the full notice below (copyright holder:
Mads Kruse). Files under `examples/` may instead use the short notice shown
after it, so documentation-focused examples stay concise:

```text
# Copyright (C) 2026 Mads Kruse
#
# This file is part of shapemesh.
#
# shapemesh is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# shapemesh is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with shapemesh. If not, see <https://www.gnu.org/licenses/>.
```

The permitted short form for files under `examples/` is:

```text
# Copyright (C) 2026 Mads Kruse
# SPDX-License-Identifier: LGPL-3.0-or-later
```

## Docstrings

Public API docstrings use **Google style**:

- A one-line summary, then a blank line, then sections.
- Sections are ``Args:``, ``Returns:`` (or ``Yields:``), ``Raises:``, and
  ``Examples:``. Each entry is ``name: description``, indented 4 spaces, with
  continuation lines indented to align under the description.
- Omit the type in ``Args``/``Returns`` when the signature already annotates it.
- Cross-reference project objects with single backticks (`` `Mesh` ``) and
  inline code with double backticks (`` ``.msh`` ``). Do not use Sphinx roles
  (``:class:``, ``:meth:``, ``:func:``).
- Document every public class, function, method, and property. Include a
  ``Raises:`` section whenever it can raise.

```python
def build(shapes, settings):
    """Realize ``shapes`` with Gmsh and return the resulting `Mesh`.

    Args:
        shapes: A shape or a sequence of shapes to mesh.
        settings: The mesh settings.

    Returns:
        The entity-classified mesh.

    Raises:
        BuildError: If a Gmsh call fails.
    """
```

## Commit messages

We follow the spirit of [Common Changelog](https://common-changelog.org/):
commits and changelogs are written for humans, not machines.

- Write the subject in the **past tense, active voice**, leading with a verb:
  `Added ...`, `Fixed ...`, `Refactored ...`, `Removed ...`, `Documented ...`,
  `Bumped ...`.
- Capitalize the first word; no trailing period.
- Do **not** use Conventional Commits prefixes (`feat:`, `fix:`, `chore:`).
- Keep the subject to a single line; put the "why" and detail in the body.
- Lead with `Breaking:` when a change is breaking.

## Remote pushes

Never push commits, tags, or branches to a remote without explicit approval.
Prepare the commit locally, show it for review, and wait for the user to approve
the push. Approval applies only to the reviewed changes. Do not rewrite
published history or force-push without separate explicit approval.

## Changelog

`CHANGELOG.md` follows [Common Changelog](https://common-changelog.org/) and is
updated when a release is prepared, not with every commit. Do not add an
`Unreleased` section.

- Start each release with `## [VERSION] - YYYY-MM-DD` and link the version to
  its GitHub release with a reference-style link at the bottom of the file.
- Add a one-sentence italic notice when useful. First releases use
  `_First release._`.
- Group changes under `### Changed`, `### Added`, `### Removed`, and
  `### Fixed`, in that order, and omit empty groups.
- Write each change in the imperative mood on one line, starting with a verb
  such as `Add`, `Fix`, `Remove`, or `Refactor`.
- End each change with one or more Markdown links to the most relevant commits,
  for example
  ``([`53bd922`](https://github.com/MEMAndersen/shapemesh/commit/53bd922))``.
- Add the author after the references, for example `(Mads Kruse)`.
- Follow [3.4. Merge related changes](https://common-changelog.org/#34-merge-related-changes):
  merge commits that implement one change and leave out maintenance that does
  not affect users. Keep the list short.
- Prefix breaking changes with `**Breaking:**` and list them first.
- Keep releases latest-first, and tag each release with the same version
  (`vX.Y.Z`).
