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

Every new source file must start with this notice (copyright holder: Mads
Kruse):

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

## Changelog

A `CHANGELOG.md` in Common Changelog format will be started once `0.1.0` is
finished — not before.
