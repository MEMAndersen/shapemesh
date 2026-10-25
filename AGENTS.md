# AGENTS.md

Guidance for contributors and AI agents working in this repository.

## Design

The source of truth for the current scope is `.plans/design-v1.md`. Keep it up
to date as you implement within its scope.

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
