# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`; do not combine multiple tickets into one file.
- Record the issue's current state in a `Status:` line near the top, using the state labels in `docs/agents/triage-labels.md`.
- Track ownership separately with an optional `Assignee:` line; an issue without an assignee is unclaimed.
- Track completion separately with a required `Resolution: open` or `Resolution: resolved` line; new issues start as `open`.
- Append comments and conversation history under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or issue number directly.

## Wayfinding

- **Map:** `.scratch/<effort>/map.md`, containing Notes, Decisions-so-far, and Fog.
- **Child ticket:** `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`; a `Type:` line records `research`, `prototype`, `grilling`, or `task`.
- **Blocking:** use a `Blocked by: NN, NN` line. A ticket is unblocked when all listed tickets have `Resolution: resolved`.
- **Frontier:** scan for open, unblocked, unclaimed issues; the lowest number goes first.
- **Claim:** set `Assignee: <name>` before beginning work; do not change the triage `Status:`.
- **Resolve:** add the answer under `## Answer`, set `Resolution: resolved`, then update the map's Decisions-so-far.
