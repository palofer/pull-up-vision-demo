# Domain Docs

How engineering skills should consume this repo's domain documentation.

## Before exploring

- Read `GLOSSARY.md` at the repo root, if it exists.
- Read relevant decisions in `docs/adr/`, if they exist.
- If a referenced file does not exist, proceed without treating its absence as a blocker.

## File structure

This is a single-context repo:

- `GLOSSARY.md` holds project vocabulary.
- `docs/adr/` holds architecture decision records.
- `src/` holds the application code.

## Use the glossary's vocabulary

Use terms as defined in `GLOSSARY.md` when describing domain concepts in issues, proposals, or tests. If a needed term is missing, note the gap rather than silently inventing competing terminology.

## Flag ADR conflicts

If a proposal contradicts an existing ADR, surface the conflict explicitly rather than silently overriding it.
