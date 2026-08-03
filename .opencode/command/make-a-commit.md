---
description: Analyze status, stage files, commit, and push to main
agent: build
subtask: true
---
Create a git commit for the current repository.

The language `en`: write the full commit message in English.

Required workflow:

1. Run `git status --short --branch` first.
2. Inspect the relevant diffs before staging anything.
3. Stage the intended files with `git add`.
4. Do not stage or commit secrets, generated artifacts, or unrelated work.
5. If the worktree contains ambiguous or mixed changes and intent is unclear, stop and ask the user.
6. Write a full multiline commit message in the requested language.
7. The first line must be short and must use a conventional commit tag such as `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `test:`, `build:`, or another clearly appropriate conventional-commit type.
8. After the first line, include a blank line and a structured body that explains what changed, why it changed, and any relevant implementation notes.
9. Create the commit with a multiline message, for example by using repeated `git commit -m` flags.
10. Push directly to `main` with `git push origin HEAD:main`.

Before committing, verify that the staged files match the actual change you are committing.
