# Plan mode in this repository

Implementation plans for this repo live in the tracked directory `.opencode/plan/`
(see `AGENTS.md`). Plan mode is explicitly configured to write there.

- When you need to save or update a plan document, use `.opencode/plan/<name>.md`.
- Do **not** redirect plan files to `~/.opencode/plan` or any other location.
- This override is granted by `.opencode/opencode.jsonc`, which appends an
  `edit` allow rule for `*.opencode/plan/*` after the built-in plan deny rule.
- Everything else in plan mode stays read-only: do not edit source, configs,
  tests, or docs outside `.opencode/plan/`.
