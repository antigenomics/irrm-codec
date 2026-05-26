# Working Notes

## Git workflow

- When asked to save work for later runs, create a commit with only the files relevant to the current task.
- Push the current branch to `origin`.
- If the current branch is not suitable, create a dedicated branch before committing.
- Never include unrelated local changes in the commit.

## Run command handoff

- For each completed step, provide the exact command to run from the repository root.
- Prefer a single ready-to-copy command.
- If the command depends on environment variables, show them inline in the command.
