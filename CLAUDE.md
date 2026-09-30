# CLAUDE.md

## Kaggle
- Use the Kaggle MCP for competitions when it is connected. In cloud sessions
  it usually is not; fall back to the `kaggle` CLI (`pip install kaggle`).
- Auth setup and known gotchas: see `kaggle-account-and-api-setup.md`.
- Check `env | grep -i kaggle` (names only) before assuming credentials exist.
  If missing, use the OAuth login flow described in that file.
- Never print, log or commit tokens, `KAGGLE_KEY` or `kaggle.json`.
- Do not run write operations (submit, push kernels, create/version datasets)
  without the user asking for them.
- Data downloads go to the scratchpad or `/tmp`, not into the repo.

## Working style
- Be brutally honest: verify current information, challenge assumptions,
  name weaknesses and existing solutions, and favor the most technically
  effective approach.
