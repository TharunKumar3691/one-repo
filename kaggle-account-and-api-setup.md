# Kaggle account and API setup

How Kaggle CLI access is set up for this repo in Claude Code cloud sessions.
Account: `tharunkumar369`. **Never commit tokens, keys or `kaggle.json`.**

## Authentication (pick one)

### A. Environment secrets (persistent, recommended)
1. kaggle.com → Settings → API → Create New Token.
2. In the cloud environment settings (session title bar → Edit), add
   `KAGGLE_USERNAME` and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`).
3. Start a **new** session. Variables are read only at session start.
4. Verify: `pip install kaggle && kaggle competitions list`.

### B. OAuth login (per session, no secrets)
`kaggle auth login --no-launch-browser` prints a URL and waits for a code on
stdin. Run it with a FIFO so it can wait:

```sh
mkfifo /tmp/kfifo
(nohup sh -c 'sleep 1800 > /tmp/kfifo & kaggle auth login --no-launch-browser < /tmp/kfifo' > /tmp/kaggle_login.log 2>&1 &)
cat /tmp/kaggle_login.log        # open the URL, approve, copy the code
echo '<code>' > /tmp/kfifo       # feed the code
```

Scope is `resources.admin:*` (broad). Credentials vanish when the container is
reclaimed. Revoke at Kaggle → Settings → Authorized apps.

## Verified working (2026-09-30)
Competitions (list/search/files/leaderboard/download/submissions), datasets
(list/files/download/mine), notebooks (public + mine), models (list).
Write operations (submit, push kernel, create dataset) were not tested.

## Gotchas
- `kaggle competitions submissions <slug>` returns **400 Bad Request** for a
  competition you have not joined (`userHasEntered: False`). Join it in the
  browser first. It is not an auth or CLI bug.
- Competition rules must be accepted once in the browser before download or
  submit.
- Discussions/forums are not available through the CLI or API.
- Default `competitions list` sorts by prize and includes old competitions. Use
  `--sort-by latestDeadline`, `-s <term>` or `--group entered`.
- The container is ephemeral: keep downloaded data out of the repo.
