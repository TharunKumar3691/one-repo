# HANDOFF — full context to continue in a new conversation

Paste this into a new session: *"Read HANDOFF.md, arc-agi-3/PLAN.md and arc-agi-3/README.md in repo
TharunKumar3691/one-repo (branch claude/stoic-ramanujan-w9vd7l) and continue."*

## 1. Account & environment setup (done 2026-09-30)
- Kaggle account: **tharunkumar369**. Kaggle CLI 2.2.4 works in Claude Code cloud sessions.
- Auth options (details: `kaggle-account-and-api-setup.md`):
  - Env secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` were added by the user but were **not visible** in the
    session (secrets only load in a brand-new session; verify with `env | grep -i kaggle`).
  - Working fallback: `kaggle auth login --no-launch-browser` fed through a FIFO (see setup doc). Scope
    `resources.admin:*`; credentials vanish with the container. Revoke at Kaggle → Settings → Authorized apps.
- Kaggle MCP is **not** connected in cloud sessions; CLI is used instead.
- Gotcha: `kaggle competitions submissions <slug>` → 400 if you never joined that competition.
- Repo: `main` was created from this branch (commit adecbf6). Draft PR: TharunKumar3691/one-repo#1.

## 2. Titanic (finished, files deleted from repo by request)
- Private notebook `tharunkumar369/titanic-feature-eng-ensemble` v2 still exists on Kaggle.
- Submission 56701272: public 0.76076 (CV 0.834; FE + gb/hgb/rf soft vote).

## 3. ARC Prize 2026 – ARC-AGI-3 (main work)
### Competition facts (verified via API; full text in `arc-agi-3/reference/competition-pages/`)
- Interactive agent comp; rerun plays **110 hidden games** via gateway `http://gateway:8001/`
  (API key `test-key-123`); 55 public-LB / 55 private-LB. Gateway writes the submission file.
- Limits: 9 h CPU or GPU, no internet, **1 submission/day**, external public data/models allowed.
- Deadlines: entry/team merge 2026-10-26, final 2026-11-02, winners 2026-12-04. Prizes $850k;
  open source (CC-BY 4.0) required for prizes.
- Scoring: level = min(human/agent actions,1)^2; game = level-index-weighted mean incl. unsolved = 0;
  total = mean over games. RESET = level reset in competition and **counts as an action**.
- Data: 25 public games as Python sources + human baselines (`environment_files/`), `arc_agi_3_wheels/`
  (arc_agi 0.9.8, arcengine 0.9.3, cp312), ARC-AGI-3-Agents repo. ACTION7 behaves as UNDO in public games.
- Local scoring quirk: set `ONLY_RESET_LEVELS=true` only *after* `arcade.make()`, else no scorecard play.

### Landscape (forum, `arc-agi-3/reference/forum/`)
- Top LB 45.33 (Tufa Labs). Rank ~100 ≈ 4.8, rank 25 ≈ 7.9.
- Nearly all public strong notebooks = "Duck"/TAAF LLM harness (Qwen3.8 via vLLM, BYOD docker image,
  RTX Pro 6000). Public-game ~10–22 → LB ~3–7, very noisy.
- User history: ~31 earlier submissions, best **3.64** (v31, third-party Duck baseline), v44 2.43.

### What was built (v1, symbolic, user chose "symbolic, submit today")
- Code: `arc-agi-3/src/agent.py` (Explorer), `runner.py`, `build_nb.py`, harnesses, mock `gateway.py`.
- Notebook: Kaggle `tharunkumar369/arc-agi-3-symbolic-graph-explorer` **v1, private, CPU**.
- Local public score: 3-seed mean 0.702 (avatar ablation 0.680→0.696→0.702).
- **Submission 56704680 → public LB 0.24.**
- Version history of the agent: v0 naive graph 0.159 → masked change detection 0.350 →
  cost-aware planner/move-limit/undo 0.535 → avatar/target model 0.70 (3-seed).

### Key lessons
- Blind exploration costs 5–50× human actions; squared scoring kills it. Public→hidden ratio ≈ 0.35.
- Per-run noise is huge; compare with ≥3 seeds.
- To beat 3.64 an LLM goal-inference harness is required; the symbolic agent is a good
  explorer/fallback core for it.

## 4. Recommended next steps
1. Build own LLM harness (public Qwen3.8 weights + vLLM runtime), symbolic explorer as fallback;
   debug in a *separate* private GPU notebook, then publish the final notebook as a fresh v1.
2. Re-download data each session: `kaggle competitions download arc-prize-2026-arc-agi-3`
   (data is not committed — competition license/size).
3. Local env: `python3.12 -m venv venv && venv/bin/pip install --no-index --find-links
   arc_agi_3_wheels arc-agi scipy`; benchmark: `ARC_ENV_DIR=... venv/bin/python arc-agi-3/src/pharness.py agent 8000 60`.

## 5. Not preservable
- The verbatim chat transcript, temporary container files (`/tmp`), OAuth login, downloaded data.
  Everything else is in this repo.
