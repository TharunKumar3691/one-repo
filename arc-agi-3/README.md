# ARC-AGI-3 — Symbolic Graph Explorer (v1)

From-scratch, CPU-only agent for the Kaggle competition
[ARC Prize 2026 – ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).
Kaggle notebook: `tharunkumar369/arc-agi-3-symbolic-graph-explorer` (private).

See `PLAN.md` for the competition analysis and design rationale.

## Layout
| File | Purpose |
|---|---|
| `src/agent.py` | The agent (`Explorer`): perception, state graph, priors, avatar model, planner |
| `src/runner.py` | Scheduler: all games in threads, rounds of growing action budgets, global deadline |
| `src/build_nb.py` | Builds the Kaggle notebook from `agent.py` + `runner.py` (exact tested code) |
| `src/harness.py`, `src/pharness.py` | Local benchmark on the 25 public games with the official scorer (serial / 4 procs) |
| `src/gateway.py` | Mock Kaggle gateway (`arc_agi` REST server, competition mode) for end-to-end rerun tests |
| `src/exec_nb.py` | Executes the notebook locally (nbclient) |
| `notebook/` | The notebook + `kernel-metadata.json` pushed to Kaggle |

## Reproduce locally
```sh
python3.12 -m venv venv && venv/bin/pip install --no-index --find-links <comp>/arc_agi_3_wheels arc-agi && venv/bin/pip install scipy
venv/bin/python src/pharness.py agent 8000 60        # 25 public games, 8000 actions / 60 s each
```
