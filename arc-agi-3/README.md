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

## Results (v1)
| Check | Result |
|---|---|
| Local, 25 public games, 8000 actions/game, 3 seeds | 0.555 / 0.801 / 0.751 → **mean 0.702** |
| Local notebook run (offline path, 300→2500 actions) | 0.555, 17 levels, 938 s, 321 MB |
| Mock-gateway rerun path (competition mode, REST) | scorecard open → 25 games → close OK; 0.554 at 600 actions |
| Kaggle commit run (v1) | COMPLETE, 0.555 (identical to local), 1033 s |
| Kaggle submission | 56704680, submitted 2026-09-30 10:23 UTC — **public LB 0.24** |

Ablations (3-seed means): without avatar model 0.680; with it 0.696; + hidden-state/undo/memory fixes 0.702.
Honest expectation: LLM-harness entries score ~10–22 locally and ~3–7 on the LB; this CPU agent is far
below that and is best used as a robust baseline / exploration core for an LLM version.
