"""Build the Kaggle notebook from the tested sources (agent.py, runner.py)."""
import json
import os
import sys

AGENT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")).read()
RUNNER = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")).read()
OUT = sys.argv[1] if len(sys.argv) > 1 else "notebook/arc-agi-3-symbolic-explorer.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


INTRO = r"""
# ARC-AGI-3 — Symbolic Graph Explorer (v1, CPU only)

A from-scratch agent that needs no model. For each game it plays online through the Kaggle gateway,
learns what the actions do, and searches each level's state space at minimum action cost.

**How it is scored.** Per level: `min(human_actions / agent_actions, 1)²`. Per game: level scores averaged
with weights 1, 2, 3, … by level index; unsolved levels count 0. RESET counts as an action. So the agent must
waste as few actions as possible on the levels it does solve, and keep playing for deeper levels.

**Agent**
1. *Perception* — the last frame of each action. Step counters and progress bars (small one-way changes in
   the edge rows/columns) are detected and masked, so revisited states are recognised.
2. *State graph* — nodes are masked-frame hashes; edges are actions; GAME_OVER edges are marked deadly unless
   the death happened at a new record length (then it is a move limit and plans are kept within it).
3. *Candidates* — simple actions, plus ACTION6 clicks on one pixel per colour component (UI masked out).
4. *Priors* — per action and per clicked colour: how often it actually changed the (masked) frame.
5. *Avatar model* — colours that move under actions, the mean displacement per action, and rare static
   objects as candidate targets; moves towards unvisited targets are preferred. The colour of the object
   reached when a level is won is carried to later levels.
6. *Planning* — choose the untried action with the best `score / (cost + 1)`, where reaching it may walk known
   edges, use ACTION7 once it has been verified to act as UNDO, or spend one RESET to return to the level start.
7. *Scheduling* — all games in parallel threads, in rounds of growing action budgets, under a global deadline.

In an interactive/commit run (no gateway) the notebook plays the 25 public games offline and prints the
local score with the official scorer. In the competition rerun it plays the hidden games via the gateway;
the submission file is produced by the gateway.
"""

SETUP = r"""
import os, sys, time, glob, json, subprocess, logging
NOTEBOOK_START = time.time()
IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
if not os.path.isdir(COMP_DIR):
    hits = glob.glob("/kaggle/input/**/arc_agi_3_wheels", recursive=True)
    COMP_DIR = os.path.dirname(hits[0]) if hits else COMP_DIR
print("rerun:", IS_RERUN, "| competition data:", COMP_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-warn-conflicts",
                "--disable-pip-version-check", "--find-links", f"{COMP_DIR}/arc_agi_3_wheels", "arc-agi"],
               check=True)
os.environ.setdefault("MPLBACKEND", "Agg")
logging.getLogger("arc_agi").setLevel(logging.WARNING)
import arc_agi
print("arc_agi", getattr(arc_agi, "__version__", "?"), "ready in", round(time.time() - NOTEBOOK_START, 1), "s")
"""

MAIN = r"""
import pandas as pd
from urllib.request import urlopen

RUN_LIMIT_S = float(os.getenv("ARC_RUN_LIMIT_S", 8 * 3600))  # stay well inside the 9 h limit
LOCAL_BUDGETS = [300, 1000, 2500]  # short offline check of the 25 public games
logging.disable(logging.INFO)

def log(msg):
    print(f"[{time.time() - NOTEBOOK_START:7.0f}s] {msg}", flush=True)

def wait_for_gateway(url, timeout_s=900):
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urlopen(url + "api/games", timeout=10) as r:
                if r.status < 500:
                    return True
        except Exception as e:
            last = repr(e)
        time.sleep(5)
    raise RuntimeError("gateway not ready")

if IS_RERUN:
    base = os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    key = os.environ.setdefault("ARC_API_KEY", "test-key-123")
    wait_for_gateway(base)
    arcade = arc_agi.Arcade(arc_api_key=key, arc_base_url=base.rstrip("/"),
                            operation_mode=arc_agi.OperationMode.COMPETITION, environments_dir="")
    card = arcade.open_scorecard(tags=["symbolic-explorer-v1"])
    deadline = NOTEBOOK_START + RUN_LIMIT_S
    try:
        explorers, status = run_all(arcade, Explorer, deadline, workers=16, card_id=card, log=log)
        log(f"final status: {json.dumps({k: v for k, v in status.items()})[:3000]}")
    finally:
        try:
            arcade.close_scorecard(card)
        except Exception as e:
            log(f"close_scorecard: {e!r}")
    log("done")
else:
    os.environ.pop("ONLY_RESET_LEVELS", None)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=f"{COMP_DIR}/environment_files")
    card = arcade.open_scorecard()
    ids = sorted(e.game_id for e in arcade.available_environments)
    # make() starts each game with a full reset; afterwards RESET behaves as in the competition (level reset)
    envs = {g: arcade.make(g, scorecard_id=card) for g in ids}
    os.environ["ONLY_RESET_LEVELS"] = "true"
    class _Pre:  # hand the pre-made envs to the runner
        available_environments = arcade.available_environments
        def make(self, gid, scorecard_id=None):
            return envs[gid]
    deadline = time.time() + 1500
    explorers, status = run_all(_Pre(), Explorer, deadline, workers=4, card_id=card,
                                budgets=LOCAL_BUDGETS, log=log, game_ids=ids)
    sc = arcade.get_scorecard(card)
    rows = []
    for es in sc.environments:
        best = max(es.runs, key=lambda r: r.levels_completed)
        rows.append((es.id, round(es.score, 2), best.levels_completed, best.actions))
    print(pd.DataFrame(rows, columns=["game", "score", "levels", "actions"]).to_string(index=False))
    log(f"LOCAL PUBLIC SCORE (25 games, short budget): {sc.score:.3f}")
    # interactive run: write the placeholder file so the notebook can be submitted
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]) \
        .to_parquet("/kaggle/working/submission.parquet", index=False)
    log("wrote placeholder submission.parquet")
"""

cells = [
    md(INTRO),
    md("## 1. Setup — install the ARC runtime from the competition wheels (no internet)"),
    code(SETUP),
    md("## 2. Agent"),
    code(AGENT),
    md("## 3. Scheduler"),
    code(RUNNER),
    md("## 4. Run — hidden games via the gateway (rerun) or the 25 public games offline"),
    code(MAIN),
]
for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT, sum(len("".join(c["source"])) for c in cells), "chars")
