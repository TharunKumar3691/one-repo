# First prompt for the new Claude Cowork conversation

Copy everything between the lines into the new conversation (with this folder selected):

---
This folder contains the full handoff of my previous work on the Kaggle competition
"ARC Prize 2026 – ARC-AGI-3" (arc-prize-2026-arc-agi-3). My Kaggle username is tharunkumar369.

Setup and understanding, in this order:
1. Read HANDOFF.md fully, then arc-agi-3/PLAN.md, arc-agi-3/README.md, CLAUDE.md,
   kaggle-account-and-api-setup.md, and everything in arc-agi-3/reference/ (official rules,
   evaluation, code requirements, forum threads). Summarise back to me in 10 bullets what you
   understood: competition rules/scoring, my history (best LB 3.64, v1 symbolic = 0.24), and the plan.
2. Check Kaggle access: confirm the kaggle CLI works and is authenticated as tharunkumar369
   (`kaggle competitions submission-limits arc-prize-2026-arc-agi-3`). If not, guide me through auth.
   Never print or save my API key.
3. Re-download the competition data into a data/ folder (not into git):
   `kaggle competitions download arc-prize-2026-arc-agi-3 -p data && unzip -q data/*.zip -d data`
4. Create a Python 3.12 venv, install arc-agi from data/arc_agi_3_wheels (offline) plus scipy, and
   reproduce the local benchmark: `ARC_ENV_DIR=data/environment_files venv/bin/python arc-agi-3/src/pharness.py agent 8000 60`
   Expect roughly 0.55–0.80 on the 25 public games.
5. Then stop and propose a concrete plan for v2: my own LLM-harness agent (public open-weights model,
   allowed by the rules) for goal inference, with the existing symbolic explorer as fallback,
   aiming to beat my 3.64. Be brutally honest about feasibility, GPU quota, risks and timeline
   (final deadline 2026-11-02, 1 submission/day). Don't submit anything without my approval.
---
