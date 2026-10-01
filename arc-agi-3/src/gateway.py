"""Mock Kaggle gateway: serves the 25 public games in competition mode on :8001."""
import os, json, logging
os.environ["ONLY_RESET_LEVELS"] = "true"
logging.disable(logging.INFO)
import arc_agi
def on_close(sc):
    print("GATEWAY SCORECARD CLOSED score=%.3f levels=%d/%d actions=%d" % (sc.score, sc.total_levels_completed, sc.total_levels, sc.total_actions), flush=True)
    for e in sc.environments:
        b = max(e.runs, key=lambda r: r.levels_completed)
        print("  ", e.id, round(e.score, 2), b.levels_completed, b.actions, flush=True)
a = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=os.environ.get("ARC_ENV_DIR", "environment_files"))
a.listen_and_serve(host="127.0.0.1", port=8001, competition_mode=True, on_scorecard_close=on_close)
