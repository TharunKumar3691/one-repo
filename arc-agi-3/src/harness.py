"""Local benchmark: runs an agent on the 25 public games offline, scores like the competition."""
import os, sys, time, json, logging, importlib
os.environ.pop("ONLY_RESET_LEVELS", None)
logging.disable(logging.CRITICAL)
import arc_agi
from arc_agi import EnvironmentScorecard
ENV_DIR = os.environ.get("ARC_ENV_DIR", "environment_files")

def run(play, games=None, max_actions=None, verbose=True):
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)
    envs = sorted(arcade.available_environments, key=lambda e: e.game_id)
    if games: envs = [e for e in envs if e.game_id.split('-')[0] in games]
    card = arcade.open_scorecard()
    t0 = time.time(); res = {}
    for e in envs:
        g0 = time.time()
        os.environ.pop('ONLY_RESET_LEVELS', None)
        env = arcade.make(e.game_id, scorecard_id=card)
        os.environ['ONLY_RESET_LEVELS'] = 'true'  # competition: RESET = level reset
        info = play(env, e.game_id)
        res[e.game_id] = (time.time() - g0, info)
    sc = arcade.get_scorecard(card)
    out = {}
    for es in sc.environments:
        r = es.runs[0] if hasattr(es, 'runs') else es
        best = max(es.runs, key=lambda r: r.levels_completed)
        out[es.id] = best
        if verbose:
            ls = [f"{a}/{b}" for a, b in zip(best.level_actions or [], best.level_baseline_actions or [])][:best.levels_completed + 1]
            print(f"{es.id:15s} {str(res.get(es.id,(0,''))[1])[:9]:9s} score={es.score:6.2f} lv={best.levels_completed} acts={best.actions:6d} t={res.get(es.id,(0,))[0]:6.1f}s  {' '.join(ls)}")
    if verbose:
        for k,(t,i) in res.items(): print('   ', k[:4], str(i)[:300])
    print(f"TOTAL score={sc.score:.3f}  levels={sc.total_levels_completed}/{sc.total_levels}  time={time.time()-t0:.0f}s")
    return sc
