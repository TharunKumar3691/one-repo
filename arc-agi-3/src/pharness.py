"""Parallel local benchmark (4 procs). Per-game competition scoring; total = mean over games."""
import os, sys, time, logging, importlib, json
from multiprocessing import Pool
ENV_DIR = os.environ.get("ARC_ENV_DIR", "environment_files")

def _one(args):
    modname, gid, maxa, tl, seed = args
    logging.disable(logging.CRITICAL)
    os.environ.pop("ONLY_RESET_LEVELS", None)
    import arc_agi
    mod = importlib.import_module(modname)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)
    card = arcade.open_scorecard()
    env = arcade.make(gid, scorecard_id=card)
    os.environ["ONLY_RESET_LEVELS"] = "true"
    t0 = time.time()
    try:
        ex = mod.Explorer(env, gid, deadline=time.time() + tl, max_actions=maxa, seed=seed)
        res = ex.play()
    except Exception as e:
        import traceback; res = "EXC " + traceback.format_exc()[-400:]
    sc = arcade.get_scorecard(card)
    es = sc.environments[0]
    best = max(es.runs, key=lambda r: r.levels_completed)
    lv = [f"{a}/{b}" for a, b in zip(best.level_actions or [], best.level_baseline_actions or [])][:best.levels_completed + 1]
    return gid, es.score, best.levels_completed, best.actions, time.time() - t0, res, " ".join(lv)

def run(modname="agent", maxa=8000, tl=60, games=None, seed=0, quiet=False):
    import arc_agi, glob
    ids = sorted(os.path.basename(os.path.dirname(os.path.dirname(p))) for p in glob.glob(f"{ENV_DIR}/*/*/metadata.json"))
    gids = []
    for p in sorted(glob.glob(f"{ENV_DIR}/*/*/metadata.json")):
        m = json.load(open(p)); gids.append(m["game_id"])
    if games: gids = [g for g in gids if g[:4] in games]
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(_one, [(modname, g, maxa, tl, seed) for g in gids], chunksize=1)
    tot = sum(o[1] for o in out) / len(out)
    lv = sum(o[2] for o in out)
    if not quiet:
        for gid, s, l, a, t, res, lvs in out:
            print(f"{gid[:4]} {s:6.2f} lv={l} acts={a:6d} t={t:5.1f}s {str(res)[:8]:8s} {lvs[:120]}")
    print(f"TOTAL {tot:.3f} levels={lv} wall={time.time()-t0:.0f}s")
    return tot, out

if __name__ == "__main__":
    mod = sys.argv[1] if len(sys.argv) > 1 else "agent"
    maxa = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 60
    games = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else None
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    run(mod, maxa, tl, games, seed)
