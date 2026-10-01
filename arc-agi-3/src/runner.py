"""Scheduler: plays every game of an Arcade in parallel threads, in rounds of growing action budgets.

Round k gives each unfinished game a cumulative budget BUDGETS[k]; games are resumed where they
stopped, so short early rounds guarantee every game gets played before long rounds deepen them.
Everything stops at a global deadline. One env per game (gateway rule), one scorecard overall.
"""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

BUDGETS = [600, 2000, 5000, 10000, 20000, 40000, 80000, 160000]


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def run_all(arcade, explorer_cls, deadline: float, workers: int = 12, card_id=None,
            budgets=BUDGETS, log=print, game_ids=None):
    ids = game_ids or [e.game_id for e in arcade.available_environments]
    log(f"[runner] {len(ids)} games, workers={workers}, "
        f"time left={deadline - time.time():.0f}s")
    envs, explorers, status = {}, {}, {}

    def make(gid):
        env = arcade.make(gid, scorecard_id=card_id)
        if env is None:
            raise RuntimeError("make returned None")
        return env

    for gid in ids:
        try:
            envs[gid] = make(gid)
            explorers[gid] = explorer_cls(envs[gid], gid, deadline=deadline, max_actions=0)
            status[gid] = "READY"
        except Exception:
            status[gid] = "MAKE_FAIL"
            log(f"[runner] make failed for {gid}: {traceback.format_exc(limit=2)}")

    def play(gid, budget):
        ex = explorers[gid]
        ex.max_actions = budget
        ex.deadline = deadline
        try:
            return gid, ex.play()
        except Exception:
            return gid, "EXC " + traceback.format_exc(limit=3)[-300:]

    for rnd, budget in enumerate(budgets):
        todo = [g for g in ids if status.get(g) not in ("WIN", "MAKE_FAIL")]
        if not todo or time.time() > deadline - 5:
            break
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(play, g, budget) for g in todo]
            for f in as_completed(futs):
                gid, res = f.result()
                status[gid] = "WIN" if res == "WIN" else res.split(" ")[0]
                if res.startswith("EXC"):
                    log(f"[runner] {gid}: {res}")
        lv = sum((explorers[g].level or 0) for g in explorers)
        acts = sum(explorers[g].actions_taken for g in explorers)
        wins = sum(1 for s in status.values() if s == "WIN")
        log(f"[runner] round {rnd} budget={budget} games={len(todo)} took={time.time() - t0:.0f}s "
            f"levels={lv} wins={wins} actions={acts} rss={rss_mb()}MB time_left={deadline - time.time():.0f}s")
    return explorers, status
