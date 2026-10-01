"""ARC-AGI-3 symbolic explorer agent (written from scratch).

Plays one game through an arc_agi EnvironmentWrapper (local or remote gateway).
Per level it builds a graph of masked-frame states and explores it at minimum action cost:
  * step counters / progress bars are masked so revisited states are recognised;
  * untried actions are ranked by a learned prior (does this action / clicked colour change anything?);
  * the next action is chosen by prior / (navigation cost + 1), where navigation can walk known
    edges, use a learned UNDO action, or spend one RESET to return to the level start;
  * GAME_OVER caused by a move limit is told apart from a deadly action, and plans respect the limit.
"""
from __future__ import annotations

import hashlib
import zlib
import random
import time

import numpy as np
from arcengine import GameAction, GameState

EDGE = 2           # edge band (pixels) where step counters live
COUNTER_MAX = 8    # max pixels of a counter diff component
MAX_CLICKS = 128   # cap on click candidates per state
import os as _os
TOWARD_W = float(_os.environ.get("AG_TOWARD", 1.5))  # weight for moving the avatar towards unvisited salient objects
AWAY_W = float(_os.environ.get("AG_AWAY", 0.25))     # weight for avatar moves that do not approach any target
MAX_SCORE = AWAY_W + TOWARD_W
DEATH = "DEATH"
RETRIES = 4        # transport retries per action (remote gateway)
CACHE_NODES = 15000  # above this many states per level, untried caches are dropped periodically


# ----------------------------------------------------------------------------- perception
try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def components(frame: np.ndarray):
    """4-connected components of equal colour -> list of (color, ys, xs)."""
    if _ndi is not None:
        out = []
        for c in np.unique(frame):
            lab, n = _ndi.label(frame == c)
            if n == 0:
                continue
            idx = np.argsort(lab, axis=None, kind="stable")
            flat = lab.ravel()[idx]
            starts = np.searchsorted(flat, np.arange(1, n + 1))
            ends = np.append(starts[1:], flat.size)
            ys_all, xs_all = np.divmod(idx, frame.shape[1])
            for s, e in zip(starts, ends):
                out.append((int(c), ys_all[s:e], xs_all[s:e]))
        return out
    return _components_py(frame)


def _components_py(frame: np.ndarray):
    h, w = frame.shape
    lab = -np.ones((h, w), dtype=np.int32)
    out = []
    f = frame
    for y in range(h):
        for x in range(w):
            if lab[y, x] >= 0:
                continue
            c = f[y, x]
            idx = len(out)
            lab[y, x] = idx
            stack = [(y, x)]
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                if cy > 0 and lab[cy - 1, cx] < 0 and f[cy - 1, cx] == c:
                    lab[cy - 1, cx] = idx; stack.append((cy - 1, cx))
                if cy < h - 1 and lab[cy + 1, cx] < 0 and f[cy + 1, cx] == c:
                    lab[cy + 1, cx] = idx; stack.append((cy + 1, cx))
                if cx > 0 and lab[cy, cx - 1] < 0 and f[cy, cx - 1] == c:
                    lab[cy, cx - 1] = idx; stack.append((cy, cx - 1))
                if cx < w - 1 and lab[cy, cx + 1] < 0 and f[cy, cx + 1] == c:
                    lab[cy, cx + 1] = idx; stack.append((cy, cx + 1))
            out.append((int(c), ys, xs))
    return out


def mask_components(mask: np.ndarray):
    """8-connected components of a boolean mask -> list of pixel lists."""
    left = set(zip(*np.nonzero(mask)))
    out = []
    while left:
        s = left.pop()
        comp = [s]
        stack = [s]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in left:
                        left.remove(q)
                        comp.append(q)
                        stack.append(q)
        out.append(comp)
    return out


def click_candidates(frame: np.ndarray, mask: np.ndarray):
    """One representative pixel per colour component (masked UI excluded), small first."""
    cands = []
    for color, ys, xs in components(frame):
        ys = np.asarray(ys); xs = np.asarray(xs)
        keep = ~mask[ys, xs]
        if not keep.any():
            continue
        ys, xs = ys[keep], xs[keep]
        cy, cx = ys.mean(), xs.mean()
        i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cands.append((len(ys), color, int(xs[i]), int(ys[i])))
    cands.sort(key=lambda t: t[0])
    return cands[:MAX_CLICKS]


# ----------------------------------------------------------------------------- graph
class Node:
    __slots__ = ("key", "_z", "untried", "edges", "alt")

    def __init__(self, key, frame):
        self.key = key
        self._z = zlib.compress(np.asarray(frame, dtype=np.uint8).tobytes(), 1)
        self.untried = None   # list of candidate actions not yet taken here
        self.edges = {}       # action key -> dst key | DEATH (latest outcome)
        self.alt = None       # action key -> set of other observed outcomes (hidden state)

    @property
    def frame(self):
        return np.frombuffer(zlib.decompress(self._z), dtype=np.uint8).reshape(64, 64).astype(np.int16)


class Explorer:
    def __init__(self, env, game_id: str, deadline: float, max_actions: int = 100000,
                 log=None, seed: int = 0):
        self.env = env
        self.game_id = game_id
        self.deadline = deadline
        self.max_actions = max_actions
        self.log = log or (lambda *a: None)
        self.rng = random.Random(seed)
        self.mask = np.zeros((64, 64), dtype=bool)
        self.actions_taken = 0
        # priors shared across levels of this game
        self.act_stats: dict = {}     # action id -> [changed, tries]
        self.color_stats: dict = {}   # clicked colour -> [changed, tries]
        # avatar model (shared across levels)
        self.move_color: dict = {}    # colour -> #transitions where it moved
        self.act_vec: dict = {}       # simple action -> [sum dy, sum dx, n]
        self.goal_colors: dict = {}   # colour of the object touched when a level was won
        self.jitter = {}
        self.level = None
        self.undo_ok = 0
        self.undo_bad = 0
        self.stats = {'try': 0, 'noop': 0, 'nav': 0, 'reset': 0, 'death': 0, 'undo': 0}
        self.level_log = []
        self.new_level()

    # -- level state -----------------------------------------------------------
    def new_level(self):
        self.nodes: dict[str, Node] = {}
        self.start_key = None
        self.t = 0                  # actions since level start / last reset
        self.max_life = 0           # longest survival (in steps) seen this level
        self.move_limit = None      # inferred per-life move budget
        self.history: list = []     # node keys visited this life (for undo)
        self.exhaust_count = 0
        self.exhaust_total = 0
        self.fail: dict = {}          # frontier node -> failed attempts to reach it
        self.dist_cache = None
        self.pos: dict = {}           # node key -> avatar (y, x)
        self.targets = None           # salient static objects of this level
        self.target_sig = None
        self.start_frame = None

    def key_of(self, frame):
        f = np.where(self.mask, 255, frame).astype(np.uint8)
        return hashlib.blake2b(f.tobytes(), digest_size=12).hexdigest()

    def node(self, frame):
        k = self.key_of(frame)
        n = self.nodes.get(k)
        if n is None:
            n = Node(k, frame)
            self.nodes[k] = n
            self.dist_cache = None
            if len(self.nodes) >= CACHE_NODES and len(self.nodes) % 5000 == 0:
                # memory guard: untried lists are a cache (candidates minus tried edges); rebuild lazily
                for m in self.nodes.values():
                    m.untried = None
        return n

    def rebuild(self):
        """Mask changed: re-key every node and merge duplicates."""
        old = self.nodes
        remap = {k: self.key_of(n.frame) for k, n in old.items()}
        self.nodes = {}
        for k, n in old.items():
            nk = remap[k]
            if nk not in self.nodes:
                self.nodes[nk] = Node(nk, n.frame)
        for k, n in old.items():
            m = self.nodes[remap[k]]
            for a, d in n.edges.items():
                m.edges[a] = d if d == DEATH else remap.get(d, d)
            if n.alt:
                m.alt = m.alt or {}
                for a, ds in n.alt.items():
                    m.alt.setdefault(a, set()).update(remap.get(d, d) for d in ds)
        if self.start_key is not None:
            self.start_key = remap.get(self.start_key, self.start_key)
        self.history = [remap.get(h, h) for h in self.history]
        self.dist_cache = None

    def update_mask(self, f0, f1):
        diff = (f0 != f1) & ~self.mask
        if not diff.any():
            return False
        changed = False
        for comp in mask_components(diff):
            if len(comp) > COUNTER_MAX:
                continue
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            if all(y <= EDGE for y in ys) or all(y >= 63 - EDGE for y in ys):
                for y in set(ys):
                    self.mask[y, :] = True
                changed = True
            elif all(x <= EDGE for x in xs) or all(x >= 63 - EDGE for x in xs):
                for x in set(xs):
                    self.mask[:, x] = True
                changed = True
        return changed

    # -- actions -----------------------------------------------------------------
    def undo_known(self):
        return self.undo_ok >= 2 and self.undo_ok > 3 * self.undo_bad

    def candidates(self, n: Node, avail):
        acts = []
        for a in avail:
            if a == 0:
                continue
            if a == 6:
                for size, color, x, y in click_candidates(n.frame, self.mask):
                    acts.append((6, x, y, color, size))
            elif a == 7 and self.undo_known():
                continue
            else:
                acts.append((a,))
        return acts

    # -- avatar / target model -------------------------------------------------
    def mobile(self):
        tot = sum(self.move_color.values())
        mob = {c for c, n in self.move_color.items() if n >= 2 and n >= 0.15 * tot}
        f = self.start_frame
        if f is not None and mob:
            # the avatar is small; big mobile colours are the floor it walks over
            small = {c for c in mob if (f == c).sum() <= 256}
            if small:
                return small
        return mob

    def observe_motion(self, act, f0, f1, k0, k1):
        diff = (f0 != f1) & ~self.mask
        if not diff.any():
            if k1 not in self.pos and k0 in self.pos:
                self.pos[k1] = self.pos[k0]
            return
        for c in np.unique(np.concatenate([f0[diff], f1[diff]])):
            app = int(((f1 == c) & diff).sum())
            dis = int(((f0 == c) & diff).sum())
            if app and dis and min(app, dis) >= 0.5 * max(app, dis):
                self.move_color[int(c)] = self.move_color.get(int(c), 0) + 1
        mob = self.mobile()
        if not mob:
            return
        mobl = list(mob)
        new = diff & np.isin(f1, mobl)
        old = diff & np.isin(f0, mobl)
        if new.any():
            ys, xs = np.nonzero(new)
            p1 = (float(ys.mean()), float(xs.mean()))
            self.pos[k1] = p1
            if old.any() and act[0] != 6:
                ys0, xs0 = np.nonzero(old)
                p0 = (float(ys0.mean()), float(xs0.mean()))
                v = self.act_vec.setdefault(act[0], [0.0, 0.0, 0])
                v[0] += p1[0] - p0[0]; v[1] += p1[1] - p0[1]; v[2] += 1
        elif k0 in self.pos:
            self.pos[k1] = self.pos[k0]

    def level_targets(self):
        mob = self.mobile()
        sig = (tuple(sorted(mob)), tuple(sorted(self.goal_colors)))
        if self.targets is not None and sig == self.target_sig:
            return self.targets
        self.target_sig = sig
        f = self.start_frame
        tg = []
        if f is not None and mob:
            vals, cnt = np.unique(f, return_counts=True)
            bg = {int(v) for v, c in zip(vals, cnt) if c > 0.15 * f.size}
            comps = [c for c in components(f) if c[0] not in bg and c[0] not in mob and len(c[1]) <= 200]
            ncomp = {}
            for color, ys, xs in comps:
                ncomp[color] = ncomp.get(color, 0) + 1
            for color, ys, xs in comps:
                ys = np.asarray(ys); xs = np.asarray(xs)
                if self.mask[ys, xs].all():
                    continue
                w = (1.0 + 3.0 * self.goal_colors.get(color, 0)) / ncomp[color]
                tg.append([float(ys.mean()), float(xs.mean()), color, w, False])
        self.targets = tg
        return tg

    def toward(self, key, act):
        if act[0] == 6 or key not in self.pos:
            return None
        v = self.act_vec.get(act[0])
        if not v or v[2] == 0:
            return None
        vy, vx = v[0] / v[2], v[1] / v[2]
        norm = (vy * vy + vx * vx) ** 0.5
        if norm < 0.5:
            return None
        tg = [t for t in self.level_targets() if not t[4]]
        if not tg:
            return None
        py, px = self.pos[key]
        best = max(tg, key=lambda t: t[3] / (1.0 + ((t[0] - py) ** 2 + (t[1] - px) ** 2) ** 0.5))
        d0 = ((best[0] - py) ** 2 + (best[1] - px) ** 2) ** 0.5
        d1 = ((best[0] - py - vy) ** 2 + (best[1] - px - vx) ** 2) ** 0.5
        return max(0.0, min(1.0, (d0 - d1) / norm))

    def direction_factor(self, key, act):
        t = self.toward(key, act)
        if t is None:
            return 1.0
        return AWAY_W + TOWARD_W * t

    def mark_visited(self, key):
        if key not in self.pos or not self.targets:
            return
        py, px = self.pos[key]
        for t in self.targets:
            if not t[4] and abs(t[0] - py) <= 4 and abs(t[1] - px) <= 4:
                t[4] = True

    def prior(self, act):
        a = act[0]
        ch, tr = self.act_stats.get(a, (0, 0))
        p = (ch + 1.0) / (tr + 2.0)
        if a == 6:
            cch, ctr = self.color_stats.get(act[3], (0, 0))
            p = (cch + 0.5) / (ctr + 1.0)
            p -= 0.02 * min(act[4], 1024) / 1024.0
        return max(p, 0.01)

    def ensure_untried(self, n: Node, avail):
        if n.untried is None:
            done = set(n.edges)
            n.untried = [c for c in self.candidates(n, avail) if c[:3] not in done]

    def best_untried(self, n: Node):
        if not n.untried:
            return None, 0.0
        k = n.key
        jit = self.jitter
        sc = lambda a: self.prior(a) * self.direction_factor(k, a) * (1.0 + jit.get(a[:3], 0.0))
        for a in n.untried:
            if a[:3] not in jit:
                jit[a[:3]] = self.rng.random() * 1e-3
        best = max(n.untried, key=sc)
        return best, sc(best)

    # -- environment ---------------------------------------------------------------
    def _call(self, a, data):
        for attempt in range(RETRIES):
            try:
                r = self.env.step(a, data)
            except Exception:
                r = None
            if r is not None:
                return r
            time.sleep(0.5 * (attempt + 1))
        return None

    def env_step(self, act):
        a = GameAction.from_id(act[0])
        data = {"x": int(act[1]), "y": int(act[2])} if act[0] == 6 else {}
        r = self._call(a, data)
        self.actions_taken += 1
        return r

    def env_reset(self):
        r = self._call(GameAction.RESET, {})
        self.actions_taken += 1
        self.stats['reset'] += 1
        self.t = 0
        self.history = []
        return r

    @staticmethod
    def last_frame(r):
        if r is None or not r.frame:
            return None
        f = np.asarray(r.frame[-1], dtype=np.int16)
        if f.shape != (64, 64):
            g = np.zeros((64, 64), dtype=np.int16)
            if f.ndim == 2:
                h, w = min(64, f.shape[0]), min(64, f.shape[1])
                g[:h, :w] = f[:h, :w]
            f = g
        return f

    # -- planning --------------------------------------------------------------------
    def edge_ok(self, d):
        return d != DEATH and d in self.nodes

    @staticmethod
    def outcomes(n: Node):
        for a, d in n.edges.items():
            yield a, d
        if n.alt:
            for a, ds in n.alt.items():
                for d in ds:
                    yield a, d

    def dist_from_start(self):
        if self.dist_cache is not None:
            return self.dist_cache
        dist = {}
        if self.start_key in self.nodes:
            dist[self.start_key] = 0
            frontier = [self.start_key]
            while frontier:
                nxt = []
                for k in frontier:
                    for a, d in self.outcomes(self.nodes[k]):
                        if self.edge_ok(d) and d not in dist:
                            dist[d] = dist[k] + 1
                            nxt.append(d)
                frontier = nxt
        self.dist_cache = dist
        return dist

    def plan(self, cur: Node, avail):
        """Pick (cost, route, action) maximising prior/(cost+1).

        Sources: current node (cost 0), undo ancestors (cost k), level start via RESET (cost 1).
        route is a list of steps: ("act", a) | ("undo",) | ("reset",).
        """
        sources = [(0, cur.key, [])]
        if self.undo_known() and self.history:
            hist = self.history
            for k in range(1, min(len(hist), 30) + 1):
                sources.append((k, hist[-1 - k] if k < len(hist) else None,
                                [("undo", hist[-1 - j]) for j in range(1, k + 1)] if k < len(hist) else []))
            sources = [s for s in sources if s[1] is not None]
        if self.start_key in self.nodes and cur.key != self.start_key:
            sources.append((1, self.start_key, [("reset", self.start_key)]))
        sources.sort(key=lambda s: s[0])

        best = None
        seen = {}
        # Dijkstra with unit edges and small integer source offsets == bucketed BFS
        buckets: dict[int, list] = {}
        for c, k, r in sources:
            buckets.setdefault(c, []).append((k, r, c == 1 and r and r[0][0] == "reset"))
        cost = 0
        maxcost = max(buckets) if buckets else 0
        limit = self.move_limit
        while cost <= maxcost:
            for k, route, via_reset in buckets.pop(cost, []):
                if k in seen:
                    continue
                seen[k] = cost
                n = self.nodes.get(k)
                if n is None:
                    continue
                # steps into this life when we'd take the new action
                if via_reset:
                    life_t = cost - 1
                else:
                    life_t = self.t + sum(1 for s in route if s[0] == "act")
                self.ensure_untried(n, avail)
                act, p = self.best_untried(n)
                if act is not None and (limit is None or life_t + 1 < limit) \
                        and (not route or self.fail.get(k, 0) < 3):
                    score = p / (cost + 1.0)
                    if best is None or score > best[0]:
                        best = (score, route, act, k)
                if limit is not None and life_t + 1 >= limit:
                    continue
                for a, d in self.outcomes(n):
                    if self.edge_ok(d) and d not in seen:
                        buckets.setdefault(cost + 1, []).append((d, route + [("act", a, d)], via_reset))
                        if cost + 1 > maxcost:
                            maxcost = cost + 1
            cost += 1
            # early exit: nothing further can beat best (prior <= 1)
            if best is not None and MAX_SCORE / (cost + 1.0) < best[0]:
                break
        return best

    # -- main loop ---------------------------------------------------------------------
    def play(self):
        """Play until WIN or budget. Resumable: call again with a larger max_actions/deadline."""
        r = self.env.observation_space
        if r is None or r.state == GameState.NOT_PLAYED or not r.frame:
            r = self.env_reset()
            if r is None:
                return "ERROR"
        if self.level is None or r.levels_completed != self.level:
            self.level = r.levels_completed
            self.new_level()
        level = self.level
        frame = self.last_frame(r)
        if frame is None:
            r = self.env_reset()
            frame = self.last_frame(r)
            if frame is None:
                return "ERROR"
        cur = self.node(frame)
        if self.start_key is None:
            self.start_key = cur.key
            self.start_frame = cur.frame
        route: list = []
        pending = None
        target = None
        while True:
            if r is None:
                return "ERROR"
            if r.state == GameState.WIN:
                return "WIN"
            if time.time() > self.deadline or self.actions_taken >= self.max_actions:
                return "BUDGET"
            avail = [a for a in (r.available_actions or [1, 2, 3, 4, 5, 6]) if a != 0]

            if r.state == GameState.GAME_OVER or frame is None:
                r = self.env_reset()
                route, pending = [], None
                frame = self.last_frame(r)
                if frame is not None:
                    cur = self.node(frame)
                    if self.start_key is None:
                        self.start_key = cur.key
                        self.start_frame = cur.frame
                continue

            # ---- decide
            if not route and pending is None:
                choice = self.plan(cur, avail)
                if choice is None:
                    self.exhaust_count += 1
                    if self.exhaust_count <= 2 and cur.key != self.start_key:
                        r = self.env_reset()
                        frame = self.last_frame(r)
                        if frame is not None:
                            cur = self.node(frame)
                        continue
                    # graph exhausted: random walk (keeps the graph); forget it only as a last resort
                    self.exhaust_total += 1
                    if self.exhaust_total % 25 == 0:
                        self.nodes = {}
                        self.dist_cache = None
                        cur = self.node(frame)
                        self.start_key = None
                    for _ in range(20):
                        if not avail:
                            break
                        a = self.rng.choice(avail)
                        act = (6, self.rng.randrange(64), self.rng.randrange(64)) if a == 6 else (a,)
                        r = self.env_step(act)
                        self.t += 1
                        if r is None or r.state != GameState.NOT_FINISHED or r.levels_completed > level:
                            break
                    frame = self.last_frame(r)
                    if frame is not None and r.levels_completed == level and r.state == GameState.NOT_FINISHED:
                        cur = self.node(frame)
                    self.exhaust_count = 0
                    if r is not None and r.levels_completed > level:
                        level = r.levels_completed
                        self.level = level
                        self.level_log.append((level - 1, dict(self.stats), 0))
                        self.new_level()
                        if frame is not None:
                            cur = self.node(frame)
                            self.start_key = cur.key
                            self.start_frame = cur.frame
                    continue
                _, route, pending, target = choice
                route = list(route)

            route_step = bool(route)
            step = None
            if route:
                step = route.pop(0)
                if step[0] == "reset":
                    r = self.env_reset()
                    frame = self.last_frame(r)
                    if frame is None:
                        route, pending = [], None
                        continue
                    cur = self.node(frame)
                    if cur.key != self.start_key:
                        self.start_key = cur.key
                        self.start_frame = cur.frame
                        route, pending = [], None
                    continue
                if step[0] == "undo":
                    act = (7,)
                    self.stats['undo'] += 1
                else:
                    act = step[1]
                    self.stats['nav'] += 1
            else:
                act = pending
                pending = None
                self.stats['try'] += 1

            # ---- act
            prev_frame, prev_key = frame, cur.key
            r = self.env_step(act)
            self.t += 1
            if r is None:
                return "ERROR"
            frame = self.last_frame(r)
            akey = act[:3]
            if cur.untried is not None:
                cur.untried = [c for c in cur.untried if c[:3] != akey]
            expect = step[-1] if (route_step and step[0] in ("act", "undo")) else None

            if r.levels_completed > level or r.state == GameState.WIN:
                self.observe_motion(act, prev_frame, frame if frame is not None else prev_frame, cur.key, "__win__")
                p = self.pos.get("__win__") or self.pos.get(cur.key)
                if p is not None and self.targets:
                    t = min(self.targets, key=lambda t: (t[0] - p[0]) ** 2 + (t[1] - p[1]) ** 2)
                    if (t[0] - p[0]) ** 2 + (t[1] - p[1]) ** 2 <= 100:
                        self.goal_colors[t[2]] = self.goal_colors.get(t[2], 0) + 1
                self.level_log.append((level, dict(self.stats), len(self.nodes)))
                self.stats = {k: 0 for k in self.stats}
                self.log(f"{self.game_id} level {level}->{r.levels_completed} at {self.actions_taken}")
                level = r.levels_completed
                self.level = level
                self.new_level()
                route, pending = [], None
                if frame is not None:
                    cur = self.node(frame)
                    self.start_key = cur.key
                    self.start_frame = cur.frame
                continue

            if r.state == GameState.GAME_OVER or frame is None:
                self.stats['death'] += 1
                if self.t >= max(self.max_life + 1, 6):
                    # died at a new record length: most likely a move limit, not this action
                    if self.move_limit is None or self.t > self.move_limit:
                        self.move_limit = self.t
                else:
                    cur.edges[akey] = DEATH
                    self.dist_cache = None
                route, pending = [], None
                continue

            self.max_life = max(self.max_life, self.t)
            if self.move_limit is not None and self.max_life >= self.move_limit:
                self.move_limit = None

            if self.update_mask(prev_frame, frame):
                self.rebuild()
                cur = self.nodes[self.key_of(prev_frame)]
                prev_key = cur.key
            changed = bool(((prev_frame != frame) & ~self.mask).any())
            if not changed:
                self.stats['noop'] += 1
            st = self.act_stats.setdefault(act[0], [0, 0])
            st[1] += 1
            st[0] += int(changed)
            if act[0] == 6 and len(act) > 3:
                cs = self.color_stats.setdefault(act[3], [0, 0])
                cs[1] += 1
                cs[0] += int(changed)

            nxt = self.node(frame)
            self.observe_motion(act, prev_frame, frame, cur.key, nxt.key)
            self.mark_visited(nxt.key)
            if act[0] == 7:
                # learn whether ACTION7 behaves as undo
                if len(self.history) >= 2 and changed:
                    if nxt.key == self.history[-2]:
                        self.undo_ok += 1
                    else:
                        self.undo_bad += 1
                elif not changed and len(self.history) >= 2:
                    self.undo_bad += 1
                if self.undo_known():
                    if self.history:
                        self.history.pop()
                    if not self.history or self.history[-1] != nxt.key:
                        self.history.append(nxt.key)
                    cur = nxt
                    if expect is not None and cur.key != expect:
                        self.fail[target] = self.fail.get(target, 0) + 1
                        route, pending = [], None
                    continue
            expected = cur.edges.get(akey)
            cur.edges[akey] = nxt.key
            if expected is not None and expected != nxt.key and expected != DEATH:
                if cur.alt is None:
                    cur.alt = {}
                cur.alt.setdefault(akey, set()).add(expected)
            if expected is None or expected != nxt.key:
                self.dist_cache = None
            if expected is not None and expected != nxt.key:
                route, pending = [], None   # nondeterminism / hidden state: replan
            if not self.history or self.history[-1] != prev_key:
                self.history.append(prev_key)
            if self.history[-1] != nxt.key:
                self.history.append(nxt.key)
            if len(self.history) > 200:
                self.history = self.history[-200:]
            cur = nxt
            if expect is not None and cur.key != expect:
                self.fail[target] = self.fail.get(target, 0) + 1
                route, pending = [], None
