"""ARC-AGI-3 symbolic explorer agent (from scratch).

Plays one game through an arc_agi EnvironmentWrapper (local or remote gateway).
Per level it builds a graph of masked-frame states and explores it efficiently.
"""
from __future__ import annotations

import hashlib
import time
from collections import deque

import numpy as np
from arcengine import GameAction, GameState

EDGE = 2          # edge band (pixels) where step counters live
COUNTER_MAX = 8   # max pixels of a counter diff component
MAX_CLICKS = 96   # cap on click candidates per state


# ----------------------------------------------------------------------------- perception
def components(frame: np.ndarray, conn8: bool = False):
    """Connected components of equal colour. Returns list of (color, pixels[list of (y,x)])."""
    h, w = frame.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if conn8:
        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            c = frame[y, x]
            stack = [(y, x)]
            seen[y, x] = True
            pix = []
            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and frame[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((int(c), pix))
    return out


def mask_components(mask: np.ndarray):
    """8-connected components of a boolean mask -> list of pixel lists."""
    pts = list(zip(*np.nonzero(mask)))
    left = set(pts)
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


def click_candidates(frame: np.ndarray):
    """One representative pixel per colour component, smallest components first."""
    comps = components(frame)
    cands = []
    for color, pix in comps:
        arr = np.array(pix)
        cy, cx = arr.mean(0)
        i = int(np.argmin((arr[:, 0] - cy) ** 2 + (arr[:, 1] - cx) ** 2))
        y, x = int(arr[i, 0]), int(arr[i, 1])
        cands.append((len(pix), color, x, y))
    cands.sort(key=lambda t: t[0])
    return cands[:MAX_CLICKS]


# ----------------------------------------------------------------------------- graph
class Node:
    __slots__ = ("key", "frame", "untried", "edges", "dist")

    def __init__(self, key, frame):
        self.key = key
        self.frame = frame
        self.untried = None   # list of action tuples, filled lazily
        self.edges = {}       # action -> dst key, or "DEATH"
        self.dist = None


class Explorer:
    def __init__(self, env, game_id: str, deadline: float, max_actions: int = 100000, log=None):
        self.env = env
        self.game_id = game_id
        self.deadline = deadline
        self.max_actions = max_actions
        self.log = log or (lambda *a: None)
        self.mask = np.zeros((64, 64), dtype=bool)
        self.actions_taken = 0
        # priors shared across levels of this game: action-id -> [changes, tries]
        self.act_stats = {}
        self.color_stats = {}
        self.stats = {'try':0,'noop':0,'nav':0,'reset':0,'death':0}
        self.level_log = []
        self.reset_level_state()

    # -- bookkeeping ---------------------------------------------------------
    def reset_level_state(self):
        self.nodes: dict[str, Node] = {}
        self.start_key = None

    def key_of(self, frame):
        f = np.where(self.mask, 255, frame).astype(np.uint8)
        return hashlib.blake2b(f.tobytes(), digest_size=12).hexdigest()

    def node(self, frame):
        k = self.key_of(frame)
        n = self.nodes.get(k)
        if n is None:
            n = Node(k, frame)
            self.nodes[k] = n
        return n

    def rebuild(self):
        """Mask changed: re-key every node and merge duplicates."""
        old = self.nodes
        remap = {}
        self.nodes = {}
        for k, n in old.items():
            nk = self.key_of(n.frame)
            remap[k] = nk
            if nk not in self.nodes:
                m = Node(nk, n.frame)
                m.edges = {}
                self.nodes[nk] = m
        for k, n in old.items():
            m = self.nodes[remap[k]]
            for a, d in n.edges.items():
                m.edges[a] = d if d == "DEATH" else remap.get(d, d)
        for m in self.nodes.values():
            m.untried = None
        if self.start_key is not None:
            self.start_key = remap.get(self.start_key, self.start_key)

    def update_mask(self, f0, f1):
        diff = f0 != f1
        if not diff.any():
            return False
        changed = False
        for comp in mask_components(diff & ~self.mask):
            if len(comp) > COUNTER_MAX:
                continue
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            top = all(y < EDGE + 1 for y in ys)
            bot = all(y > 63 - EDGE - 1 for y in ys)
            lef = all(x < EDGE + 1 for x in xs)
            rig = all(x > 63 - EDGE - 1 for x in xs)
            if top or bot:
                for y in set(ys):
                    self.mask[y, :] = True
                changed = True
            elif lef or rig:
                for x in set(xs):
                    self.mask[:, x] = True
                changed = True
        return changed

    # -- action candidates ---------------------------------------------------
    def candidates(self, n: Node, avail):
        acts = []
        for a in avail:
            if a == 0:
                continue
            if a == 6:
                for size, color, x, y in click_candidates(n.frame):
                    acts.append((6, x, y, color, size))
            else:
                acts.append((a,))
        return acts

    def prior(self, act):
        a = act[0]
        ch, tr = self.act_stats.get(a, (0, 0))
        p = (ch + 1.0) / (tr + 2.0)
        if a == 6:
            cch, ctr = self.color_stats.get(act[3], (0, 0))
            p = 0.5 * p + (cch + 1.0) / (ctr + 2.0)
            p -= 0.001 * min(act[4], 400) / 400.0
        return p

    def ensure_untried(self, n: Node, avail):
        if n.untried is None:
            n.untried = [c for c in self.candidates(n, avail) if c[:3] not in {e[:3] for e in n.edges}]

    # -- environment ---------------------------------------------------------
    def step(self, act):
        a = GameAction.from_id(act[0])
        data = {"x": int(act[1]), "y": int(act[2])} if act[0] == 6 else {}
        r = self.env.step(a, data)
        self.actions_taken += 1
        return r

    @staticmethod
    def last_frame(r):
        if r is None or not r.frame:
            return None
        return np.asarray(r.frame[-1], dtype=np.int16)

    # -- planning ------------------------------------------------------------
    def path_to_frontier(self, src: str, avail):
        """BFS over known safe edges to the nearest node with untried actions."""
        prev = {src: None}
        q = deque([src])
        while q:
            k = q.popleft()
            n = self.nodes.get(k)
            if n is None:
                continue
            self.ensure_untried(n, avail)
            if n.untried:
                path = []
                while prev[k] is not None:
                    pk, a = prev[k]
                    path.append(a)
                    k = pk
                return path[::-1]
            for a, d in n.edges.items():
                if d != "DEATH" and d not in prev:
                    prev[d] = (k, a)
                    q.append(d)
        return None

    # -- main loop -----------------------------------------------------------
    def play(self):
        r = self.env.observation_space
        if r is None or r.state == GameState.NOT_PLAYED or not r.frame:
            r = self.env.step(GameAction.RESET, {})
        level = r.levels_completed
        frame = self.last_frame(r)
        cur = self.node(frame)
        self.start_key = cur.key
        plan: list = []
        while True:
            if r.state == GameState.WIN:
                return "WIN"
            if time.time() > self.deadline or self.actions_taken >= self.max_actions:
                return "BUDGET"
            avail = list(r.available_actions or [1, 2, 3, 4, 5, 6])
            if r.state == GameState.GAME_OVER or frame is None:
                r = self.env.step(GameAction.RESET, {})
                self.actions_taken += 1
                plan = []
                frame = self.last_frame(r)
                if frame is None:
                    continue
                cur = self.node(frame)
                if self.start_key is None:
                    self.start_key = cur.key
                continue

            self.ensure_untried(cur, avail)
            if plan:
                act = plan.pop(0)
                self.stats['nav'] += 1
            elif cur.untried:
                best = max(range(len(cur.untried)), key=lambda i: self.prior(cur.untried[i]))
                act = cur.untried[best]
                self.stats['try'] += 1
            else:
                path = self.path_to_frontier(cur.key, avail)
                if path:
                    plan = path
                    act = plan.pop(0)
                    self.stats['nav'] += 1
                else:
                    self.stats['reset'] += 1
                    r = self.env.step(GameAction.RESET, {})
                    self.actions_taken += 1
                    frame = self.last_frame(r)
                    if frame is None:
                        continue
                    cur = self.node(frame)
                    if not self.path_to_frontier(cur.key, avail) and not cur.untried:
                        return "EXHAUSTED"
                    continue

            prev_frame = frame
            r = self.step(act)
            if r is None:
                return "ERROR"
            frame = self.last_frame(r)
            akey = act[:3]
            if cur.untried is not None:
                cur.untried = [c for c in cur.untried if c[:3] != akey]

            if r.levels_completed > level or r.state == GameState.WIN:
                self.level_log.append((level, dict(self.stats), len(self.nodes))); self.stats = {k:0 for k in self.stats}
                self.log(f"{self.game_id} level {level}->{r.levels_completed} after {self.actions_taken} actions, nodes={len(self.nodes)}")
                level = r.levels_completed
                self.reset_level_state()
                plan = []
                if frame is not None:
                    cur = self.node(frame)
                    self.start_key = cur.key
                continue

            if r.state == GameState.GAME_OVER or frame is None:
                cur.edges[akey] = "DEATH"
                self.stats['death'] += 1
                plan = []
                continue

            changed = bool((prev_frame != frame).any())
            if not changed: self.stats['noop'] += 1
            # learn priors
            st = self.act_stats.setdefault(act[0], [0, 0])
            st[1] += 1
            st[0] += int(changed)
            if act[0] == 6 and len(act) > 3:
                cs = self.color_stats.setdefault(act[3], [0, 0])
                cs[1] += 1
                cs[0] += int(changed)

            if changed and self.update_mask(prev_frame, frame):
                self.rebuild()
                cur = self.nodes[self.key_of(prev_frame)]
            nxt = self.node(frame)
            expected = cur.edges.get(akey)
            cur.edges[akey] = nxt.key
            if expected is not None and expected != nxt.key:
                plan = []   # nondeterminism / hidden state: replan
            cur = nxt
