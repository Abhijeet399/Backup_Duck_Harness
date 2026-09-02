"""
probe_ls20_goal.py — find ls20's WIN condition empirically + report wall colors.

Two jobs:
  1. Collision: on every refused move, print the color(s) in the 5 cells ahead of
     the avatar's leading edge (traversable field 0/3 and consumable color-5
     excluded), so the blocking color is unambiguous.
  2. Goal: greedily drive the avatar into the nearest color-5 object to consume
     it, route around walls when blocked, reset through the step-timer, and watch
     for LEVEL_COMPLETED. Prints the win diff when it fires. If ALL color-5 is
     consumed with no win, says so — that falsifies the eat-all-5 hypothesis.

Run:  uv run python stage1/probe_ls20_goal.py --game ls20 --envs /test/ARC3/env_files
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects

UNIT = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
VEC = {k: (v[0] * 5, v[1] * 5) for k, v in UNIT.items()}
PERP = {"ACTION1": ["ACTION3", "ACTION4"], "ACTION2": ["ACTION3", "ACTION4"],
        "ACTION3": ["ACTION1", "ACTION2"], "ACTION4": ["ACTION1", "ACTION2"]}
TRAVERSABLE = {0, 3}          # field the avatar passes through freely
AV_COLORS = {9, 12}
CONSUMABLE = {5}


def _blobs(g):
    out = []
    for o in extract_objects(g):
        if (o.color == 12 and abs(o.size - 10) <= 4) or (o.color == 9 and abs(o.size - 15) <= 5):
            out.append(o)
    return out


def avatar_pos(g):
    cells = [c for o in _blobs(g) for c in o.cells]
    if not cells:
        return None
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return np.array([sum(rs) / len(rs), sum(cs) / len(cs)])


def avatar_cells(g):
    return [c for o in _blobs(g) for c in o.cells]


def c5_targets(g):
    out = []
    for o in extract_objects(g):
        if o.color == 5:
            r0, c0, r1, c1 = o.bbox
            out.append(np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0]))
    return out


def c5_total(g):
    return sum(o.size for o in extract_objects(g) if o.color == 5)


def blocking_report(g, action):
    ur, uc = UNIT[action]; H, W = g.shape
    hist, edge = {}, False
    for (r, c) in avatar_cells(g):
        for k in range(1, 6):
            nr, nc = r + ur * k, c + uc * k
            if not (0 <= nr < H and 0 <= nc < W):
                edge = True; continue
            v = int(g[nr, nc])
            if v in TRAVERSABLE or v in AV_COLORS or v in CONSUMABLE:
                continue
            hist[v] = hist.get(v, 0) + 1
    return hist, edge


def meaningful(before, after):
    d = object_diff(before, after); parts = []
    for o, off in d.moved:
        if o.size <= 40:
            parts.append(f"c{o.color}/{o.size} moved {off}")
    for o in d.appeared:
        if o.color not in (3, 11) and o.size <= 200:
            parts.append(f"c{o.color}/{o.size}+@{o.bbox[:2]}")
    for o in d.disappeared:
        if o.color not in (3, 11) and o.size <= 200:
            parts.append(f"c{o.color}/{o.size}-@{o.bbox[:2]}")
    return "; ".join(parts) or "(no small change)"


def md(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _apply(env, g, av, action, recent, tag=""):
    """Step one action, print a line, return (g, av, term, moved)."""
    before = g
    g, term = env.step(Action(action))
    q = avatar_pos(g)
    moved = q is not None and av is not None and md(q, av) >= 1
    line = f"  {action}{tag} c5_left={c5_total(g)} term={term} | {meaningful(before, g)}"
    if not moved:
        hist, edge = blocking_report(before, action)
        line += f"  BLOCKED by {hist or '{}'}{' +EDGE' if edge else ''}"
    print(line)
    if term == "LEVEL_COMPLETED":
        print("  *** LEVEL_COMPLETED — WIN DIFF:", object_diff(before, g).summary())
    return g, (q if moved else av), term, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--budget", type=int, default=300)
    args = ap.parse_args()

    env = ArcEnv(args.game, args.envs)
    g = env.reset()
    av = avatar_pos(g)
    recent = deque(maxlen=8)
    stall = 0
    step = 0
    print(f"[start] avatar={tuple(np.round(av,1))} c5_total={c5_total(g)} c5_objects={len(c5_targets(g))}")

    while step < args.budget:
        tgts = c5_targets(g)
        if not tgts:
            print(f"[{step:03d}] ALL color-5 consumed with NO LEVEL_COMPLETED "
                  f"-> eating-all-5 is NOT the win condition.")
            break
        tgt = min(tgts, key=lambda t: md(av, t))
        ranked = sorted(VEC, key=lambda a: md(np.array(av) + VEC[a], tgt))
        chosen = None
        for a in ranked:
            pred = tuple(np.round(np.array(av) + VEC[a]).astype(int))
            if pred in recent and len(ranked) > 1:
                continue
            chosen = a; break
        chosen = chosen or ranked[0]

        print(f"[{step:03d}] -> nearest c5 @({tgt[0]:.0f},{tgt[1]:.0f}) d={md(av,tgt):.0f}")
        g, av, term, moved = _apply(env, g, av, chosen, recent)
        step += 1
        if term == "LEVEL_COMPLETED":
            break
        if term == "GAME_OVER":
            print("  (timer expired -> RESET)")
            g, _ = env.step(Action("RESET")); av = avatar_pos(g); recent.clear(); stall = 0; continue
        if moved:
            recent.append(tuple(np.round(av).astype(int))); stall = 0
            continue

        # blocked: route around via best perpendicular
        stall += 1
        for pa in sorted(PERP[chosen], key=lambda a: md(np.array(av) + VEC[a], tgt)):
            g, av, term, pmoved = _apply(env, g, av, pa, recent, tag="(route)")
            step += 1
            if term == "LEVEL_COMPLETED":
                return
            if term == "GAME_OVER":
                print("  (timer -> RESET)")
                g, _ = env.step(Action("RESET")); av = avatar_pos(g); recent.clear(); stall = 0; break
            if pmoved:
                recent.append(tuple(np.round(av).astype(int))); stall = 0; break
        if stall >= 6:
            print(f"[{step:03d}] STUCK near {tuple(np.round(av,1))}; blockers all sides:")
            for a in VEC:
                h, e = blocking_report(g, a)
                print(f"    {a}: {h or '{}'}{' +EDGE' if e else ''}")
            break
    else:
        print(f"[budget {args.budget} exhausted] c5_left={c5_total(g)} — no win within budget")
    print("[done]")


if __name__ == "__main__":
    main()
