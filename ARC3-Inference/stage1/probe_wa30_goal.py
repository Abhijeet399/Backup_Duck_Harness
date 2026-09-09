"""
probe_wa30_goal.py — find wa30's win condition by watching the c4/c7 counter.

Known from the drive probe: avatar is color-14 on a 4-cell lattice; a color-4
counter GROWS (1,2,3,...) and a color-7 counter SHRINKS (64,63,62,...) on only
SOME moves. This probe:
  * discovers each action's movement vector + classifies up/down/left/right,
  * snake-sweeps the reachable board (drive till blocked, step to next row, flip),
  * on every counter change prints the colors the avatar just moved ONTO (before-
    grid values of its newly covered cells) + the diff  -> what triggers a score,
  * drives until the counter stops changing (maxed) or LEVEL_COMPLETED (win diff).

Run: uv run python stage1/probe_wa30_goal.py --game wa30 --envs /test/ARC3/env_files
"""
from __future__ import annotations

import argparse
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects


def _names(env, g):
    return [a.name for a in env.available_actions(g)]


def _c14_centroid(g):
    """Centroid of the color-14 avatar; robust to shape-change (unlike object_diff.moved)."""
    cells = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 14]
    if not cells:
        return None
    return (sum(r for r, _ in cells) / len(cells), sum(c for _, c in cells) / len(cells))


def discover(ctor):
    """Return (avatar_colors, {action: (dr,dc)}) using color-14 centroid displacement.

    The avatar reconfigures shape on horizontal moves, so object_diff.moved misses
    left/right. Tracking the color-14 blob centroid catches all four directions.
    """
    env = ctor(); g = env.reset()
    names = _names(env, g)
    sig = {14}
    vecs = {}
    base = _c14_centroid(g)
    for name in names:
        e = ctor(); g0 = e.reset()
        b0 = _c14_centroid(g0)
        try:
            g1, _ = e.step(Action(name))
        except Exception:
            continue
        b1 = _c14_centroid(g1)
        if b0 is None or b1 is None:
            continue
        dr, dc = b1[0] - b0[0], b1[1] - b0[1]
        # round to nearest int direction; ignore sub-cell shape jitter on the minor axis
        vr = int(round(dr)); vc = int(round(dc))
        if abs(dr) < 0.6:  # treat small row wobble on horizontal moves as 0
            vr = 0
        if abs(dc) < 0.6:
            vc = 0
        if (vr, vc) != (0, 0):
            vecs[name] = (vr, vc)
    return sig, vecs, names


def av_cells(g, colors):
    return {c for o in extract_objects(g) if o.color in colors for c in o.cells}


def centroid(cells):
    if not cells:
        return None
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def counts(g):
    return int((g == 4).sum()), int((g == 7).sum())


def landed_on(before, after, colors):
    newly = av_cells(after, colors) - av_cells(before, colors)
    hist = {}
    for (r, c) in newly:
        v = int(before[r, c])
        if v not in colors and v != 0:
            hist[v] = hist.get(v, 0) + 1
    return hist


def summary(before, after):
    d = object_diff(before, after)
    return "; ".join(
        [f"c{o.color}/{o.size} moved {off}" for o, off in d.moved if o.size <= 60]
        + [f"c{o.color}/{o.size}+" for o in d.appeared if o.size <= 200]
        + [f"c{o.color}/{o.size}-" for o in d.disappeared if o.size <= 200]
    ) or "(no small change)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--max-resets", type=int, default=3)
    args = ap.parse_args()
    ctor = lambda: ArcEnv(args.game, args.envs)

    colors, vecs, names = discover(ctor)
    up = next((n for n, v in vecs.items() if v[0] < 0 and v[1] == 0), None)
    down = next((n for n, v in vecs.items() if v[0] > 0 and v[1] == 0), None)
    left = next((n for n, v in vecs.items() if v[1] < 0 and v[0] == 0), None)
    right = next((n for n, v in vecs.items() if v[1] > 0 and v[0] == 0), None)
    print(f"[avatar colors] {sorted(colors)}  [vectors] {vecs}")
    print(f"[dirs] up={up} down={down} left={left} right={right}")
    if not (left and right and (up or down)):
        print("could not classify all directions; aborting"); return

    env = ctor(); g = env.reset()
    c4, c7 = counts(g)
    pos = centroid(av_cells(g, colors))
    print(f"[start] avatar={None if pos is None else tuple(np.round(pos,1))} c4={c4} c7={c7}")

    horiz = [left, right]; hi = 0
    steps = resets = 0
    last_tick_step = 0

    def do(action):
        nonlocal g, c4, c7, steps, pos, last_tick_step, resets
        before = g
        g, term = env.step(Action(action)); steps += 1
        q = centroid(av_cells(g, colors))
        moved = pos is not None and q is not None and (abs(q[0]-pos[0]) + abs(q[1]-pos[1]) >= 1)
        n4, n7 = counts(g)
        if (n4, n7) != (c4, c7):
            last_tick_step = steps
            print(f"  [{steps:03d} TICK] {action} pos={None if q is None else tuple(np.round(q,1))} "
                  f"landed_on={landed_on(before, g, colors) or '{}'} "
                  f"c4:{c4}->{n4} c7:{c7}->{n7} | {summary(before, g)}")
            c4, c7 = n4, n7
        if term == "LEVEL_COMPLETED":
            print(f"  [{steps:03d}] *** LEVEL_COMPLETED — WIN DIFF: {object_diff(before, g).summary()}")
            return "win", moved
        if term == "GAME_OVER":
            print(f"  [{steps:03d}] GAME_OVER (resets={resets})")
            if resets < args.max_resets:
                g, _ = env.step(Action("RESET")); resets += 1
                return "reset", moved
            return "stop", moved
        if moved:
            pos = q
        return "ok", moved

    while steps < args.budget:
        # sweep horizontally until blocked
        hd = horiz[hi % 2]
        progressed = False
        while steps < args.budget:
            st, moved = do(hd)
            if st == "win" or st == "stop":
                print("[done]"); return
            if st == "reset":
                c4, c7 = counts(g); pos = centroid(av_cells(g, colors)); break
            if not moved:
                break
            progressed = True
        # step to next row (prefer down, else up)
        stepped = False
        for vd in [down, up]:
            if not vd:
                continue
            st, moved = do(vd)
            if st == "win" or st == "stop":
                print("[done]"); return
            if st == "reset":
                c4, c7 = counts(g); pos = centroid(av_cells(g, colors)); stepped = True; break
            if moved:
                stepped = True; break
        hi += 1
        if not progressed and not stepped:
            print("[swept board, no more reachable cells]"); break
        if steps - last_tick_step > 40:
            print("[counter idle for 40 steps — likely maxed or goal not counter-based]"); break

    print(f"[end] steps={steps} c4={c4} c7={c7} — "
          f"{'no LEVEL_COMPLETED' if steps>=args.budget else 'stopped'}")
    print("[done]")


if __name__ == "__main__":
    main()
