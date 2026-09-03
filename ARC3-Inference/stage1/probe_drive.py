"""
probe_drive.py — game-general straight-line drive probe with automatic avatar
discovery. Unlike probe_ls20_drive (hardcoded to ls20 colours), this finds the
avatar as whatever small object(s) move under a simple action, so it works on any
game. Maps collision/range per direction and flags any LEVEL_COMPLETED + diff.

Run:  uv run python stage1/probe_drive.py --game wa30 --envs /test/ARC3/env_files
"""
from __future__ import annotations

import argparse
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects


def _small_movers(before, after, maxsize=60):
    d = object_diff(before, after)
    return [(o, off) for o, off in d.moved if o.size <= maxsize]


def _names(env, g):
    return [a.name for a in env.available_actions(g)]  # RESET already excluded


def discover_signature(ctor):
    """Probe each simple action once from a fresh start; the first that moves a
    small object defines the avatar signature {(color, size), ...}."""
    env = ctor(); g = env.reset()
    for name in _names(env, g):
        e2 = ctor(); g0 = e2.reset()
        try:
            g1, _ = e2.step(Action(name))
        except Exception:
            continue  # mouse/coord action — needs (x,y)
        m = _small_movers(g0, g1)
        if m:
            return {(o.color, o.size) for o, _ in m}, name
    return set(), None


def centroid(g, sig):
    cells = [c for o in extract_objects(g) if (o.color, o.size) in sig for c in o.cells]
    if not cells:
        return None
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return np.array([sum(rs) / len(rs), sum(cs) / len(cs)])


def meaningful(before, after):
    d = object_diff(before, after); parts = []
    for o, off in d.moved:
        if o.size <= 60:
            parts.append(f"c{o.color}/{o.size} moved {off}")
    for o in d.appeared:
        if o.size <= 200:
            parts.append(f"c{o.color}/{o.size}+")
    for o in d.disappeared:
        if o.size <= 200:
            parts.append(f"c{o.color}/{o.size}-")
    return "; ".join(parts) or "(no small change)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--presses", type=int, default=15)
    args = ap.parse_args()
    ctor = lambda: ArcEnv(args.game, args.envs)

    sig, via = discover_signature(ctor)
    print(f"[avatar signature] {sorted(sig) if sig else 'NONE'} (found via {via})")
    if not sig:
        print("No simple action moves an object — coordinate/state-gated game, not a "
              "clean movement target."); return

    env0 = ctor(); g0 = env0.reset()
    for name in _names(env0, g0):
        env = ctor()
        g = env.reset()
        p = centroid(g, sig)
        print(f"\n=== {name} — start {None if p is None else tuple(np.round(p,1))} ===")
        for i in range(args.presses):
            before = g
            try:
                g, term = env.step(Action(name))
            except Exception:
                print(f"  {name} needs coords (mouse) — skipping"); break
            q = centroid(g, sig)
            delta = None if (p is None or q is None) else (round(q[0]-p[0], 1), round(q[1]-p[1], 1))
            print(f"  {i+1:2d} pos={None if q is None else tuple(np.round(q,1))} "
                  f"delta={delta} term={term} | {meaningful(before, g)}")
            if term == "LEVEL_COMPLETED":
                print(f"     *** LEVEL_COMPLETED — WIN DIFF: {object_diff(before, g).summary()}")
                break
            if term == "GAME_OVER":
                print("     (GAME_OVER)"); break
            if delta is not None and abs(delta[0]) < 1 and abs(delta[1]) < 1:
                print(f"     stopped after {i+1} (wall/edge)"); break
            p = q
    print("\n[done]")


if __name__ == "__main__":
    main()
