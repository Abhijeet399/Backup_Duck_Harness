"""
probe_ls20_drive.py — corrected ls20 collision/range map.

Fixes the seek probe's bug: the avatar is tracked by its DISTINCTIVE moving blob
(color-12 size~10 component), not by raw color 9/12 (which also appear as static
board cells and polluted the centroid). Drives each direction in a straight line
from a fresh start; prints true avatar delta + a noise-filtered diff (background
field size>200 and timer-bar color 11 suppressed).

Reveals, per direction: does the avatar move a full 5 each press, where does it
stop, and does anything interact on the way.

Run:  uv run python stage1/probe_ls20_drive.py --game ls20 --envs /test/ARC3/env_files
"""
from __future__ import annotations

import argparse
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects

DIRS = [("ACTION1", "up"), ("ACTION2", "down"), ("ACTION3", "left"), ("ACTION4", "right")]


def avatar_pos(g):
    """Centroid of the avatar's distinctive component: prefer color-12 size~10,
    fall back to color-9 size~15."""
    best = None
    for o in extract_objects(g):
        score = None
        if o.color == 12:
            score = abs(o.size - 10)
        elif o.color == 9:
            score = abs(o.size - 15) + 100  # only if no color-12 blob found
        if score is not None and (best is None or score < best[1]):
            r0, c0, r1, c1 = o.bbox
            best = (np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0]), score)
    return None if best is None else best[0]


def meaningful(before, after) -> str:
    d = object_diff(before, after)
    parts = []
    for o, off in d.moved:
        if o.size <= 40:
            parts.append(f"c{o.color}/{o.size} moved {off}")
    for o in d.appeared:
        if o.color not in (3, 11) and o.size <= 200:
            parts.append(f"c{o.color}/{o.size} appeared@{o.bbox[:2]}")
    for o in d.disappeared:
        if o.color not in (3, 11) and o.size <= 200:
            parts.append(f"c{o.color}/{o.size} gone@{o.bbox[:2]}")
    return "; ".join(parts) or "(no small-object change)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--presses", type=int, default=15)
    args = ap.parse_args()

    for action_name, label in DIRS:
        env = ArcEnv(args.game, args.envs)   # fresh start per direction
        g = env.reset()
        p = avatar_pos(g)
        print(f"\n=== drive {label} ({action_name}) — start avatar "
              f"{None if p is None else tuple(np.round(p,1))} ===")
        for i in range(args.presses):
            before = g
            g, term = env.step(Action(action_name))
            q = avatar_pos(g)
            delta = None if (p is None or q is None) else (round(q[0]-p[0], 1), round(q[1]-p[1], 1))
            print(f"  {i+1:2d} avatar={None if q is None else tuple(np.round(q,1))} "
                  f"delta={delta} term={term} | {meaningful(before, g)}")
            if term == "LEVEL_COMPLETED":
                print(f"     *** LEVEL_COMPLETED driving {label}")
                break
            if term == "GAME_OVER":
                print("     (GAME_OVER)")
                break
            if delta is not None and abs(delta[0]) < 1 and abs(delta[1]) < 1:
                print(f"     stopped after {i+1} presses (wall/edge)")
                break
            p = q
    print("\n[done]")


if __name__ == "__main__":
    main()
