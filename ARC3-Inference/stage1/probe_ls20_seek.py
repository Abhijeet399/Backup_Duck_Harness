"""
probe_ls20_seek.py — use the learned movement model to drive ls20's avatar into
the static features, to surface the two things round-robin didn't: COLLISION
behaviour (does something block the avatar?) and the WIN condition.

Movement learned from the round-robin probe (5-cell steps):
  ACTION1=(-5,0) up   ACTION2=(+5,0) down   ACTION3=(0,-5) left   ACTION4=(0,+5) right

For each candidate target (colors 1/4/5/8), greedily pick the action that most
reduces Manhattan distance from the avatar centroid, step, then compare the
avatar's ACTUAL delta to the expected one. A mismatch = collision (BLOCKED).
On LEVEL_COMPLETED it prints the full object diff — that's the goal signal.

Run:  uv run python stage1/probe_ls20_seek.py --game ls20 --envs /test/ARC3/env_files
"""
from __future__ import annotations

import argparse
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects

AVATAR_COLORS = (9, 12)
CAND_COLORS = (1, 4, 5, 8)
BG_LIKE = (0, 3, 11)           # background field + step-timer bar
VEC = {"ACTION1": (-5, 0), "ACTION2": (5, 0), "ACTION3": (0, -5), "ACTION4": (0, 5)}


def avatar_centroid(g):
    cells = np.argwhere(np.isin(g, AVATAR_COLORS))
    return cells.mean(axis=0) if len(cells) else None


def candidates(g):
    out = []
    for o in extract_objects(g):
        if o.color in CAND_COLORS:
            r0, c0, r1, c1 = o.bbox
            out.append((o.color, np.array([(r0 + r1) / 2, (c0 + c1) / 2]), o.size))
    return out


def best_action(av, tgt):
    return min(VEC.items(),
               key=lambda kv: abs(av[0] + kv[1][0] - tgt[0]) + abs(av[1] + kv[1][1] - tgt[1]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--max-steps", type=int, default=80)
    args = ap.parse_args()

    env = ArcEnv(args.game, args.envs)
    g = env.reset()
    av = avatar_centroid(g)
    cands = candidates(g)
    print(f"[avatar] centroid={None if av is None else tuple(np.round(av,1))}")
    print(f"[targets] " + ", ".join(f"color{c} @({p[0]:.0f},{p[1]:.0f}) size{s}" for c, p, s in cands))

    order = sorted(range(len(cands)), key=lambda i: abs(av[0]-cands[i][1][0])+abs(av[1]-cands[i][1][1]))
    blocked_streak = 0
    for ti in order:
        color, tgt, _ = cands[ti]
        print(f"\n=== seeking color{color} @({tgt[0]:.0f},{tgt[1]:.0f}) ===")
        for _ in range(args.max_steps):
            a = best_action(av, tgt)
            exp = VEC[a]
            before, before_av = g, av
            g, term = env.step(Action(a))
            av = avatar_centroid(g)
            act = None if (av is None or before_av is None) else (av[0]-before_av[0], av[1]-before_av[1])
            blk = act is not None and (abs(act[0]-exp[0]) > 1 or abs(act[1]-exp[1]) > 1)
            blocked_streak = blocked_streak + 1 if blk else 0
            tag = " BLOCKED" if blk else ""
            actstr = "None" if act is None else f"({act[0]:.0f},{act[1]:.0f})"
            print(f"  {a:8s} exp={exp} act={actstr} term={term} levels={env._levels}{tag}")
            if term == "LEVEL_COMPLETED":
                print("  *** LEVEL_COMPLETED — goal diff:")
                print("      " + object_diff(before, g).summary())
                return
            if term == "GAME_OVER":
                print("  (GAME_OVER -> RESET)")
                g, _ = env.step(Action("RESET")); av = avatar_centroid(g); break
            d = abs(av[0]-tgt[0]) + abs(av[1]-tgt[1])
            if d <= 5 or blocked_streak >= 3:
                print(f"  (reached vicinity d={d:.0f} or stuck) -> next target")
                break
    print("\n[done] no LEVEL_COMPLETED reached; see BLOCKED lines for collision behaviour")


if __name__ == "__main__":
    main()
