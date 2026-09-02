"""
probe_env.py — validate the adapter against a REAL game AND reveal its mechanics.

Constructs ArcEnv(game), resets, then round-robins the available actions, printing
the object-level diff for each transition. Two purposes:
  1. If this runs without a traceback and shows sensible frames, seams 1-3 of the
     adapter are correct against the live environment.
  2. The per-action diffs are the raw signal for hand-writing the world model
     (e.g. build_ls20_model) — paste the output back to design the rules.

Run with the harness interpreter (arcengine lives in .venv):
  uv run python stage1/probe_env.py --game ls20 --envs /test/ARC3/env_files --passes 3
"""
from __future__ import annotations

import argparse
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects


def describe(g) -> str:
    colors = sorted(int(c) for c in np.unique(g))
    return f"shape={tuple(g.shape)} colors={colors} objects={len(extract_objects(g))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    env = ArcEnv(args.game, args.envs)
    g = env.reset()
    print(f"[init]    {describe(g)}")

    acts = env.available_actions(g)
    print(f"[actions] {[a.name for a in acts]}")
    if not acts:  # some games need one RESET to seed the first playable frame
        print("[note]    no actions at start -> sending RESET to seed")
        g, t = env.step(Action('RESET'))
        print(f"[seed]    {describe(g)} term={t}")
        acts = env.available_actions(g)
        print(f"[actions] {[a.name for a in acts]}")

    step = 0
    for _ in range(args.passes):
        for a in acts:
            before = g
            g, term = env.step(a)
            step += 1
            d = object_diff(before, g)
            print(f"[{step:03d}] {a.name:8s} -> {d.summary()} "
                  f"| changed_cells={d.changed_cells} term={term} levels={env._levels}")
            if term == "GAME_OVER":
                g, _ = env.step(Action('RESET'))
                print(f"        (GAME_OVER -> RESET) {describe(g)}")
            elif term == "LEVEL_COMPLETED":
                print(f"        (LEVEL_COMPLETED at step {step})")
    print("[done]")


if __name__ == "__main__":
    main()
