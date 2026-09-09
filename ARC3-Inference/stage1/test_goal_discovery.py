"""
test_goal_discovery.py — can candidate-goal exploration PROVOKE sp80's first win?

Induce the certified sp80 model, then run provoke_win: plan the avatar onto/next-to
each candidate target color and execute for real. If a real LEVEL_COMPLETED fires,
we've discovered the goal that was never observable from passive history.

Run: uv run python stage1/test_goal_discovery.py --endpoint http://127.0.0.1:1234/v1
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage1"))
sys.path.insert(0, "stage1")
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, Transition, backtest
from qwen_inducer import QwenInducer
from goal_discovery import provoke_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--game", default="sp80")
    args = ap.parse_args()

    # HUD mask
    e = ArcEnv(args.game, "/test/ARC3/env_files"); g = e.reset()
    mask = np.zeros_like(g, dtype=bool)
    for _ in range(4):
        g0 = g; g, term = e.step(Action("ACTION5")); mask |= (g0 != g)
        if term == "GAME_OVER": break
    for r in sorted(set(int(r) for r, c in np.argwhere(mask))): mask[r, :] = True

    # gather arrow transitions to induce physics
    e2 = ArcEnv(args.game, "/test/ARC3/env_files"); g = e2.reset()
    arrows = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]; hist = []
    for i in range(12):
        a = Action(arrows[i % 4]); g1, term = e2.step(a); hist.append(Transition(g, a, g1, term)); g = g1
        if term: break

    ind = QwenInducer(endpoint=args.endpoint, verbose=True)
    ind.ignore_mask = mask
    models = ind(hist, None, None)
    if not models:
        print("no model induced"); return
    model = models[0]
    res = backtest(model, hist)
    print(f"certified model backtest (masked): {res.exact_match_rate:.3f}")

    # avatar color = the color that moves in the transitions (color 9 for sp80)
    avatar_color = 9
    # env constructor so provoke_win gets a fresh clean start per candidate
    env_ctor = lambda: ArcEnv(args.game, "/test/ARC3/env_files")
    action_gen = lambda gg: [Action(n) for n in arrows]

    result = provoke_win(env_ctor, model, action_gen, avatar_color, verbose=True)
    if result:
        print(f"\n*** GOAL DISCOVERED: reach color {result['target_color']} "
              f"({result['goal_kind']}) — won in {len(result['transitions'])} real actions")
        print("This goal spec can now seed goal_fn for planning. Stage 1 goal-discovery works.")
    else:
        print("\nNo win provoked. sp80's goal may not be 'reach a colored target' — "
              "needs a richer candidate-goal generator (arrangement/count goals).")


if __name__ == "__main__":
    main()
