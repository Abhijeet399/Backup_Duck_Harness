"""
solve_live.py — end-to-end solver on a live game, using the pieces we validated.

For sp80 the two hard problems are SOLVED:
  * physics: QwenInducer induces step() that backtests 1.0 (HUD-masked)
  * goal:    goal_writer produced a goal_reached() validated 15/15 win, 565/565 non-win
             AND the win requires a SUBMIT action (ACTION5) once the goal state holds.

This wires them into a live play loop:
  1. explore a little to gather transitions
  2. induce physics model (masked)
  3. install the validated goal_reached as the model's goal_fn
  4. BFS-plan in the model to a state where goal_reached is True
  5. execute the plan for real, then press the SUBMIT action
  6. check LEVEL_COMPLETED

Run: uv run python stage1/solve_live.py --game sp80 \
        --submit ACTION5 --endpoint http://127.0.0.1:1234/v1
"""
from __future__ import annotations
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage1"))
sys.path.insert(0, "stage1")
import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, Transition, backtest, bfs_plan, WorldModel
from qwen_inducer import QwenInducer
from goal_writer import discover_goal

KEY = "/host-duck/ARC3-Inference/.cache/arc3_runtime/server-api-key"


def derive_hud_mask(env, noop, tries=4):
    g = env.reset()
    mask = np.zeros_like(g, dtype=bool)
    for _ in range(tries):
        g0 = g
        g, term = env.step(Action(noop))
        mask |= (g0 != g)
        if term == "GAME_OVER":
            break
    for r in sorted(set(int(r) for r, c in np.argwhere(mask))):
        mask[r, :] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sp80")
    ap.add_argument("--submit", default="ACTION5", help="the confirm/submit action")
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--arrows", default="ACTION1,ACTION2,ACTION3,ACTION4")
    args = ap.parse_args()
    arrows = args.arrows.split(",")

    ind = QwenInducer(endpoint=args.endpoint, verbose=True, key_file=KEY)

    # --- 1. HUD mask + explore for physics transitions ---
    e = ArcEnv(args.game, args.envs)
    mask = derive_hud_mask(e, args.submit)
    print(f"[solve] HUD mask: {int(mask.sum())} cells")

    e2 = ArcEnv(args.game, args.envs); g = e2.reset()
    hist = []
    for i in range(12):
        a = Action(arrows[i % len(arrows)]); g1, term = e2.step(a)
        hist.append(Transition(g, a, g1, term)); g = g1
        if term:
            break

    # --- 2. induce physics model ---
    ind.ignore_mask = mask
    models = ind(hist, None, None)
    if not models:
        print("[solve] physics induction failed"); return
    model = models[0]
    res = backtest(model, hist)
    print(f"[solve] physics backtest (masked): {res.exact_match_rate:.3f}")

    # --- 3. discover + install the validated goal predicate ---
    goal, gstats = discover_goal(args.game, args.submit, ind, max_tries=3, verbose=True)
    if not goal:
        print(f"[solve] goal discovery failed: {gstats}"); return
    goal_fn = goal["goal_reached"]
    print(f"[solve] goal validated: {goal['stats']}")

    # install as the model's goal
    solved_model = WorldModel(rules=model.rules, goal_fn=goal_fn,
                              terminal_fn=model.terminal_fn, background=model.background)
    solved_model.ignore_mask = mask

    # --- 4+5. plan to goal on a fresh env, execute, then SUBMIT ---
    er = ArcEnv(args.game, args.envs); grid = er.reset()
    action_gen = lambda gg: [Action(n) for n in arrows]

    # if we're already at the goal state, just submit; else BFS to it
    if goal_fn(grid):
        plan = []
        print("[solve] start already satisfies goal")
    else:
        plan = bfs_plan(solved_model, grid, action_gen, max_seconds=8.0)
    if plan is None:
        print("[solve] BFS found no plan to the goal state"); return
    print(f"[solve] plan: {len(plan)} moves, then submit {args.submit}")

    # execute movement plan for real
    for a in plan:
        grid, term = er.step(a)
        if term == "LEVEL_COMPLETED":
            print("  *** WON during movement (unexpected but great) ***"); return
        if term == "GAME_OVER":
            print("  GAME_OVER during movement"); return

    # verify goal_reached actually holds now (the model's plan should have achieved it)
    holds = bool(goal_fn(grid))
    print(f"[solve] after plan: goal_reached(real_grid)={holds}")

    # press the submit action to confirm the win
    grid, term = er.step(Action(args.submit))
    print(f"[solve] submit {args.submit} -> term={term}")
    if term == "LEVEL_COMPLETED":
        print(f"\n*** {args.game} SOLVED END-TO-END: physics + validated goal + BFS + submit ***")
    else:
        print(f"\n{args.game} not solved: reached goal={holds}, submit gave {term}")


if __name__ == "__main__":
    main()
