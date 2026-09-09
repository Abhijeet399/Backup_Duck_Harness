"""
goal_discovery.py — provoke a win to discover the goal.

The world model's physics are certified (backtest 1.0), but its goal_fn is a
guess and BFS finds no plan. We can't INDUCE the goal from history that contains
no win. So instead we PROVOKE one: enumerate candidate target objects, use the
certified physics to plan the avatar onto each candidate, execute in the real
env, and watch for LEVEL_COMPLETED. The first candidate that wins IS the goal.

This reuses the certified model (physics) + bfs_plan + the real env. The only new
idea is: substitute a candidate goal_fn into a copy of the model and search toward
it. Once a real win is observed, the goal is known and normal induction can take
over for later levels.
"""
from __future__ import annotations
import copy
import numpy as np
from typing import Optional

from stage1_core import Action, WorldModel, bfs_plan, extract_objects, grid_key


def _avatar_color(history_diffs_or_grid, model) -> Optional[int]:
    """Best guess at the avatar color: the color that MOVES. We pass it in from
    the caller who knows it (the mover in transitions). Fallback: None."""
    return getattr(model, "_avatar_color", None)


def candidate_target_colors(grid, background, avatar_color, ignore_mask=None):
    """Colors that are plausible goal targets: not background, not avatar, not HUD.
    Ranked rare-first (few cells => more likely a specific target/exit)."""
    counts = {}
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if ignore_mask is not None and ignore_mask[r, c]:
                continue
            v = int(grid[r, c])
            counts[v] = counts.get(v, 0) + 1
    skip = {int(background), int(avatar_color)}
    cands = [(col, n) for col, n in counts.items() if col not in skip]
    cands.sort(key=lambda x: x[1])          # rare first
    return [col for col, n in cands]


def _reach_goal_fn(avatar_color, target_color):
    """Goal = the avatar overlaps ANY cell of the target color (i.e. the target
    color's count drops, meaning the avatar covered/reached it)."""
    def goal(g):
        # avatar has reached target if no target cells remain OR avatar cells and
        # target cells are adjacent/overlapping. Simplest robust proxy: the target
        # color count decreased from its start is hard to track statelessly, so use
        # overlap: any avatar cell equals a former target position is captured by
        # target color disappearing. We approximate: target reached when the target
        # color is fully consumed (count == 0). Works for "eat the target" goals.
        return not (g == target_color).any()
    return goal


def _reach_adjacent_goal_fn(avatar_color, target_color):
    """Alternative goal: avatar is ADJACENT to (touching) the target color.
    Covers goals where you must move next to a target rather than onto it."""
    def goal(g):
        av = (g == avatar_color)
        tg = (g == target_color)
        if not tg.any() or not av.any():
            return not tg.any()
        # dilate avatar by 1 and check intersection with target
        a = av
        up = np.zeros_like(a); up[:-1, :] = a[1:, :]
        dn = np.zeros_like(a); dn[1:, :] = a[:-1, :]
        lf = np.zeros_like(a); lf[:, :-1] = a[:, 1:]
        rt = np.zeros_like(a); rt[:, 1:] = a[:, :-1]
        touch = a | up | dn | lf | rt
        return bool((touch & tg).any())
    return goal


def _model_with_goal(model, goal_fn) -> WorldModel:
    """Copy the certified model but swap in a candidate goal_fn. Physics unchanged."""
    m = WorldModel(rules=model.rules, goal_fn=goal_fn,
                   terminal_fn=model.terminal_fn, background=model.background)
    m.ignore_mask = getattr(model, "ignore_mask", None)
    return m


def provoke_win(env_ctor, model, action_generator, avatar_color,
                budget=None, max_targets=6, bfs_seconds=6.0, verbose=True):
    """Try to trigger the real LEVEL_COMPLETED by planning the avatar onto/next-to
    candidate target colors. Each candidate is tested from a FRESH env (env_ctor())
    so no candidate's execution contaminates the next.

    env_ctor: zero-arg callable returning a fresh ArcEnv (so we get a clean start).
    Returns on success: {"target_color","goal_kind","plan_len","transitions"} or None.
    """
    from stage1_core import Transition
    bg = model.background
    mask = getattr(model, "ignore_mask", None)
    # derive candidate targets from a clean start grid
    e0 = env_ctor(); start = e0.reset()
    targets = candidate_target_colors(start, bg, avatar_color, mask)[:max_targets]
    if verbose:
        print(f"[provoke] candidate target colors (rare-first): {targets}")

    for tc in targets:
        for kind, gf in (("consume", _reach_goal_fn(avatar_color, tc)),
                         ("adjacent", _reach_adjacent_goal_fn(avatar_color, tc))):
            # fresh env + clean start for THIS candidate
            env = env_ctor(); grid = env.reset()
            m = _model_with_goal(model, gf)
            plan = bfs_plan(m, grid, action_generator, max_seconds=bfs_seconds)
            if not plan:
                if verbose:
                    print(f"[provoke] target color {tc} ({kind}): no model-plan (unreachable)")
                continue
            if verbose:
                print(f"[provoke] target color {tc} ({kind}): model-plan {len(plan)} actions; executing")
            g = grid; trans = []; won = False
            for a in plan:
                nxt, term = env.step(a)
                if budget is not None:
                    budget.act()
                trans.append(Transition(g, a, nxt, term))
                g = nxt
                if term == "LEVEL_COMPLETED":
                    won = True; break
                if term == "GAME_OVER":
                    break
            if won:
                if verbose:
                    print(f"[provoke] *** WIN by reaching color {tc} ({kind}) in {len(trans)} actions!")
                return {"target_color": int(tc), "goal_kind": kind,
                        "plan_len": len(plan), "transitions": trans}
    if verbose:
        print("[provoke] no candidate target produced a win")
    return None
