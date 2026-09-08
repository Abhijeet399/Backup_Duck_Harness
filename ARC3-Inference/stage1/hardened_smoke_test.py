"""
hardened_smoke_test.py — prove the Stage 1 spine on the mechanics REAL games have:
COORDINATE (click) actions + MULTI-OBJECT transitions + a PRUNED action generator.

The original example_smoke_test.py only tested a single rigid avatar with 4 fixed
discrete actions. The measured reality of the ARC-AGI-3 public 25 is: 19/25 are
mouse/click games, and every non-mouse game moves 2+ objects per action. So the
original test passed while every real game stalled — it never exercised clicks or
multi-object state. This test closes that gap with a synthetic we fully control.

Toy game "push2": two blocks (colors 5 and 6) on an 8x8 grid. Clicking a block's
cell (ACTION6 @ (x=col, y=row)) pushes THAT block one cell toward the nearest edge
it's being nudged to — here, clicking pushes the block one cell RIGHT (toward its
target column). Each block has its own target cell (colors 3 and 4). Goal: both
blocks sit on their targets. This forces:
  * coordinate actions  (Action.name=='ACTION6', x/y set),
  * a PRUNED generator   (only the 2 block cells are worth clicking, not all 64),
  * a MULTI-OBJECT model  (backtest must reproduce both blocks' positions exactly).

Run:  uv run python /tmp/hardened_smoke_test.py     (from stage1/ dir, or with sys.path)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage1"))
sys.path.insert(0, "stage1")
import numpy as np

from stage1_core import Action, WorldModel
from stage1_controller import Budget, play_game

H, W = 8, 8
BG = 0
BLOCK_A, BLOCK_B = 5, 6          # the two pushable blocks
TARG_A, TARG_B = 3, 4            # their destination markers
A_START, A_TARG = (2, 1), (2, 6) # (row, col)
B_START, B_TARG = (5, 2), (5, 6)


def _initial() -> np.ndarray:
    g = np.zeros((H, W), dtype=np.int8)
    g[A_TARG] = TARG_A
    g[B_TARG] = TARG_B
    g[A_START] = BLOCK_A
    g[B_START] = BLOCK_B
    return g


def _find(g, color):
    p = np.argwhere(g == color)
    return None if len(p) == 0 else (int(p[0][0]), int(p[0][1]))


def _push_right(g, block_color, targ_color):
    """Push `block_color` one cell right, unless already on its target column."""
    out = g.copy()
    pos = _find(g, block_color)
    if pos is None:
        return out
    r, c = pos
    if c >= W - 1:
        return out
    nc = c + 1
    # leave the target marker if the block steps off it; here targets are to the right
    out[r, c] = TARG_A if (r, c) == A_TARG else (TARG_B if (r, c) == B_TARG else BG)
    # if destination is the target marker, block lands ON it (block color shows)
    out[r, nc] = block_color
    return out


# --------------------------------------------------------------------------- #
# Toy env: click a block's cell to push THAT block right. Multi-object state.
# --------------------------------------------------------------------------- #
class PushEnv:
    def __init__(self):
        self.grid = _initial()

    def reset(self):
        self.grid = _initial()
        return self.grid.copy()

    def step(self, a: Action):
        if a.name == "RESET":
            self.grid = _initial()
            return self.grid.copy(), None
        out = self.grid
        if a.name == "ACTION6" and a.x is not None:
            r, c = a.y, a.x                       # y=row, x=col  (engine convention)
            clicked = self.grid[r, c] if (0 <= r < H and 0 <= c < W) else BG
            if clicked == BLOCK_A:
                out = _push_right(self.grid, BLOCK_A, TARG_A)
            elif clicked == BLOCK_B:
                out = _push_right(self.grid, BLOCK_B, TARG_B)
            else:
                out = self.grid.copy()            # clicking empty/target = no-op
        self.grid = out
        a_done = _find(out, BLOCK_A) == A_TARG
        b_done = _find(out, BLOCK_B) == B_TARG
        terminal = "LEVEL_COMPLETED" if (a_done and b_done) else None
        return out.copy(), terminal

    def available_actions(self, grid):
        return _gen_clicks(grid)


# --------------------------------------------------------------------------- #
# PRUNED action generator: only click cells that currently hold a block.
# This is the 4096 -> handful pruning every real mouse game needs. It returns
# 2 actions here, not 64.
# --------------------------------------------------------------------------- #
def _gen_clicks(grid) -> list[Action]:
    acts = []
    for color in (BLOCK_A, BLOCK_B):
        pos = _find(grid, color)
        if pos is not None:
            r, c = pos
            acts.append(Action("ACTION6", x=c, y=r))   # x=col, y=row
    return acts


# --------------------------------------------------------------------------- #
# Correct multi-object world model, hand-written. One rule handles a click on
# either block. Must reproduce BOTH blocks' positions exactly for backtest==1.0.
# --------------------------------------------------------------------------- #
def _click_rule(grid, a: Action):
    if a.name != "ACTION6" or a.x is None:
        return None
    r, c = a.y, a.x
    if not (0 <= r < H and 0 <= c < W):
        return grid.copy()
    clicked = grid[r, c]
    if clicked == BLOCK_A:
        return _push_right(grid, BLOCK_A, TARG_A)
    if clicked == BLOCK_B:
        return _push_right(grid, BLOCK_B, TARG_B)
    return grid.copy()


def correct_model() -> WorldModel:
    goal_fn = lambda g: (_find(g, BLOCK_A) == A_TARG) and (_find(g, BLOCK_B) == B_TARG)
    return WorldModel(rules=[_click_rule], goal_fn=goal_fn, background=BG)


def stub_inducer(history, current, divergence):
    return [correct_model()]


if __name__ == "__main__":
    env = PushEnv()
    probe = _gen_clicks(_initial())
    budget = Budget(wall_clock_s=5.0, max_llm_calls=2, max_real_actions=60, reserve_s=0.3)
    result = play_game(
        env=env,
        inducer=stub_inducer,
        probe_actions=probe,
        action_generator=_gen_clicks,
        budget=budget,
    )
    print("status:          ", result.status)
    print("levels_completed:", result.levels_completed)
    print("transitions seen:", len(result.history))
    print("budget:          ", result.budget)
    # Each block starts 5 cols left of its target -> needs 5 clicks each = 10 actions.
    assert result.levels_completed >= 1, (
        "SPINE GAP: multi-object click game not solved — the spine cannot handle "
        "coordinate actions and/or multi-object backtest. THIS is what real games need."
    )
    print("\nOK — spine solves a MULTI-OBJECT CLICK game.")
    print("Validated: coordinate actions, pruned action_generator, multi-object backtest.")
