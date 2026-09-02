"""
example_smoke_test.py — prove the Stage 1 spine end-to-end with NO GPU and NO LLM.

A toy discrete game (avatar reaches target on a 6x6 grid with a wall) stands in for
a real ARC-AGI-3 game. We hand-write the correct world model and feed it through a
stub Inducer, then run the real controller. If this solves the game, then
round_robin_explore -> backtest -> bfs_plan -> execute_and_verify -> the governor
are all wired correctly, and the only thing left to build for real games is:
  (a) the Env adapter onto arc_agi, and
  (b) the Qwen Inducer that PRODUCES a model like the one hand-written below.

Run:  python example_smoke_test.py
"""

from __future__ import annotations
import numpy as np

from stage1_core import Action, WorldModel
from stage1_controller import Budget, play_game

H, W = 6, 6
AVATAR, WALL, TARGET, BG = 2, 1, 3, 0
MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
SIMPLE = [Action(n) for n in MOVES]


def _initial() -> np.ndarray:
    g = np.zeros((H, W), dtype=np.int8)
    g[1:5, 3] = WALL          # a wall that BFS must route around
    g[5, 0] = AVATAR
    g[0, 5] = TARGET
    return g


# --------------------------------------------------------------------------- #
# Toy environment (stands in for arc_agi). It is game-general test scaffolding,
# not a real game — this is exactly the kind of local simulator the world model
# is meant to replace at eval time.
# --------------------------------------------------------------------------- #
class ToyEnv:
    def __init__(self):
        self.grid = _initial()

    def reset(self) -> np.ndarray:
        self.grid = _initial()
        return self.grid.copy()

    def step(self, a: Action):
        if a.name == "RESET":
            self.grid = _initial()
            return self.grid.copy(), None
        r, c = map(int, np.argwhere(self.grid == AVATAR)[0])
        dr, dc = MOVES[a.name]
        nr, nc = r + dr, c + dc
        out = self.grid.copy()
        if 0 <= nr < H and 0 <= nc < W and self.grid[nr, nc] != WALL:
            out[r, c] = BG
            out[nr, nc] = AVATAR          # overwrites TARGET if stepping onto it
        self.grid = out
        terminal = "LEVEL_COMPLETED" if not (self.grid == TARGET).any() else None
        return self.grid.copy(), terminal

    def available_actions(self, grid):
        return SIMPLE


# --------------------------------------------------------------------------- #
# The correct world model, hand-written. One rule per move action. Each rule
# mirrors the env's movement exactly. This is the artifact the Qwen Inducer will
# eventually have to synthesize from observed transitions.
# --------------------------------------------------------------------------- #
def _move_rule(name, dr, dc):
    def rule(grid, a):
        if a.name != name:
            return None
        pos = np.argwhere(grid == AVATAR)
        if len(pos) == 0:
            return grid.copy()
        r, c = map(int, pos[0])
        nr, nc = r + dr, c + dc
        out = grid.copy()
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != WALL:
            out[r, c] = BG
            out[nr, nc] = AVATAR
        return out
    return rule


def correct_model() -> WorldModel:
    rules = [_move_rule(n, dr, dc) for n, (dr, dc) in MOVES.items()]
    goal_fn = lambda g: not (g == TARGET).any()   # target covered by avatar
    return WorldModel(rules=rules, goal_fn=goal_fn, background=BG)


def stub_inducer(history, current, divergence):
    # A real inducer runs Qwen here. The stub just returns the known-correct model.
    return [correct_model()]


if __name__ == "__main__":
    env = ToyEnv()
    budget = Budget(wall_clock_s=2.0, max_llm_calls=2, max_real_actions=60, reserve_s=0.3)
    result = play_game(
        env=env,
        inducer=stub_inducer,
        probe_actions=SIMPLE,
        action_generator=lambda g: SIMPLE,
        budget=budget,
    )
    print("status:          ", result.status)
    print("levels_completed:", result.levels_completed)
    print("transitions seen:", len(result.history))
    print("budget:          ", result.budget)
    assert result.levels_completed >= 1, "SPINE BROKEN: hand-written model failed to solve toy game"
    print("\nOK — spine solves the toy game. Plumbing is correct.")
    # Note: after the level is solved the loop may spin briefly on an empty plan
    # until the short wall clock expires; in a real game LEVEL_COMPLETED yields the
    # NEXT level's frame, so is_goal becomes false again and planning continues.
