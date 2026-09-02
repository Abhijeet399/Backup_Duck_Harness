"""
env_adapter.py — bridge the harness's taaf.game.Game + arcengine to the Stage 1
Env protocol.

Written against inference/framework/solver.py, so seams 2 (action/step) and 3
(frame/terminal) are filled from the REAL API, not templates:

  * frame grid        : game.current_state.frame.data           (solver.py:96)
  * levels_completed  : game.current_state.levels_completed      (solver.py:103)
  * just_won_level    : new_state.just_won_level                 (solver.py:705)
  * terminal state    : current_state.raw.state == GameState.*   (solver.py:139-144)
  * available actions : current_state.available_actions -> GameAction.from_id (111-113)
  * step              : game.execute_action(ActionInput(...), ...)(solver.py:683)
  * RESET             : ActionInput(id=GameAction.RESET, data={}) (solver.py:664)

Terminal logic keys off the LEVEL COUNTER, not WIN: WIN means the whole game is
solved; mid-run level completion shows as levels_completed advancing.

Only SEAM 1 remains — how run.py builds a taaf.game.Game for a game_id from an
ArcadeSpec(environments_dir=...). Paste that construction into build_game().
"""
from __future__ import annotations

# If your build_game() ends up needing inference.* helpers, uncomment this shim
# so `import inference...` resolves when running `python stage1/run_stage1.py`:
# import os, sys
# _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if _REPO not in sys.path:
#     sys.path.insert(0, _REPO)

import numpy as np
import arcengine
import taaf.game

from stage1_core import Action, Grid


# --- SEAM 1: build a taaf.game.Game for one game id (the only unfilled piece) --
def build_game(game_id: str, environments_dir: str) -> "taaf.game.Game":
    """Return ONE fresh taaf.game.Game for `game_id`. Mirrors run.py:_make_games:
    a GameAPI over an ArcadeSpec(environments_dir=...) IS a taaf.game.Game."""
    import taaf.game_api as game_api
    spec = (game_api.ArcadeSpec(environments_dir=str(environments_dir))
            if environments_dir else None)
    if spec is None:
        return game_api.GameAPI(env_name=game_id)
    return game_api.GameAPI(env_name=game_id, arcade_spec=spec)


# --- frame / terminal / action translation (filled from solver.py) -----------
def _grid(state) -> Grid:
    arr = np.asarray(state.frame.data, dtype=np.int8)   # tuple-of-tuples -> 2-D
    while arr.ndim > 2:                                  # collapse layers/sequence
        arr = arr[-1]
    return arr


def _terminal(state, prev_levels: int):
    raw = state.raw.state
    if raw == arcengine.GameState.GAME_OVER:
        return "GAME_OVER"
    if raw == arcengine.GameState.WIN \
       or getattr(state, "just_won_level", False) \
       or int(state.levels_completed) > prev_levels:
        return "LEVEL_COMPLETED"
    return None


def _action_names(state) -> list[str]:
    names = []
    for aid in state.available_actions:
        try:
            names.append(arcengine.GameAction.from_id(int(aid)).name)
        except Exception:
            pass
    return names


def _to_action_input(a: Action) -> "arcengine.ActionInput":
    gid = arcengine.GameAction.from_name(a.name)          # a.name is already engine-form
    data: dict = {}
    if gid == arcengine.GameAction.ACTION6 and a.x is not None:
        # solver.py clamps row/col to 0..63; not needed for arrow-only games (ls20).
        data = {"row": max(0, min(63, int(a.y))), "col": max(0, min(63, int(a.x)))}
    return arcengine.ActionInput(id=gid, data=data)


class ArcEnv:
    def __init__(self, game_id: str, environments_dir: str):
        self.game = build_game(game_id, environments_dir)
        # taaf.game.Game asserts current_state requires start_game() first; the
        # harness starts games in its play flow, so the standalone adapter must too.
        self.game.start_game()
        self._levels = int(self.game.current_state.levels_completed)

    def reset(self) -> Grid:
        # The game starts at its initial state on construction; no RESET needed at
        # game start (RESET is issued by the controller only after GAME_OVER).
        st = self.game.current_state
        self._levels = int(st.levels_completed)
        return _grid(st)

    def step(self, a: Action):
        ai = _to_action_input(a)
        new_state = self.game.execute_action(ai, generated_tokens=0, uncached_input_tokens=0)
        term = _terminal(new_state, self._levels)
        self._levels = int(new_state.levels_completed)
        return _grid(new_state), term

    def available_actions(self, grid) -> list[Action]:
        return [Action(n) for n in _action_names(self.game.current_state) if n != "RESET"]
