"""
env_adapter.py — bridge arc_agi's EnvironmentWrapper to the Stage 1 Env protocol.

There are THREE seams that depend on your exact arc_agi version. They are marked
TODO. Your existing harness (the code `make interactive` runs) already does all
three — building the env, dispatching an action, and reading a frame — so the
fastest correct path is to copy those exact calls into the spots below rather
than guessing.
"""
from __future__ import annotations

import numpy as np

from stage1_core import Action, Grid


# --- SEAM 1: build a wrapper for one game id ---------------------------------
def build_wrapper(game_id: str, environments_dir: str):
    """Return an arc_agi EnvironmentWrapper for `game_id`, loaded from local files.
    TODO: replace the body with the construction your harness already uses
    (some LocalEnvironmentWrapper / Arcade over `environments_dir`)."""
    raise NotImplementedError(
        "Wire this to your harness's env construction. Grep your tool_agent / "
        "interactive path for where it instantiates the environment from "
        "ENVIRONMENTS_DIR and copy it here."
    )


# --- SEAM 3: frame + terminal translation ------------------------------------
def _frame_to_grid(fd) -> Grid:
    """FrameDataRaw -> 2-D int8 grid. Frames can arrive as a SEQUENCE (animation)
    and/or as layers; we take the last 2-D plane as the settled frame."""
    arr = np.asarray(fd.frame, dtype=np.int8)   # TODO: confirm the field is `.frame`
    while arr.ndim > 2:
        arr = arr[-1]
    return arr


def _terminal(fd, prev_levels: int):
    """-> None | 'LEVEL_COMPLETED' | 'GAME_OVER'. Level completion is detected by
    the levels_completed counter advancing; whole-game WIN also counts as a
    completed level. TODO: confirm state enum spelling in your arc_agi version."""
    levels = int(getattr(fd, "levels_completed", prev_levels))
    state = str(getattr(fd, "state", "")).upper()
    if levels > prev_levels or "WIN" in state:
        return "LEVEL_COMPLETED"
    if "GAME_OVER" in state or "OVER" in state:
        return "GAME_OVER"
    return None


class ArcEnv:
    def __init__(self, game_id: str, environments_dir: str):
        self.w = build_wrapper(game_id, environments_dir)
        self._levels = 0
        self._by_name: dict[str, object] = {}

    def _sync_actions(self):
        self._by_name = {a.name: a for a in self.w.action_space}

    def reset(self) -> Grid:
        fd = self.w.reset()
        self._sync_actions()
        self._levels = int(getattr(fd, "levels_completed", 0))
        return _frame_to_grid(fd)

    def step(self, a: Action):
        ga = self._by_name.get(a.name)
        if ga is None:
            self._sync_actions()
            ga = self._by_name[a.name]
        # --- SEAM 2: attach coords for ACTION6 and dispatch exactly as your
        # harness does. Confirm the coord field names and the step() signature. ---
        if a.x is not None:
            ga.set_data({"x": a.x, "y": a.y})   # TODO: confirm coord keys
        fd = self.w.step(ga)                     # TODO: confirm step() call form
        term = _terminal(fd, self._levels)
        self._levels = int(getattr(fd, "levels_completed", self._levels))
        self._sync_actions()
        return _frame_to_grid(fd), term

    def available_actions(self, grid) -> list[Action]:
        # RESET is excluded so BFS never plans a level restart; the controller
        # issues RESET explicitly via Action("RESET") after GAME_OVER.
        return [Action(a.name) for a in self.w.action_space if a.name != "RESET"]
