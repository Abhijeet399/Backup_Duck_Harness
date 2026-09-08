"""
stage1_core.py — game-general spine for a Rodionov-style executable world model
for ARC-AGI-3, adapted for a LOCAL ~27B coding model (offline, single GPU).

Design contract
---------------
* Nothing in this file calls the LLM. It is all deterministic and cheap.
* The LLM only ever edits a *rule library* (see WorldModel below): small, local,
  Schema-Networks-style transition rules. Small edits are what a 27B model can do
  reliably; whole-file synthesis is what it cannot.
* Verification does the heavy lifting. The model proposes; the backtester disposes.

Interaction seam
----------------
Implement `Env` against your arc_agi wrapper (~10 lines). Everything else is
game-general.

Integrity (important for the offline Kaggle eval)
-------------------------------------------------
The hidden games ship as local game.py files at eval time. Do NOT read, import,
introspect, or `ps`-scrape them. Interact ONLY through Env. Rodionov devotes a
whole section to closing exactly these leakage channels; a harness that reads the
game source is disqualified in spirit and probably in fact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

Grid = np.ndarray  # (H, W) small-int array, cell values 0..15


# ----------------------------------------------------------------------------- #
# Actions
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Action:
    name: str                 # "ACTION1".."ACTION7", "RESET"
    x: Optional[int] = None   # column, for ACTION6 (cell-select) games
    y: Optional[int] = None   # row
    def __repr__(self) -> str:
        return self.name if self.x is None else f"{self.name}@({self.x},{self.y})"


# A recorded environment transition. `terminal` is None | "LEVEL_COMPLETED" | "GAME_OVER".
@dataclass(frozen=True)
class Transition:
    grid: Grid
    action: Action
    next_grid: Grid
    terminal: Optional[str] = None


# ----------------------------------------------------------------------------- #
# Environment seam — implement this against arc_agi's wrapper
# ----------------------------------------------------------------------------- #
class Env(Protocol):
    def reset(self) -> Grid: ...
    def step(self, a: Action) -> tuple[Grid, Optional[str]]: ...
    # Candidate actions worth trying from `grid`. For arrow games this is the 4-5
    # simple actions; for click games DO NOT return all 4096 cells — return only
    # cells your current model thinks are meaningful (see action_generator below).
    def available_actions(self, grid: Grid) -> list[Action]: ...


# ----------------------------------------------------------------------------- #
# Canonical state key (for BFS visited-set and history dedup)
# ----------------------------------------------------------------------------- #
def grid_key(g: Grid) -> bytes:
    return g.shape[0].to_bytes(2, "little") + g.shape[1].to_bytes(2, "little") + g.tobytes()


# ----------------------------------------------------------------------------- #
# Object extraction (connected components) — the object-centric lens
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Obj:
    color: int
    cells: frozenset[tuple[int, int]]  # (row, col)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return min(rs), min(cs), max(rs), max(cs)

    @property
    def size(self) -> int:
        return len(self.cells)

    def normalized(self) -> frozenset[tuple[int, int]]:
        r0, c0, _, _ = self.bbox
        return frozenset((r - r0, c - c0) for r, c in self.cells)

    def offset(self, other: "Obj") -> Optional[tuple[int, int]]:
        """If `self` is a rigid translation of `other` (same color+shape), return (dr, dc)."""
        if self.color != other.color or self.normalized() != other.normalized():
            return None
        r0, c0, _, _ = self.bbox
        or0, oc0, _, _ = other.bbox
        return (r0 - or0, c0 - oc0)


def extract_objects(grid: Grid, background: int = 0, connectivity: int = 4) -> list[Obj]:
    """Flood-fill same-color 4- or 8-connected components. Background color is skipped."""
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    objs: list[Obj] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or grid[r, c] == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            cells = []
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            objs.append(Obj(color=color, cells=frozenset(cells)))
    return objs


# ----------------------------------------------------------------------------- #
# Object-level diff — the signal you feed back to the LLM to repair a rule
# ----------------------------------------------------------------------------- #
@dataclass
class ObjectDiff:
    moved: list[tuple[Obj, tuple[int, int]]]  # (object, (dr, dc))
    appeared: list[Obj]
    disappeared: list[Obj]
    changed_cells: int

    def summary(self) -> str:
        parts = []
        for o, (dr, dc) in self.moved:
            parts.append(f"color {o.color} size {o.size} moved by ({dr},{dc})")
        for o in self.appeared:
            parts.append(f"color {o.color} size {o.size} appeared at {o.bbox[:2]}")
        for o in self.disappeared:
            parts.append(f"color {o.color} size {o.size} disappeared from {o.bbox[:2]}")
        if not parts:
            return "no object-level change"
        return "; ".join(parts)


def object_diff(before: Grid, after: Grid, background: int = 0) -> ObjectDiff:
    old = extract_objects(before, background)
    new = extract_objects(after, background)
    matched_new = [False] * len(new)
    moved, disappeared = [], []
    for o in old:
        hit = None
        for j, n in enumerate(new):
            if matched_new[j]:
                continue
            off = n.offset(o)
            if off is not None:
                hit = (j, off)
                break
        if hit is None:
            disappeared.append(o)
        else:
            j, off = hit
            matched_new[j] = True
            if off != (0, 0):
                moved.append((new[j], off))
    appeared = [n for j, n in enumerate(new) if not matched_new[j]]
    changed = int(np.count_nonzero(before != after)) if before.shape == after.shape else -1
    return ObjectDiff(moved=moved, appeared=appeared, disappeared=disappeared, changed_cells=changed)


# ----------------------------------------------------------------------------- #
# World model — the ONLY thing the LLM edits
# ----------------------------------------------------------------------------- #
# A Rule is a small local transition schema. It returns a NEW grid if it fires,
# or None if it does not apply. Rules are applied in order; first firing wins,
# so keep them mutually specific. This decomposition is what makes induction
# tractable for a 27B model: it edits/adds/removes one Rule at a time.
Rule = Callable[[Grid, Action], Optional[Grid]]


class WorldModel:
    def __init__(
        self,
        rules: list[Rule],
        goal_fn: Callable[[Grid], bool],
        terminal_fn: Callable[[Grid], Optional[str]] = lambda g: None,
        background: int = 0,
        ignore_mask=None,
    ):
        self.rules = rules
        self.goal_fn = goal_fn
        self.terminal_fn = terminal_fn
        self.background = background
        # HUD/timer cells to EXCLUDE from exact-match backtest (bool array or None).
        # The world model predicts game state, not display; masked cells are decoration.
        self.ignore_mask = ignore_mask

    def step(self, grid: Grid, a: Action) -> Optional[Grid]:
        for rule in self.rules:
            out = rule(grid, a)
            if out is not None:
                return out
        return grid.copy()  # default: no-op (explicitly model "nothing happens")

    def is_goal(self, grid: Grid) -> bool:
        return self.goal_fn(grid)

    def is_terminal(self, grid: Grid) -> Optional[str]:
        return self.terminal_fn(grid)

    def complexity(self) -> int:
        """Crude MDL proxy: number of rules. Prefer fewer rules that still pass backtest."""
        return len(self.rules)


# ----------------------------------------------------------------------------- #
# Backtest — replay recorded history through the model, localize first divergence
# ----------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    total: int
    correct: int
    first_divergence: Optional[int]           # index into transitions, or None if all pass
    divergence_diff: Optional[ObjectDiff]     # predicted-vs-observed at first divergence

    @property
    def exact_match_rate(self) -> float:
        return self.correct / self.total if self.total else 1.0


def backtest(model: WorldModel, history: list[Transition]) -> BacktestResult:
    correct = 0
    first_div = None
    div_diff = None
    for i, t in enumerate(history):
        pred = model.step(t.grid, t.action)
        if pred is None or pred.shape != t.next_grid.shape:
            ok = False
        elif getattr(model, "ignore_mask", None) is not None and model.ignore_mask.shape == t.next_grid.shape:
            keep = ~model.ignore_mask
            ok = bool(np.array_equal(pred[keep], t.next_grid[keep]))
        else:
            ok = bool(np.array_equal(pred, t.next_grid))
        if ok:
            correct += 1
        elif first_div is None:
            first_div = i
            # what the model got wrong: diff between what it predicted and reality
            if pred is not None and pred.shape == t.next_grid.shape:
                div_diff = object_diff(pred, t.next_grid, model.background)
    return BacktestResult(len(history), correct, first_div, div_diff)


# ----------------------------------------------------------------------------- #
# Planner — BFS through the executable model to the inferred goal
# ----------------------------------------------------------------------------- #
def bfs_plan(
    model: WorldModel,
    start: Grid,
    action_generator: Callable[[Grid], list[Action]],
    max_nodes: int = 200_000,
    max_depth: int = 80,
    max_seconds: float = 8.0,
) -> Optional[list[Action]]:
    """Return a shortest action sequence reaching goal inside the model, or None.

    action_generator is where you make click-games tractable: return only the
    cells/actions your model believes are meaningful from this state, not all 4096.
    Shortest-path here directly serves RHAE, which rewards being faster than the
    human median (capped at 1.15x), quadratically.
    """
    if model.is_goal(start):
        return []
    import time as _time
    _deadline = _time.monotonic() + max_seconds
    seen = {grid_key(start)}
    frontier: deque[tuple[Grid, list[Action]]] = deque([(start, [])])
    nodes = 0
    while frontier and nodes < max_nodes:
        if _time.monotonic() > _deadline:
            return None
        grid, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        for a in action_generator(grid):
            nxt = model.step(grid, a)
            if nxt is None:
                continue
            if model.is_terminal(nxt) == "GAME_OVER":
                continue
            k = grid_key(nxt)
            if k in seen:
                continue
            seen.add(k)
            nodes += 1
            new_path = path + [a]
            if model.is_goal(nxt):
                return new_path
            frontier.append((nxt, new_path))
    return None


# ----------------------------------------------------------------------------- #
# Commit-and-verify executor — the online falsification test
# ----------------------------------------------------------------------------- #
@dataclass
class ExecOutcome:
    status: str                    # "goal" | "diverged" | "terminal" | "exhausted"
    steps_taken: int
    terminal: Optional[str]
    divergence_index: Optional[int]
    divergence_diff: Optional[ObjectDiff]
    new_transitions: list[Transition]  # everything observed, append to history


def execute_and_verify(
    env: Env,
    model: WorldModel,
    start: Grid,
    plan: list[Action],
) -> ExecOutcome:
    """Execute `plan` in the real env, checking each observed frame against the
    model's prediction. Stop at the FIRST mismatch and return the real transition
    so the LLM can repair the model. A plan that runs to completion is not replay —
    it is a passed online test of the world model over `len(plan)` fresh steps.
    """
    grid = start
    recorded: list[Transition] = []
    for i, a in enumerate(plan):
        predicted = model.step(grid, a)
        observed, terminal = env.step(a)
        recorded.append(Transition(grid, a, observed, terminal))
        mismatch = (
            predicted is None
            or predicted.shape != observed.shape
            or not np.array_equal(predicted, observed)
        )
        if mismatch:
            diff = None
            if predicted is not None and predicted.shape == observed.shape:
                diff = object_diff(predicted, observed, model.background)
            return ExecOutcome("diverged", i + 1, terminal, i, diff, recorded)
        grid = observed
        if terminal == "LEVEL_COMPLETED" or model.is_goal(grid):
            return ExecOutcome("goal", i + 1, terminal, None, None, recorded)
        if terminal == "GAME_OVER":
            return ExecOutcome("terminal", i + 1, terminal, None, None, recorded)
    return ExecOutcome("exhausted", len(plan), None, None, None, recorded)


# ----------------------------------------------------------------------------- #
# Hypothesis pool — directly targets Rodionov's main failure mode
# ----------------------------------------------------------------------------- #
# Rodionov's #1 reported failure: premature commitment to one wrong model. Keep
# several candidates alive; score by (fit, then simplicity); when exploring, prefer
# actions that DISAGREE across surviving candidates (maximally falsifying).
@dataclass
class Candidate:
    model: WorldModel
    result: BacktestResult

def rank_candidates(cands: list[Candidate]) -> list[Candidate]:
    # highest exact-match first, then fewest rules (MDL proxy), then fewest divergences
    return sorted(
        cands,
        key=lambda c: (-c.result.exact_match_rate, c.model.complexity()),
    )

def falsifying_action(
    cands: list[Candidate],
    grid: Grid,
    action_generator: Callable[[Grid], list[Action]],
) -> Optional[Action]:
    """Pick the real-env action whose predicted outcome MOST splits the surviving
    candidates. Spending one scored action here buys the most model information."""
    best_a, best_spread = None, -1
    for a in action_generator(grid):
        preds = set()
        for c in cands:
            p = c.model.step(grid, a)
            preds.add(grid_key(p) if p is not None else b"None")
        if len(preds) > best_spread:
            best_spread, best_a = len(preds), a
    return best_a if best_spread > 1 else None
