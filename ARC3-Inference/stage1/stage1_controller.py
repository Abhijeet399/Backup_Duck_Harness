"""
stage1_controller.py — per-game orchestration + budget governor for the Stage 1
executable-world-model harness. Sits on top of stage1_core.py.

Operating regime this is designed for (single RTX Pro 6000, 9h, ~120 private games):
  * ~4-5 min wall-clock per game.
  * ~8-25 real (scored) actions per game.
  * ~2 LLM induction calls per game is the *realistic* ceiling (see note below).
So: spend a handful of real actions on COVERAGE, induce once, then BFS is free.

The LLM lives entirely behind the `Inducer` seam. Nothing here calls a model, so
the whole control policy is deterministic and unit-testable with a stub inducer.

Key discipline: only ever commit a BFS plan to the real environment from a world
model that reproduces ALL observed history exactly (exact_match_rate == 1.0). A
model that mispredicts anything you've already seen has no business planning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from stage1_core import (
    Action, Transition, Env, WorldModel, Candidate, ObjectDiff,
    backtest, bfs_plan, execute_and_verify, rank_candidates, falsifying_action,
)


# --------------------------------------------------------------------------- #
# The LLM seam. Given history (and optionally the current best model + the diff
# that just falsified it), return candidate world models. This is the ONLY place
# the 27B model is invoked. Keep its output compact: it should reason over the
# object-diffs already computed for it, not raw 64x64 grids.
# --------------------------------------------------------------------------- #
class Inducer(Protocol):
    def __call__(
        self,
        history: list[Transition],
        current: Optional[WorldModel],
        divergence: Optional[ObjectDiff],
    ) -> list[WorldModel]: ...


# --------------------------------------------------------------------------- #
# Budget governor
# --------------------------------------------------------------------------- #
@dataclass
class Budget:
    wall_clock_s: float
    max_llm_calls: int
    max_real_actions: int
    reserve_s: float = 20.0          # always keep this much time to execute a final plan
    _start: float = field(default_factory=time.monotonic)
    _calls: int = 0
    _actions: int = 0

    def time_left(self) -> float:
        return self.wall_clock_s - (time.monotonic() - self._start)

    def can_call_llm(self) -> bool:
        return self._calls < self.max_llm_calls and self.time_left() > self.reserve_s

    def can_act(self) -> bool:
        return self._actions < self.max_real_actions and self.time_left() > 0

    def llm_called(self) -> None:
        self._calls += 1

    def act(self, n: int = 1) -> None:
        self._actions += n

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._calls, "real_actions": self._actions,
                "time_left_s": round(self.time_left(), 1)}


@dataclass
class GameResult:
    status: str                  # "solved" | "progress" | "no_model" | "timeout"
    levels_completed: int
    history: list[Transition]
    budget: dict


# --------------------------------------------------------------------------- #
# Coverage-first cold start
# --------------------------------------------------------------------------- #
def round_robin_explore(
    env: Env,
    grid,
    budget: Budget,
    probe_actions: list[Action],
    min_per_action: int = 1,
) -> tuple[list[Transition], "object", str]:
    """Observe each distinct probe action >= min_per_action times along the (single,
    irreversible) trajectory. This is the fix for the 'committed to one action and
    never learned the rest' failure: you cannot fan out from a state, so guarantee
    coverage by cycling actions before inducing.

    `probe_actions` is caller-supplied. For pure discrete games it's the simple
    actions. For coordinate (ACTION6) games, pass a few concrete ACTION6@(x,y)
    bound to salient cells (object centroids, corners) so the coordinate dimension
    is actually exercised.
    """
    history: list[Transition] = []
    counts = {repr(a): 0 for a in probe_actions}
    i = 0
    while probe_actions and min(counts.values()) < min_per_action and budget.can_act():
        a = probe_actions[i % len(probe_actions)]
        i += 1
        nxt, term = env.step(a)
        budget.act()
        history.append(Transition(grid, a, nxt, term))
        counts[repr(a)] += 1
        if term == "LEVEL_COMPLETED":
            return history, nxt, "solved"
        if term == "GAME_OVER":
            if not budget.can_act():
                return history, nxt, "explored"
            r, rterm = env.step(Action("RESET"))
            budget.act()
            history.append(Transition(nxt, Action("RESET"), r, rterm))
            grid = r
        else:
            grid = nxt
    return history, grid, "explored"


# --------------------------------------------------------------------------- #
# Main per-game loop
# --------------------------------------------------------------------------- #
def play_game(
    env: Env,
    inducer: Inducer,
    probe_actions: list[Action],
    action_generator: Callable[["object"], list[Action]],
    budget: Budget,
    bfs_max_nodes: int = 200_000,
    bfs_max_depth: int = 80,
) -> GameResult:
    # First frame is free at game start; adjust if YOUR wrapper charges for it.
    grid = env.reset()
    history: list[Transition] = []
    levels = 0

    # Phase 1 — coverage-first cold start.
    h0, grid, status = round_robin_explore(env, grid, budget, probe_actions)
    history += h0
    if status == "solved":
        levels += 1

    # Phase 2 — first induction (one LLM call over the whole gathered trajectory).
    candidates: list[Candidate] = []
    if budget.can_call_llm():
        models = inducer(history, None, None)
        budget.llm_called()
        candidates = [Candidate(m, backtest(m, history)) for m in models]

    # Phase 3 — plan / verify / repair, under the governor.
    while budget.can_act() and budget.time_left() > budget.reserve_s:
        candidates = rank_candidates(candidates) if candidates else []
        trusted = next((c for c in candidates if c.result.exact_match_rate == 1.0), None)

        if trusted is None:
            # No model explains everything seen. Don't plan on a broken model.
            # Spend ONE real action to gather the most useful next transition:
            # the action that most splits surviving candidates, else keep covering.
            a = (falsifying_action(candidates, grid, action_generator)
                 if candidates else None) or _next_uncovered(history, probe_actions)
            grid = _step_record(env, a, grid, history, budget)
            if _just_solved(history):
                levels += 1
            if budget.can_call_llm():
                models = inducer(history, candidates[0].model if candidates else None, None)
                budget.llm_called()
                candidates = [Candidate(m, backtest(m, history)) for m in models]
            continue

        # We have a model that reproduces all history. Plan through it for free.
        plan = bfs_plan(trusted.model, grid, action_generator,
                        max_nodes=bfs_max_nodes, max_depth=bfs_max_depth)
        if plan is None:
            # Model is faithful but can't reach the goal from here: goal_fn is likely
            # wrong, or a needed mechanic is unobserved. Gather one more transition,
            # then re-induce (which may revise goal_fn).
            a = _next_uncovered(history, probe_actions)
            grid = _step_record(env, a, grid, history, budget)
            if _just_solved(history):
                levels += 1
                continue
            if budget.can_call_llm():
                models = inducer(history, trusted.model, None)
                budget.llm_called()
                candidates = [Candidate(m, backtest(m, history)) for m in models]
            continue

        # Commit-and-verify: execute the plan, checking each real frame vs prediction.
        outcome = execute_and_verify(env, trusted.model, grid, plan)
        budget.act(outcome.steps_taken)
        history += outcome.new_transitions
        if outcome.new_transitions:
            grid = outcome.new_transitions[-1].next_grid

        if outcome.status == "goal":
            levels += 1
            # Next level starts from current frame; re-plan with the same model,
            # repairing only if it diverges. Loop continues.
            continue
        if outcome.status == "diverged":
            # The model was wrong at exactly one step — the gold repair signal.
            if budget.can_call_llm():
                models = inducer(history, trusted.model, outcome.divergence_diff)
                budget.llm_called()
                candidates = [Candidate(m, backtest(m, history)) for m in models]
            # else: fall through; next iteration will try a falsifying action.
            continue
        if outcome.status == "terminal":  # GAME_OVER mid-plan
            if budget.can_act():
                grid = _step_record(env, Action("RESET"), grid, history, budget)
            continue
        # "exhausted": plan ran out without reaching goal — re-plan next loop.

    final_status = "solved" if levels and not budget.can_act() else (
        "progress" if levels else ("no_model" if not candidates else "timeout"))
    return GameResult(final_status, levels, history, budget.stats)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _step_record(env: Env, a: Action, grid, history: list[Transition], budget: Budget):
    nxt, term = env.step(a)
    budget.act()
    history.append(Transition(grid, a, nxt, term))
    if term == "GAME_OVER" and budget.can_act():
        r, rterm = env.step(Action("RESET"))
        budget.act()
        history.append(Transition(nxt, Action("RESET"), r, rterm))
        return r
    return nxt


def _just_solved(history: list[Transition]) -> bool:
    return bool(history) and history[-1].terminal == "LEVEL_COMPLETED"


def _next_uncovered(history: list[Transition], probe_actions: list[Action]) -> Action:
    """Least-observed probe action so far — keeps widening coverage when stuck."""
    seen = {}
    for t in history:
        seen[repr(t.action)] = seen.get(repr(t.action), 0) + 1
    return min(probe_actions, key=lambda a: seen.get(repr(a), 0))
