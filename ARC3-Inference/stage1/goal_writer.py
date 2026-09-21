"""
goal_writer.py — LLM-written, data-validated goal predicates (Twin method).

The bbox feature-finder cracked sp80 (one object -> one column) but not sb26
(relational / multi-object). The general fix, per Twin: have the LLM WRITE the
`goal_reached(grid)` predicate from mined win vs non-win-submit frames, then
VALIDATE it deterministically before trusting it — it must be True on every mined
win frame and False on every mined non-winning-submit frame. A wrong predicate is
rejected before any real action is spent.

Pipeline:
  1. mine_frames(game, submit_action) -> (win_frames, nonwin_submit_frames)
  2. propose_goal_predicate(win_frames, nonwin_frames, inducer) -> python source for goal_reached
  3. validate_predicate(src, win_frames, nonwin_frames) -> (ok, compiled_fn, stats)
  Only an OK predicate is returned for use by the solver.
"""
from __future__ import annotations
import sys, os, json, glob, re
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Mine win / non-win-submit frames from recorded event logs
# --------------------------------------------------------------------------- #
def mine_frames(game, submit_action, search_globs=None):
    """Return (win_frames, nonwin_submit_frames) as lists of 2D int numpy arrays.

    win_frames: the frame BEFORE a submit_action that produced a level completion.
    nonwin_submit_frames: the frame BEFORE a submit_action that did NOT win.
    (These are the certified positives and negatives for the goal predicate.)
    """
    if search_globs is None:
        search_globs = [
            f"/host-duck/example-run/artifacts/{game}*events.jsonl",
            f"/shared/arc_3_results/root/*/artifacts/{game}*events.jsonl",
        ]
    paths = []
    for g in search_globs:
        paths += glob.glob(g)
    win, nonwin = [], []
    for p in paths:
        prev = None
        try:
            events = [json.loads(l) for l in open(p)]
        except Exception:
            continue
        for e in events:
            b = e.get("board")
            if b is None:
                continue
            b = np.array(b)
            if b.ndim != 2:
                prev = None
                continue
            act = e.get("action_name") or e.get("action_display")
            won = bool(e.get("level_completed") or e.get("just_won_level"))
            if act == submit_action and prev is not None:
                (win if won else nonwin).append(prev)
            prev = b
    return win, nonwin


# --------------------------------------------------------------------------- #
# 2. Render frames compactly for the LLM (object summary, not raw 64x64)
# --------------------------------------------------------------------------- #
def _objects(grid):
    """Per color: count + bbox (top,left,bottom,right). Compact, LLM-friendly."""
    out = {}
    for c in np.unique(grid):
        ys, xs = np.where(grid == c)
        out[int(c)] = {
            "count": int(len(ys)),
            "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
        }
    return out


def _render(frames, label, cap=8):
    lines = [f"# {label} ({len(frames)} frames, showing up to {cap})"]
    for i, f in enumerate(frames[:cap]):
        lines.append(f"{label}[{i}]: " + json.dumps(_objects(f)))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Ask the LLM to write goal_reached(grid)
# --------------------------------------------------------------------------- #
_PROMPT = """You are reverse-engineering the WIN CONDITION of a grid puzzle.

Each frame is a 2D numpy array of small ints (colors). Below are frames summarized
as per-color {{count, bbox=[top,left,bottom,right]}}.

WIN frames: the board state at the moment the level was won (after pressing the
submit action). NON-WIN frames: states where the submit action was pressed but the
level was NOT won. The win condition is TRUE on every WIN frame and FALSE on every
NON-WIN frame.

{win_block}

{nonwin_block}

Write a Python function exactly named `goal_reached(grid)` that takes a 2D numpy
array `grid` and returns True iff the win condition holds. Use numpy (imported as
np). Infer the RELATIONAL/positional condition that separates WIN from NON-WIN
(e.g. one object's left edge at a column, two objects aligned, an object reaching a
marker). Keep it simple and general. Return ONLY a ```python code block with the
function, no prose."""


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code


def propose_goal_predicate(win_frames, nonwin_frames, inducer, cap=8):
    prompt = _PROMPT.format(
        win_block=_render(win_frames, "WIN", cap),
        nonwin_block=_render(nonwin_frames, "NONWIN", cap),
    )
    # reuse the inducer's raw LLM call (thinking-off, fast — this is spec-like emission)
    reply = inducer._call(prompt)  # noqa: SLF001 (intentional reuse of the server call)
    return _extract_code(reply)


# --------------------------------------------------------------------------- #
# 4. Validate the predicate against mined frames (deterministic filter)
# --------------------------------------------------------------------------- #
def validate_predicate(src, win_frames, nonwin_frames):
    ns = {"np": np}
    try:
        exec(src, ns)
        fn = ns.get("goal_reached")
        if fn is None:
            return False, None, {"error": "no goal_reached defined"}
    except Exception as ex:
        return False, None, {"error": f"compile failed: {ex}"}

    def safe(f, g):
        try:
            return bool(f(g))
        except Exception:
            return None

    win_true = sum(1 for f in win_frames if safe(fn, f) is True)
    nonwin_false = sum(1 for f in nonwin_frames if safe(fn, f) is False)
    errors = sum(1 for f in (win_frames + nonwin_frames) if safe(fn, f) is None)

    stats = {
        "win_true": win_true, "win_total": len(win_frames),
        "nonwin_false": nonwin_false, "nonwin_total": len(nonwin_frames),
        "eval_errors": errors,
    }
    # SOUND acceptance: must be True on ALL wins and False on ALL non-wins.
    ok = (win_true == len(win_frames)
          and nonwin_false == len(nonwin_frames)
          and errors == 0
          and len(win_frames) > 0)
    return ok, (fn if ok else None), stats


def discover_goal(game, submit_action, inducer, max_tries=3, cap=8, verbose=True):
    """Full loop: mine -> propose -> validate, retrying the LLM up to max_tries."""
    win, nonwin = mine_frames(game, submit_action)
    if verbose:
        print(f"[goal] {game}: mined {len(win)} win, {len(nonwin)} non-win-submit frames")
    if not win:
        return None, {"error": "no wins mined"}
    last_stats = None
    for t in range(max_tries):
        src = propose_goal_predicate(win, nonwin, inducer, cap=cap)
        ok, fn, stats = validate_predicate(src, win, nonwin)
        last_stats = stats
        if verbose:
            print(f"[goal] try {t+1}: {stats} -> {'ACCEPT' if ok else 'reject'}")
        if ok:
            return {"goal_reached": fn, "src": src, "submit_action": submit_action,
                    "stats": stats}, stats
    return None, last_stats
