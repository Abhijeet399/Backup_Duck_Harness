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
    if not text:
        return ""
    # prefer a fenced python block; else take from the first def goal_reached
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1)
    else:
        idx = text.find("def goal_reached")
        code = text[idx:] if idx != -1 else text
    # trim anything before the first import/def (stray prose/backticks)
    lines = code.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(("import ", "def goal_reached", "from ")):
            start = i; break
    code = "\n".join(lines[start:]).strip()
    return code


def _parses(code):
    import ast
    try:
        ast.parse(code); return True
    except Exception:
        return False


def propose_goal_predicate(win_frames, nonwin_frames, inducer, cap=8):
    prompt = _PROMPT.format(
        win_block=_render(win_frames, "WIN", cap),
        nonwin_block=_render(nonwin_frames, "NONWIN", cap),
    )
    # reuse the inducer's raw LLM call (thinking-off, fast — this is spec-like emission)
    for _ in range(3):
        reply = inducer._call(prompt)  # noqa: SLF001
        code = _extract_code(reply)
        if code and _parses(code) and "goal_reached" in code:
            return code
        # nudge for valid syntax on retry
        prompt = prompt + ("\n\nIMPORTANT: your previous answer was not valid Python. "
                           "Return ONLY a syntactically correct ```python code block "
                           "defining def goal_reached(grid): using numpy as np. No prose.")
    return code  # last attempt (may still be bad; validate_predicate will reject)


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




_REFINE = """Your goal_reached predicate was TOO PERMISSIVE. It correctly returned
True on all WIN frames, but it ALSO returned True on these NON-WIN frames (where the
level was NOT won). Make the predicate STRICTER so it is False on these, while
staying True on all wins.

Your previous predicate:
```python
{prev_src}
```

NON-WIN frames it wrongly accepted (make these return False):
{fp_block}

WIN frames (must still be True):
{win_block}

Return ONLY a ```python code block with the corrected `goal_reached(grid)`."""


def _refine_goal_predicate(prev_src, win_frames, fp_frames, inducer, cap=6):
    prompt = _REFINE.format(
        prev_src=prev_src.strip(),
        fp_block=_render(fp_frames, "WRONGLY_ACCEPTED", cap),
        win_block=_render(win_frames, "WIN", cap),
    )
    return _extract_code(inducer._call(prompt))


def discover_goal(game, submit_action, inducer, max_tries=3, cap=8, verbose=True):
    """Full loop: mine -> propose -> validate, retrying the LLM up to max_tries."""
    win, nonwin = mine_frames(game, submit_action)
    if verbose:
        print(f"[goal] {game}: mined {len(win)} win, {len(nonwin)} non-win-submit frames")
    if not win:
        return None, {"error": "no wins mined"}
    last_stats = None
    src = propose_goal_predicate(win, nonwin, inducer, cap=cap)
    for t in range(max_tries):
        ok, fn, stats = validate_predicate(src, win, nonwin)
        last_stats = stats
        if verbose:
            print(f"[goal] try {t+1}: {stats} -> {'ACCEPT' if ok else 'reject'}")
        if ok:
            return {"goal_reached": fn, "src": src, "submit_action": submit_action,
                    "stats": stats}, stats
        # If the predicate is right on wins but too permissive, REFINE using the
        # false-positive frames (non-wins it wrongly accepted). Else re-propose fresh.
        if stats.get("win_true") == stats.get("win_total") and stats.get("eval_errors", 1) == 0:
            # collect the non-win frames the current predicate wrongly accepts
            ns = {"np": np}
            try:
                exec(src, ns); cur = ns["goal_reached"]
                fps = [f for f in nonwin if _safe_true(cur, f)][:6]
            except Exception:
                fps = []
            if fps:
                src = _refine_goal_predicate(src, win, fps, inducer, cap=cap)
                continue
        # fallback: fresh proposal
        src = propose_goal_predicate(win, nonwin, inducer, cap=cap)
    return None, last_stats


def _safe_true(fn, g):
    try: return bool(fn(g)) is True
    except Exception: return False
