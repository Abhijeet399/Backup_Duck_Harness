"""
rule_spec.py — Layer 1 of the Qwen inducer: a NARROW rule-spec DSL + a
deterministic compiler from a spec dict to a stage1_core.WorldModel.

Design bet (from the handoff §1): a 27B model cannot reliably write executable
Rule callables, but it CAN emit a small structured spec describing the mechanic.
So the LLM emits a spec (JSON); THIS module compiles it to Rule callables. The
compiler is pure Python, deterministic, and unit-testable with no server.

Scope is deliberately small — only the mechanics we've actually observed in the
25 games. Add a primitive only when a real game's backtest fails for lack of one.

Spec format (a dict, what Qwen emits as JSON):
{
  "background": 0,
  "rules": [ <rule-spec>, ... ],     # applied in order, first firing wins
  "goal": <goal-spec>,
  "terminal": <terminal-spec>        # optional
}

Rule-spec kinds (v1):
  {"kind":"translate", "color":C, "action":"ACTION1", "vector":[dr,dc],
   "blocked_by":[colors...]}        # move the single C-colored object by vector,
                                     # canceled if any destination cell holds a
                                     # blocked_by color or is off-grid.
  {"kind":"translate_click", "color":C, "vector":[dr,dc], "blocked_by":[...]}
                                     # ACTION6 click ON a C-object pushes THAT
                                     # object by vector (the clicked object only).
  {"kind":"convert_cell", "action":"ANY|ACTIONx", "from":A, "to":B, "count":N}
                                     # each firing, recolor N cells of color A->B
                                     # (the counter mechanic).  count default 1.
Goal-spec kinds (v1):
  {"kind":"color_absent", "color":C}          # win when no cell of color C remains
  {"kind":"color_count", "color":C, "eq":N}   # win when count(color C)==N
  {"kind":"object_at", "color":C, "row":R, "col":Col}  # C-object's bbox top-left ==(R,Col)
Terminal-spec kinds (v1):
  {"kind":"color_count", "color":C, "eq":0, "result":"GAME_OVER"}  # e.g. budget out
"""
from __future__ import annotations
import numpy as np
from stage1_core import Action, WorldModel, extract_objects


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _single_obj(grid, color, background):
    objs = [o for o in extract_objects(grid, background) if o.color == color]
    return objs[0] if len(objs) == 1 else None


def _obj_at_cell(grid, r, c, background):
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return None
    color = int(grid[r, c])
    if color == background:
        return None
    for o in extract_objects(grid, background):
        if (r, c) in o.cells:
            return o
    return None


def _translate_cells(grid, obj, dr, dc, blocked_by, background):
    """Return new grid with obj shifted by (dr,dc), or None if blocked/off-grid."""
    H, W = grid.shape
    blocked = set(blocked_by or [])
    dest = [(r + dr, c + dc) for (r, c) in obj.cells]
    src = set(obj.cells)
    for (r, c) in dest:
        if not (0 <= r < H and 0 <= c < W):
            return None
        if (r, c) in src:
            continue                       # moving into own vacated cell is fine
        if int(grid[r, c]) in blocked:
            return None
    out = grid.copy()
    for (r, c) in obj.cells:
        out[r, c] = background             # vacate (may be overwritten below)
    for (r, c) in dest:
        out[r, c] = obj.color
    return out


# ----------------------------------------------------------------------------- #
# rule compilers — each returns a Rule: Callable[[Grid, Action], Optional[Grid]]
# ----------------------------------------------------------------------------- #
def _compile_translate(spec, background):
    color = int(spec["color"]); act = spec["action"]
    dr, dc = spec["vector"]; blocked = spec.get("blocked_by", [])

    def rule(grid, a):
        if a.name != act:
            return None
        obj = _single_obj(grid, color, background)
        if obj is None:
            return grid.copy()
        moved = _translate_cells(grid, obj, dr, dc, blocked, background)
        return moved if moved is not None else grid.copy()  # blocked = no-op
    return rule


def _compile_translate_click(spec, background):
    color = int(spec["color"]); dr, dc = spec["vector"]
    blocked = spec.get("blocked_by", [])

    def rule(grid, a):
        if a.name != "ACTION6" or a.x is None:
            return None
        obj = _obj_at_cell(grid, a.y, a.x, background)
        if obj is None or obj.color != color:
            return None                    # not my object -> let another rule try
        moved = _translate_cells(grid, obj, dr, dc, blocked, background)
        return moved if moved is not None else grid.copy()
    return rule


def _compile_convert_cell(spec, background):
    act = spec.get("action", "ANY")
    frm = int(spec["from"]); to = int(spec["to"]); n = int(spec.get("count", 1))

    def rule(grid, a):
        if act != "ANY" and a.name != act:
            return None
        out = grid.copy()
        ys, xs = np.where(out == frm)
        for k in range(min(n, len(ys))):
            out[ys[k], xs[k]] = to
        return out
    return rule


_RULE_COMPILERS = {
    "translate": _compile_translate,
    "translate_click": _compile_translate_click,
    "convert_cell": _compile_convert_cell,
}


# ----------------------------------------------------------------------------- #
# goal / terminal compilers
# ----------------------------------------------------------------------------- #
def _compile_goal(spec, background):
    kind = spec["kind"]
    if kind == "color_absent":
        c = int(spec["color"]);  return lambda g: not (g == c).any()
    if kind == "color_count":
        c = int(spec["color"]); n = int(spec["eq"]); return lambda g: int((g == c).sum()) == n
    if kind == "object_at":
        c = int(spec["color"]); R = int(spec["row"]); C = int(spec["col"])
        def goal(g):
            o = _single_obj(g, c, background)
            return o is not None and o.bbox[0] == R and o.bbox[1] == C
        return goal
    raise ValueError(f"unknown goal kind: {kind}")


def _compile_terminal(spec, background):
    if spec is None:
        return lambda g: None
    kind = spec["kind"]
    if kind == "color_count":
        c = int(spec["color"]); n = int(spec["eq"]); res = spec.get("result", "GAME_OVER")
        return lambda g: res if int((g == c).sum()) == n else None
    raise ValueError(f"unknown terminal kind: {kind}")


# ----------------------------------------------------------------------------- #
# top-level compile
# ----------------------------------------------------------------------------- #
class SpecError(Exception):
    pass


def compile_spec(spec: dict) -> WorldModel:
    """Compile a spec dict into a WorldModel. Raises SpecError on malformed spec."""
    try:
        background = int(spec.get("background", 0))
        rules = []
        for rs in spec["rules"]:
            comp = _RULE_COMPILERS.get(rs["kind"])
            if comp is None:
                raise SpecError(f"unknown rule kind: {rs['kind']}")
            rules.append(comp(rs, background))
        goal_fn = _compile_goal(spec["goal"], background)
        terminal_fn = _compile_terminal(spec.get("terminal"), background)
        return WorldModel(rules=rules, goal_fn=goal_fn,
                          terminal_fn=terminal_fn, background=background)
    except SpecError:
        raise
    except Exception as ex:
        raise SpecError(f"compile failed: {type(ex).__name__}: {ex}") from ex
