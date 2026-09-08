"""
test_rule_spec.py — prove the Layer-1 spec compiler with NO LLM.

We hand-write the push-game's mechanic AS A SPEC DICT (the thing Qwen will emit),
compile it, and backtest it against real transitions collected from PushEnv. If it
reaches exact_match_rate == 1.0, then (a) the DSL can express a real multi-object
click mechanic and (b) the compiler produces correct Rule callables. Only then is
it worth asking Qwen to PRODUCE such a spec.

Run: uv run python stage1/test_rule_spec.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage1"))
sys.path.insert(0, "stage1")
import numpy as np

from stage1_core import Action, Transition, backtest
from rule_spec import compile_spec

# reuse the validated push game
from hardened_smoke_test import PushEnv, _gen_clicks, BLOCK_A, BLOCK_B, TARG_A, TARG_B, A_TARG, B_TARG


def collect_history(n=12):
    """Drive PushEnv with a coverage policy, record real transitions."""
    env = PushEnv(); grid = env.reset()
    hist = []
    for _ in range(n):
        acts = _gen_clicks(grid)
        if not acts:
            break
        a = acts[_ % len(acts)]
        nxt, term = env.step(a)
        hist.append(Transition(grid, a, nxt, term))
        grid = nxt
        if term == "LEVEL_COMPLETED":
            break
    return hist


# The spec a correct inducer should emit for the push game.
# Blocks are pushed +1 col on click; goal is both blocks on their target columns.
PUSH_SPEC = {
    "background": 0,
    "rules": [
        {"kind": "translate_click", "color": BLOCK_A, "vector": [0, 1], "blocked_by": []},
        {"kind": "translate_click", "color": BLOCK_B, "vector": [0, 1], "blocked_by": []},
    ],
    # goal: both blocks reached their target columns. Express as: no TARG marker
    # visible (block sits on it) for BOTH targets. Simplest expressible: the two
    # target-marker colors are both absent (covered by blocks).
    "goal": {"kind": "color_absent", "color": TARG_A},  # refined below for both
}


def main():
    hist = collect_history()
    print(f"collected {len(hist)} real transitions from PushEnv")

    # --- test 1: the movement rules backtest exactly ---
    # Use a goal that can't interfere with backtest (backtest only checks step()).
    model = compile_spec(PUSH_SPEC)
    res = backtest(model, hist)
    print(f"[movement] exact_match_rate={res.exact_match_rate:.3f} "
          f"({res.correct}/{res.total}) first_divergence={res.first_divergence}")
    if res.first_divergence is not None:
        t = hist[res.first_divergence]
        print("  divergence at:", t.action)
        print("  predicted vs observed diff:",
              res.divergence_diff.summary() if res.divergence_diff else "(shape mismatch)")

    assert res.exact_match_rate == 1.0, (
        "COMPILER/SCHEMA GAP: translate_click spec did not reproduce the push mechanic. "
        "Fix the compiler or the schema before involving Qwen."
    )
    print("\nOK — hand-written spec compiles and backtests 1.0 on real transitions.")
    print("Layer 1 (schema + compiler) validated. Qwen's job is now to EMIT this dict.")


if __name__ == "__main__":
    main()
