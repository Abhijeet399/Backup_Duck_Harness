"""
run_stage1.py — standalone Stage 1 runner.

Talks to arc_agi directly through env_adapter.ArcEnv. Does NOT go through
`make interactive` (that runs the OLD agent). Two modes:

  --handwritten : stub inducer returns a model you register in HANDWRITTEN below.
                  Needs NO vLLM server. Use this to validate the spine on a real
                  game (Step 4) before building the Qwen inducer.

  (default)     : real Qwen inducer against --endpoint. Needs `make server` up.
                  qwen_inducer.py is Step 6 and not built yet.

Examples:
  # spine validation on a game you understand, no GPU:
  python stage1/run_stage1.py --game ls20 --handwritten --wall 600 --max-actions 300 --max-calls 1

  # real run once the inducer exists (server must be up):
  python stage1/run_stage1.py --game ls20 --endpoint http://127.0.0.1:1234/v1
"""
from __future__ import annotations

import argparse
import time

from stage1_controller import Budget, play_game
from env_adapter import ArcEnv


# ---- Hand-written models for Step-4 validation -------------------------------
# Register game_id -> (WorldModel, probe_actions). Write the model exactly like
# correct_model() in example_smoke_test.py. Leave empty until you hand-write one.
def _handwritten_registry():
    reg = {}
    # from stage1_core import WorldModel, Action
    # reg["ls20"] = (build_ls20_model(), [Action("ACTION1"), Action("ACTION2"),
    #                                     Action("ACTION3"), Action("ACTION4")])
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--handwritten", action="store_true")
    ap.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--wall", type=float, default=270.0)     # 4.5 min eval envelope
    ap.add_argument("--max-calls", type=int, default=2)
    ap.add_argument("--max-actions", type=int, default=25)
    ap.add_argument("--reserve", type=float, default=30.0)
    args = ap.parse_args()

    env = ArcEnv(args.game, args.envs)

    if args.handwritten:
        reg = _handwritten_registry()
        if args.game not in reg:
            raise SystemExit(
                f"No hand-written model registered for '{args.game}'. Add one to "
                f"_handwritten_registry() in run_stage1.py (pattern: correct_model() "
                f"in example_smoke_test.py)."
            )
        model, probes = reg[args.game]
        inducer = lambda history, current, div: [model]
    else:
        from qwen_inducer import QwenInducer, default_probes  # Step 6
        inducer = QwenInducer(endpoint=args.endpoint)
        probes = default_probes(env)

    action_gen = lambda g: env.available_actions(g)
    budget = Budget(
        wall_clock_s=args.wall,
        max_llm_calls=args.max_calls,
        max_real_actions=args.max_actions,
        reserve_s=args.reserve,
    )

    t0 = time.monotonic()
    result = play_game(env, inducer, probes, action_gen, budget)
    print(
        f"game={args.game} status={result.status} "
        f"levels={result.levels_completed} "
        f"actions={result.budget['real_actions']} "
        f"calls={result.budget['llm_calls']} "
        f"wall={time.monotonic() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
