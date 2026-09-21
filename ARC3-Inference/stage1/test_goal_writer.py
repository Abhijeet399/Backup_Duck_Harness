"""
test_goal_writer.py — does the LLM write a VALID goal predicate for sb26 & sp80?

Validation is deterministic: the predicate must be True on every mined win frame
and False on every mined non-winning-submit frame. If it validates, we have a
data-certified goal_reached() for that game — no live actions spent.

Run: uv run python stage1/test_goal_writer.py --endpoint http://127.0.0.1:1234/v1
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage1"))
sys.path.insert(0, "stage1")

from qwen_inducer import QwenInducer
from goal_writer import discover_goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    args = ap.parse_args()

    ind = QwenInducer(endpoint=args.endpoint, verbose=False,
                      key_file="/host-duck/ARC3-Inference/.cache/arc3_runtime/server-api-key",
                      enable_thinking=True, max_tokens=8192)

    # (game, submit_action) — submit actions found by win-mining earlier
    targets = [
        ("sp80", "ACTION5"),   # known simple (col 24); should validate easily
        ("sb26", "ACTION5"),   # relational; the real test
        ("vc33", "ACTION6"),   # another submit game
        ("su15", "ACTION6"),
    ]
    for game, submit in targets:
        print(f"\n===== {game} (submit={submit}) =====")
        result, stats = discover_goal(game, submit, ind, max_tries=3, verbose=True)
        if result:
            print(f"  *** VALID GOAL for {game}:")
            print("  " + result["src"].strip().replace("\n", "\n  "))
        else:
            print(f"  no valid predicate found (best stats: {stats})")


if __name__ == "__main__":
    main()
