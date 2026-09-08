"""
test_qwen_inducer.py — can Qwen EMIT a spec that compiles and backtests 1.0?

Points the real QwenInducer at the push game's real transitions (whose correct
spec we already know) and checks whether the model's emitted spec:
  (a) parses to JSON,
  (b) compiles via rule_spec,
  (c) backtests to exact_match_rate == 1.0 on the observed history.

This is the ft09 feasibility probe's successor: not "does Qwen reason about the
mechanic" (it does) but "does Qwen emit a COMPILABLE, CORRECT spec in our DSL."

Run: uv run python stage1/test_qwen_inducer.py --endpoint http://127.0.0.1:1234/v1
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage1"))
sys.path.insert(0, "stage1")

from stage1_core import backtest
from qwen_inducer import QwenInducer
from test_rule_spec import collect_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default="vrfai/Qwen3.6-27B-FP8")
    ap.add_argument("--key-file", default=".cache/arc3_runtime/server-api-key")
    args = ap.parse_args()

    key = ""
    try:
        key = open(args.key_file).read().strip()
    except Exception:
        pass

    hist = collect_history()
    print(f"collected {len(hist)} real push-game transitions")
    print("object-diffs the inducer will see:")
    from qwen_inducer import _render_history, _guess_background
    bg = _guess_background(hist)
    print(_render_history(hist, bg))
    print()

    ind = QwenInducer(base_url=args.endpoint, model=args.model, api_key=key, verbose=True)
    models = ind(hist, None, None)      # cold induction

    if not models:
        print("\nRESULT: inducer returned no model (parse/compile failed). "
              "Iterate the prompt or schema.")
        return

    res = backtest(models[0], hist)
    print(f"\nRESULT: emitted model backtest exact_match_rate={res.exact_match_rate:.3f} "
          f"({res.correct}/{res.total})")
    if res.exact_match_rate == 1.0:
        print("*** Qwen emitted a COMPILABLE, CORRECT spec. Layer 2 works end-to-end.")
    else:
        print(f"first_divergence={res.first_divergence}; "
              f"{res.divergence_diff.summary() if res.divergence_diff else 'shape mismatch'}")
        print("Qwen emitted a compilable but imperfect spec -> Layer 3 (repair loop) needed,")
        print("or the prompt needs work. This is expected on early attempts.")


if __name__ == "__main__":
    main()
