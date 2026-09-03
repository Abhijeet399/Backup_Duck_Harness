"""
triage_games.py — one-line simplicity fingerprint for every game, to pick the
easiest spine-validation target (single clean avatar + a goal you recognize).

Per game: action count, #colors, #objects, per-action count of small rigidly
moving objects (1 = clean single avatar = easiest to model; 2 = compound sprite
like ls20; 0 = action does nothing here; many = complex), and whether a level
completed during the short probe. Then a shortlist ranked by hand-model ease.

Run:  uv run python stage1/triage_games.py --envs /test/ARC3/env_files
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from env_adapter import ArcEnv
from stage1_core import Action, object_diff, extract_objects

ALL_GAMES = ['ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59',
             'lf52', 'lp85', 'ls20', 'm0r0', 're86', 's5i5', 'sb26', 'sc25',
             'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30', 'r11l']


def fingerprint(game: str, envs: str) -> dict:
    try:
        env = ArcEnv(game, envs)
    except Exception as e:
        return {"game": game, "error": type(e).__name__ + ": " + str(e)[:50]}
    g = env.reset()
    acts = [a.name for a in env.available_actions(g)]
    n_obj, n_col = len(extract_objects(g)), int(np.unique(g).size)
    movers, win = [], False
    cur = g
    for a in acts:
        before = cur
        cur, term = env.step(Action(a))
        d = object_diff(before, cur)
        movers.append(sum(1 for o, _ in d.moved if o.size <= 60))
        if term == "LEVEL_COMPLETED":
            win = True
    typical = Counter(movers).most_common(1)[0][0] if movers else 0
    verdict = ({0: "static?", 1: "CLEAN", 2: "compound"}.get(typical, "complex"))
    return {"game": game, "actions": len(acts), "colors": n_col, "objects": n_obj,
            "movers": movers, "win": win, "shape": tuple(g.shape),
            "typical": typical, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="/test/ARC3/env_files")
    ap.add_argument("--games", nargs="*", default=ALL_GAMES)
    args = ap.parse_args()

    rows = []
    print(f"{'game':6} {'act':>3} {'col':>3} {'obj':>4}  {'verdict':9} {'win':>3}  movers")
    print("-" * 60)
    for game in args.games:
        fp = fingerprint(game, args.envs)
        if "error" in fp:
            print(f"{game:6}  -- {fp['error']}")
            continue
        rows.append(fp)
        print(f"{fp['game']:6} {fp['actions']:>3} {fp['colors']:>3} {fp['objects']:>4}  "
              f"{fp['verdict']:9} {'yes' if fp['win'] else '  .':>3}  {fp['movers']}")

    # shortlist: CLEAN first, then fewest objects+colors (simplest boards)
    clean = [r for r in rows if r["verdict"] == "CLEAN"]
    clean.sort(key=lambda r: (r["objects"] + r["colors"]))
    print("\nEasiest to hand-model (single clean avatar, simple board):")
    if clean:
        for r in clean[:5]:
            print(f"  {r['game']}  ({r['objects']} objects, {r['colors']} colors, "
                  f"{r['actions']} actions)")
    else:
        print("  (none scored CLEAN; next best are the 'compound' rows)")
    print("\nPick one whose WIN CONDITION you already know -> that's the spine target.")


if __name__ == "__main__":
    main()
