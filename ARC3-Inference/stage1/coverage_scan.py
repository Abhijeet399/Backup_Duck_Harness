import sys, json, glob, collections; sys.path.insert(0,"stage1")
import numpy as np
from qwen_inducer import QwenInducer
from goal_writer import mine_frames, discover_goal

KEY="/host-duck/ARC3-Inference/.cache/arc3_runtime/server-api-key"
ind=QwenInducer(endpoint="http://127.0.0.1:1234/v1", verbose=False,
                key_file=KEY, enable_thinking=True, max_tokens=8192)

# find every game that has recorded wins + its dominant winning action
def winning_action(game):
    paths=glob.glob(f"/host-duck/example-run/artifacts/{game}*events.jsonl")+\
          glob.glob(f"/shared/arc_3_results/root/*/artifacts/{game}*events.jsonl")
    acts=collections.Counter()
    for p in paths:
        try: events=[json.loads(l) for l in open(p)]
        except: continue
        for e in events:
            if e.get("level_completed") or e.get("just_won_level"):
                acts[e.get("action_name") or e.get("action_display")]+=1
    return acts

GAMES=["sp80","sb26","vc33","su15","ar25","tu93","ft09","re86","bp35","lp85","tn36"]
print(f"{'game':8s} {'wins':>5s} {'top_action':12s} {'goal_validated':16s} {'stats'}")
for game in GAMES:
    acts=winning_action(game)
    if not acts:
        print(f"{game:8s} {'0':>5s} {'-':12s} {'no wins':16s}"); continue
    total=sum(acts.values()); top=acts.most_common(1)[0]
    submit=top[0]
    # only try goal-discovery if there's a dominant submit action (>60% of wins)
    if top[1]/total < 0.6:
        print(f"{game:8s} {total:>5d} {str(dict(acts))[:12]:12s} {'mixed-action':16s} (movement-goal)"); continue
    goal,stats=discover_goal(game, submit, ind, max_tries=4, verbose=False)
    ok = "YES" if goal else "no"
    print(f"{game:8s} {total:>5d} {submit:12s} {ok:16s} {stats}")
