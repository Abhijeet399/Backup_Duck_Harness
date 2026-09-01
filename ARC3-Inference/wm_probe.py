#!/usr/bin/env python3
import argparse, json, sys, re, statistics, os

def load_transitions(path):
    events = [json.loads(l) for l in open(path)]
    tr, prev = [], None
    for e in events:
        b = e.get("board")
        if e.get("type") == "action" and e.get("action_name") and prev is not None:
            tr.append({"before": prev, "action": e["action_name"], "after": b,
                       "board_changed": bool(e.get("board_changed"))})
        if b is not None: prev = b
    return tr

def diff_cells(a, b):
    d = []
    for r in range(len(a)):
        for c in range(len(a[r])):
            if a[r][c] != b[r][c]:
                d.append((r, c, a[r][c], b[r][c]))
    return d

def cell_accuracy(pred, truth):
    if not isinstance(pred, list) or len(pred) != len(truth): return 0.0
    tot = same = 0
    for pr, trow in zip(pred, truth):
        if not isinstance(pr, list) or len(pr) != len(trow): return 0.0
        for x, y in zip(pr, trow):
            tot += 1; same += (x == y)
    return same / tot if tot else 0.0

def extract_code(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL)
    return m.group(1) if m else t

def build_prompt(train, actions):
    p = ["You are reverse-engineering the transition rule of a 64x64 grid puzzle.",
         "Cells are integers (colors). Instead of full boards, each example lists ONLY "
         "the cells that CHANGED as (row, col, old_value, new_value).",
         f"Action space: {sorted(actions)}.", "",
         "First reason briefly in plain prose OUTSIDE any code block. Then write ONE "
         "Python function `step(board, action)` returning the next board (same 2D shape). "
         "The code block must be directly executable: NO analysis, NO multi-line docstrings, "
         "only working code with short comments. End with the code block.", ""]
    for i, t in enumerate(train):
        d = diff_cells(t["before"], t["after"])
        p += [f"### Transition {i+1} action={t['action']} changed_cells={len(d)}",
              "changes (r,c,old,new): " + json.dumps(d[:200]), ""]
    return "\n".join(p)

def call_model(base_url, model, prompt, api_key, max_tokens=8192):
    import urllib.request
    body = json.dumps({"model": model, "messages":[{"role":"user","content":prompt}],
                       "temperature":0.2, "max_tokens":max_tokens,
                       "chat_template_kwargs":{"enable_thinking":True}}).encode()
    h = {"Content-Type":"application/json"}
    if api_key: h["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/")+"/chat/completions", data=body, headers=h)
    with urllib.request.urlopen(req, timeout=1200) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default="vrfai/Qwen3.6-27B-FP8")
    ap.add_argument("--api-key", default=os.environ.get("PROBE_API_KEY",""))
    ap.add_argument("--n-train", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=2)
    a = ap.parse_args()

    trans = load_transitions(a.events)
    changed = [t for t in trans if t["board_changed"]]
    pool = changed if len(changed) >= a.n_train+a.n_test else trans
    print(f"Loaded {len(trans)} transitions ({len(changed)} changed). Pool={len(pool)}.")
    train, test = pool[:a.n_train], pool[a.n_train:a.n_train+a.n_test]
    actions = {t["action"] for t in trans}
    prompt = build_prompt(train, actions)
    print(f"Prompt ~{len(prompt)} chars. Calling model...")
    reply = call_model(a.base_url, a.model, prompt, a.api_key)
    code = extract_code(reply)
    print("\n===== MODEL step() (first 1800 chars) =====\n"+code[:1800]+"\n===== end =====\n")
    ns={}
    try:
        exec(code, ns); step=ns.get("step")
        if not step: print("VERDICT: INCOHERENT — no step()."); return
    except Exception as ex:
        print(f"VERDICT: INCOHERENT — exec failed: {ex}"); return
    def ev(label, items):
        accs=[]
        for i,t in enumerate(items):
            try:
                pred=step([r[:] for r in t["before"]], t["action"])
                acc=cell_accuracy(pred, t["after"]); print(f"  {label}[{i}] {t['action']}: acc={acc:.4f}")
            except Exception as ex:
                acc=0.0; print(f"  {label}[{i}] {t['action']}: RAISED {ex}")
            accs.append(acc)
        return statistics.mean(accs) if accs else 0.0
    print("TRAIN:"); tr=ev("train",train)
    print("HELD-OUT:"); te=ev("test",test)
    idb=statistics.mean(cell_accuracy(t["before"],t["after"]) for t in test)
    print(f"\nSUMMARY: train={tr:.4f} held_out={te:.4f} identity_baseline={idb:.4f}")
    print("\n--- VERDICT ---")
    if te>=0.999 and te>idb+0.0005:
        print("STRONG: generalizes to held-out AND beats do-nothing. Stage 1 viable on Qwen.")
    elif tr>=0.999 and te<0.99:
        print("OVERFIT: fits shown, fails held-out. Needs stronger model.")
    elif te<=idb+0.0005:
        print("NO SIGNAL: not beating do-nothing baseline. Needs stronger model / richer probe.")
    else:
        print(f"PARTIAL: held_out={te:.4f}. Some structure; compare a stronger model before committing.")

if __name__=="__main__": main()
