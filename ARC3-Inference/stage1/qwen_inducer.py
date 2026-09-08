"""
qwen_inducer.py — Layer 2 of the Stage 1 inducer: the LLM seam.

Implements the Inducer protocol (stage1_controller.Inducer):
    __call__(history, current, divergence) -> list[WorldModel]

It does NOT ask Qwen for code. It asks for a rule-SPEC dict (the DSL in
rule_spec.py) and compiles that deterministically. The LLM does inference
(what is the rule); the compiler does coding (turn it into callables).

Lessons baked in from the ft09 feasibility probe:
  * thinking ON, generous max_tokens (the model reasons then emits),
  * feed COMPACT object-diffs, never raw 64x64 grids,
  * read content, fall back to reasoning field; strip prose, parse JSON.

Three call modes (per the controller call sites):
  cold   : current=None, divergence=None         -> induce fresh
  refine : current set,  divergence=None         -> improve current model
  repair : current set,  divergence=ObjectDiff    -> fix what just diverged
"""
from __future__ import annotations
import json, re, urllib.request
from typing import Optional

from stage1_core import Transition, WorldModel, object_diff
from rule_spec import compile_spec, SpecError


SCHEMA_DOC = """\
You output ONLY a JSON object (no prose, no code fence) describing the game's
transition rule, in this schema:

{
  "background": <int, the empty/background color>,
  "rules": [ <rule>, ... ],          // applied in order; first match wins
  "goal": <goal>,
  "terminal": <terminal or null>
}

rule kinds:
  {"kind":"translate","color":C,"action":"ACTION1","vector":[dr,dc],"blocked_by":[colors]}
     one C-colored object moves by (dr,dc) on that action; canceled if a
     destination cell holds a blocked_by color or is off-grid.
  {"kind":"translate_click","color":C,"vector":[dr,dc],"blocked_by":[colors]}
     clicking (ACTION6) ON a C-object pushes THAT object by (dr,dc).
  {"kind":"convert_cell","action":"ANY","from":A,"to":B,"count":N}
     each firing recolor N cells of color A to color B (a counter/consume).

goal kinds:
  {"kind":"color_absent","color":C}                 win when color C has 0 cells
  {"kind":"color_count","color":C,"eq":N}            win when count(color C)==N
  {"kind":"object_at","color":C,"row":R,"col":Col}   win when C-object top-left ==(R,Col)

terminal kinds (or null):
  {"kind":"color_count","color":C,"eq":0,"result":"GAME_OVER"}

Use the FEWEST rules that explain every transition. Only use colors you see in
the observations. Output the JSON object and nothing else."""


def _render_transition(t: Transition, background: int) -> str:
    d = object_diff(t.grid, t.next_grid, background)
    term = f" -> {t.terminal}" if t.terminal else ""
    return f"action={t.action!r}{term}: {d.summary()}"


def _render_history(history: list[Transition], background: int, cap: int = 40) -> str:
    lines = [_render_transition(t, background) for t in history[-cap:]]
    return "\n".join(lines)


def _guess_background(history: list[Transition]) -> int:
    # most common cell value in the first observed grid
    import numpy as np
    if not history:
        return 0
    g = history[0].grid
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmax()])


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    # strip code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        # find the first balanced {...} spanning the largest region
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            blob = text[start:end + 1]
    if blob is None:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


class QwenInducer:
    def __init__(self, base_url: str, model: str, api_key: str,
                 max_tokens: int = 8192, timeout: float = 600.0,
                 enable_thinking: bool = True, temperature: float = 0.3,
                 verbose: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.verbose = verbose

    # -- the Inducer protocol -------------------------------------------------
    def __call__(self, history: list[Transition],
                 current: Optional[WorldModel],
                 divergence) -> list[WorldModel]:
        if not history:
            return []
        background = _guess_background(history)
        prompt = self._build_prompt(history, current, divergence, background)
        reply = self._call(prompt)
        spec = _extract_json(reply)
        if spec is None:
            if self.verbose:
                print("[inducer] no JSON parsed from reply")
            return []
        # ensure background is present/consistent
        spec.setdefault("background", background)
        try:
            model = compile_spec(spec)
        except SpecError as ex:
            if self.verbose:
                print(f"[inducer] spec did not compile: {ex}")
            return []
        if self.verbose:
            print(f"[inducer] compiled a model with {len(spec.get('rules', []))} rules")
        return [model]

    # -- prompt construction --------------------------------------------------
    def _build_prompt(self, history, current, divergence, background) -> str:
        parts = [SCHEMA_DOC, "", f"background color appears to be {background}.", ""]
        parts.append("Observed transitions (object-level diffs):")
        parts.append(_render_history(history, background))
        parts.append("")
        if divergence is not None:
            parts.append("Your previous rule mispredicted the LAST step. The difference "
                         "between what you predicted and what actually happened is:")
            parts.append(divergence.summary() if hasattr(divergence, "summary") else str(divergence))
            parts.append("Revise the spec so it also explains this step.")
            parts.append("")
        elif current is not None:
            parts.append("You already proposed a model; improve it so it explains ALL "
                         "transitions with the fewest rules.")
            parts.append("")
        parts.append("Output the JSON spec now:")
        return "\n".join(parts)

    # -- server call ----------------------------------------------------------
    def _call(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
        except Exception as ex:
            if self.verbose:
                print(f"[inducer] server call failed: {ex}")
            return ""
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
