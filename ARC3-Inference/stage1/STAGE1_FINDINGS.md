# Stage 1 Findings — Executable World-Model Harness for ARC-AGI-3

Status as of this checkpoint. Repo: `Abhijeet399/Backup_Duck_Harness`, branch
`stage0-improvements`, dir `ARC3-Inference/stage1/`. Server config: **TP=1** (single
GPU, eval-matching), `enforce_eager=true`, `gpu_memory_utilization=0.90`,
`max_model_len=32768`.

## The headline

**The core Stage 1 bet is proven: a local 27B (Qwen3.6-27B-FP8) can build a
CERTIFIED executable world model of a real ARC-AGI-3 game.** On sp80, the induced
model reproduces every observed gameplay transition exactly (`exact_match_rate =
1.000`, HUD-masked). The full pipeline — observe → object-diff → LLM-induce →
compile → HUD-masked backtest → BFS plan → execute-and-verify — runs end-to-end on
a real game, stably, within the compute envelope.

**The remaining blocker is GOAL-DISCOVERY, and it is a genuine open research
problem, not a bug.** We can model the physics; we cannot yet reliably infer what
a game WANTS (its win condition), because wins are rarely observed and, when
observed, underdetermine the goal.

## What works (validated, committed)

- **Spine** (`stage1_core.py`, `stage1_controller.py`): object extraction, rigid
  `object_diff`, `WorldModel` (ordered local rules + goal_fn + terminal_fn),
  `backtest` (now HUD-mask-aware), `bfs_plan` (now wall-clock-bounded,
  `max_seconds`), `execute_and_verify` commit-and-verify, hypothesis pool.
- **Rule-spec DSL + compiler** (`rule_spec.py`): the LLM emits a structured spec
  (JSON); we compile deterministically to `Rule` callables. Primitives:
  `translate`, `translate_click`, `convert_cell`; goals: `color_absent`,
  `color_count`, `object_at`; terminal: `color_count`. Validated: hand-written
  push-game spec backtests 1.0.
- **QwenInducer** (`qwen_inducer.py`): implements the `Inducer` protocol; renders
  transition object-diffs, prompts Qwen for a spec, compiles, returns
  `list[WorldModel]`. Stamps a HUD `ignore_mask` on returned models.
  **Thinking OFF** (critical finding — see below).
- **HUD masking**: `WorldModel.ignore_mask` + masked `backtest`. Mask derived from
  a no-op action's changed cells, widened to the full timer row(s). This is what
  makes real-game backtest reach 1.0 (the timer bar otherwise breaks every
  transition). Derived in `run_stage1.py` via `--noop ACTION5`.
- **Smoke tests** (`example_smoke_test.py`, `hardened_smoke_test.py`): spine passes
  on discrete AND multi-object CLICK games. `test_rule_spec.py`,
  `test_qwen_inducer.py` pass.

## Key empirical findings

1. **Thinking OFF is ~16x faster AND more reliable for spec-emission.**
   thinking=True: 135s/call, often runs out of tokens mid-reasoning and emits no
   spec. thinking=False: **8.6s/call, backtest 1.0.** The DSL + precomputed
   object-diffs do the structuring that deliberation would otherwise need; the
   ft09-feasibility "needs thinking" result was for raw induction, not
   spec-emission in our scaffold. This solved the eval-envelope speed problem.
2. **TP=2 is unstable under inducer load on this CC-8.9 dual-GPU box.** Heavy
   requests reliably crash the engine (`EngineDeadError` / `shm_broadcast
   cancelled`). **TP=1 is stable** and matches the Kaggle single-GPU eval anyway.
   Develop on TP=1.
3. **HUD/timer bars break exact-match backtest.** Real games render per-action
   display elements (sp80: a color-14 timer bar in row 0, depletes ~2/action,
   hits 0 → GAME_OVER). The world model must predict game state, not pixels →
   mask HUD regions. No-op-diff derivation isolates them cleanly.
4. **BFS on a real 64x64 model needs a wall-clock bound.** A wrong/unreachable
   goal makes BFS grind to its node ceiling (200k) over minutes. `max_seconds`
   makes a bad goal fail in seconds so the loop can iterate.
5. **The 25 games' controls (measured, object_diff-based):** ZERO clean
   single-arrow games. 19/25 mouse (ACTION6 coordinate); the arrow games mostly
   move 2+ objects/action. sp80 (clean single arrow avatar) is the exception.
6. **NO game wins by simple cyclic movement** (tested 12 scoring games). Goals
   require understanding, by design.

## The open problem: goal-discovery (precisely characterized)

On sp80 we have a CERTIFIED physics model but cannot determine the win condition.
Empirically FALSIFIED (each tested by real execution, not speculation):
- **consume a color** (drive avatar to consume each candidate) — no win.
- **touch/adjacent a color** — no win.
- **reach a fixed position** — avatar reached the exact winning coord (12,24) from
  a mined real win; **level did NOT complete.** So position-reaching is not it.

The mined winning board (`goal_discovery` win-mining from Stage 0 run logs) was
**nearly identical to a normal board** — only the avatar had moved, and the
winning action was ACTION5 (a no-op). This strongly implies sp80's win trigger is
**not readable from a single board state**: it may be history-dependent (a
sequence), timer-value-dependent, or a derived/relative configuration we have not
characterized. A SINGLE observed win underdetermines the goal.

### Why this is hard (and expected)
This is the core difficulty ARC-AGI-3 is built around — "figure out what the game
wants." The Duck's LLM scored 0 on these games for the same reason. Goal-discovery
needs deliberate design, not more one-off hypotheses:
- infer goals from **multiple** mined wins (triangulate the underdetermined spec),
- support **history/timer-dependent** and **relative/derived** goal features,
- a multi-hypothesis generate-and-test loop that PROVOKES wins to confirm.

## Assets for continuing
- **Win-mining works**: `level_completed`/`just_won_level` flags in
  `/shared/arc_3_results/root/<TS>/artifacts/<game>_events.jsonl` give real
  winning transitions. Most scoring games have >=1; ft09 has 2-3. This is a
  labeled goal dataset from Stage 0.
- **provoke_win** (`goal_discovery.py`): candidate-goal exploration (fresh env per
  candidate, BFS-to-target on certified model, execute, watch for
  LEVEL_COMPLETED). Mechanism works; candidate SET is too narrow.

## Honest status
World-model induction on real ARC-AGI-3 games with a local 27B: **DONE, proven.**
Turning certified models into wins: **blocked on goal-discovery, a real research
problem now precisely scoped.** This is a strong, defensible position — the hard
infrastructure and the uncertain feasibility question are both settled; the
remaining work is a well-defined (hard) inference problem.
