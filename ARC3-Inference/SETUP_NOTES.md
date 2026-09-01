
## Results ladder (official 25, single pass)
- 0.6083  baseline: TP=1, enforce_eager, 16K, 20min
- 0.9883  TP=2 + CUDA graphs + 32K context match, 20min
- 1.5934  same, 30min/game
- 1.6576  + stagnation supervisor + analyzer.max_output=2048, 30min
          first level-3 (ft09=17.93); beats Tufa Duck public ref (~1.6)
          note: cap traded breadth for depth (ar25/sp80 -> 0, ft09 L2->L3)
          13/25 games still score 0 (reasoning-limited -> Stage 1)

## Stage 1 feasibility probe (result)
- Tool: wm_probe.py — asks the served model to induce step(board,action) from real ft09 diffs.
- Finding: Qwen3.6-27B (thinking on, diffs not raw boards) CORRECTLY characterized ft09
  mechanics: ACTION6 turns a 6x6 block of 9s->8s + decrements a 2-cell bottom-row counter,
  and tracked the block's top-left across transitions. Strong signal the model can do
  world-model induction — best suited to an ITERATIVE write->backtest->revise loop, not one-shot.
- Caveat: needs generous token budget + thinking on; one-shot probe understates ability.
- Verdict: green light to build Stage 1 (executable world-model) on Qwen; revisit stronger
  model (GPT-OSS-120B on 96GB card) only if the iterative harness plateaus.
