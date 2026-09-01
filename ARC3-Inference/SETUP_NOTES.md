
## Results ladder (official 25, single pass)
- 0.6083  baseline: TP=1, enforce_eager, 16K, 20min
- 0.9883  TP=2 + CUDA graphs + 32K context match, 20min
- 1.5934  same, 30min/game
- 1.6576  + stagnation supervisor + analyzer.max_output=2048, 30min
          first level-3 (ft09=17.93); beats Tufa Duck public ref (~1.6)
          note: cap traded breadth for depth (ar25/sp80 -> 0, ft09 L2->L3)
          13/25 games still score 0 (reasoning-limited -> Stage 1)
