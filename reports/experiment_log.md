# Experiment Log

Append one entry per meaningful experiment.
Do not edit or backfill past entries.
No marketing language. Factual reporting only.

---

## Entry 001 — Phase 4 Setup: Multi-Seed State vs. Pixel Comparison

**Date:** 2026-03-01
**Environment:** Hopper-v4 (Gymnasium + MuJoCo)
**Obs modes:** state, pixels
**Img size:** 64 × 64 (pixels mode only)
**Seed(s):** 0, 1, 2
**Timesteps:** 2 000 (smoke run; not a meaningful learning experiment)
**Eval every:** 1 000 steps
**n_steps:** 256 | **batch_size:** 64 | **epochs:** 2
**Device:** cpu

### Purpose

Phase 4 infrastructure smoke test: verify that `run_compare.py` launches all
6 runs (2 modes × 3 seeds) without error, that `metrics.csv` files are
written correctly for each seed, and that `plot_results.py` produces a
`compare_eval_return.png` from the resulting data.

This is not intended to demonstrate learning. Hopper requires significantly
more than 2 000 steps to show measurable improvement.

### Observations

- Training stability: n/a (too few steps)
- Sample efficiency differences: not measurable at this scale
- Representation bottlenecks: not observable yet
- Variance across seeds: expected to be high at 2 000 steps

### Quantitative

- Initial eval return (state):  ~10–30 (random policy baseline)
- Initial eval return (pixels): ~10–30 (random policy baseline)
- Final eval return: not meaningful at 2 000 steps
- Mean ± std across seeds: not reported (too few steps)

### Bugs encountered

None during infrastructure setup.

### Fixes applied

None.

### Limitations

- Compute budget: smoke run only (2 000 timesteps per run).
- Short horizon: no learning signal expected.
- Hyperparameter sensitivity: default PPO hyperparameters not tuned for
  pixel-based Hopper; pixel mode is expected to learn more slowly.
- Single environment: Hopper-v4 only; no cross-environment generalisation.

---

## Entry 002 — Phase 5 Research-Artifact Upgrade

**Date:** 2026-03-01
**Environment:** Hopper-v4 (Gymnasium + MuJoCo)
**Obs modes:** state, pixels
**Img size:** 64 × 64 (pixels mode only)
**Seed(s):** 0, 1, 2
**Timesteps:** 20 000 (reproduce_hopper_v4_20k.sh benchmark)
**Eval every:** 2 000 steps
**n_steps:** 2048 | **batch_size:** 64 | **epochs:** 10
**Device:** cpu

### Purpose

Structural upgrade of the repository to research-artifact tier.
No algorithm changes. Changes include:

- Full README rewrite with research framing, protocol tables, and
  reproducibility narrative.
- `scripts/aggregate_results.py`: multi-seed aggregation with markdown table
  output and overlay plot generation.
- `scripts/reproduce_hopper_v4_20k.sh`: end-to-end reproduction script for
  the 20k-step benchmark (3 seeds, 5 eval episodes, both modes).
- `tests/test_smoke.py` and `tests/test_determinism.py`: pytest suite covering
  env construction, model shapes, GAE correctness, and seed reproducibility.
- `.github/workflows/ci.yml`: CI workflow (imports + smoke tests, < 5 min).
- `reports/results_hopper_v4.md`: structured results document (template;
  populated by running the reproduce script).
- `reports/experiment_log.md`: updated with this entry.
- `pyproject.toml`: added `slow` pytest marker.

### Observations

- Training stability: infrastructure validated via smoke run.
- Sample efficiency differences: not yet quantified at 20k scale.
- Representation bottlenecks: not yet quantified.
- Variance across seeds: requires the 20k reproduce run to assess.

### Quantitative

- Initial eval return: ~10–30 (random policy; smoke runs confirmed).
- Final eval return (20k): pending — run `bash scripts/reproduce_hopper_v4_20k.sh`.
- Mean ± std: pending.

### Bugs encountered

None in Phase 5 structural changes.

### Fixes applied

Added `slow` marker to `pyproject.toml` to suppress pytest marker warning.

### Limitations

- 20k benchmark is insufficient for convergence; serves as infrastructure
  validation and an honest lower-bound comparison.
- No pixel-mode hyperparameter tuning.
- CI does not run full training (MuJoCo + training would exceed 5 min budget).

---

## Entry 003 — Phase 6: Frame Stacking Benchmark (20k · 3 conditions · 3 seeds)

**Date:** 2026-03-04
**Environment:** Hopper-v4 (Gymnasium + MuJoCo)
**Obs modes:** state, pixels (no stack), pixels (stack=4)
**Img size:** 64 × 64 (pixels modes only)
**Seed(s):** 0, 1, 2
**Timesteps:** 20 000 (`reproduce_hopper_v4_20k.sh`)
**Eval every:** 2 000 steps · 5 deterministic eval episodes
**n_steps:** 2048 | **batch_size:** 64 | **epochs:** 10
**Device:** cpu

### Purpose

First full 3-condition benchmark following Phase 6 (frame stacking) implementation.
`FrameStackWrapper` concatenates N consecutive pixel frames along the channel axis,
giving the policy temporal information unavailable from a single frame.
This entry records the first run comparing state / pixels / pixels_fs4 under
identical hyperparameters and the same `reproduce_hopper_v4_20k.sh` pipeline.

### Observations

- **Training stability (state):** Clear upward trend beginning around 12k steps.
  Mean rises from ~50 at 2k steps to 219 by 20k. Std narrows over training (8.7 at final step ≈ 4% of mean), indicating stable cross-seed learning.

- **Training stability (pixels, no stack):** Flattens early. Curve plateaus around 50–80 from step 4k onward; little improvement after that. Very low variance across seeds (std 3.1 ≈ 6%), suggesting all three seeds converge to the same limited solution.

- **Training stability (pixels, stack=4):** High early returns in some seeds (one seed reaches ~200 at step 4k), followed by a decline stabilising around 65–80 by final steps. Final std of 14.2 (≈22% of mean) is notably higher than the other two conditions, indicating unstable cross-seed behaviour. The early spike and decline may reflect a seed that briefly discovers a locomotion policy before it degrades under continued optimisation pressure without sufficient temporal representation.

- **Sample efficiency differences:** State mode diverges clearly from both pixel conditions after ~12k steps. The two pixel conditions track each other in the early phase and separate modestly by the final step (65.6 vs 50.1).

- **Representation bottleneck:** Both pixel conditions remain well below state at 20k steps. Stack=4 partially reduces the gap but does not resolve the fundamental representation bottleneck at this training budget.

- **Variance across seeds:** State: low (σ=8.7). Pixels no-stack: very low (σ=3.1). Pixels stack=4: high (σ=14.2). Frame stacking introduces more inter-seed variance, consistent with a more complex optimisation landscape.

### Quantitative

| Mode              | Final Eval Return (mean ± std) | Seeds | Step   |
|:------------------|:-------------------------------|------:|-------:|
| state             | 219.2 ± 8.7                    |     3 | 20 480 |
| pixels (no stack) |  50.1 ± 3.1                    |     3 | 20 480 |
| pixels (stack=4)  |  65.6 ± 14.2                   |     3 | 20 480 |

- State vs pixels (no stack): **4.4×** gap.
- State vs pixels (stack=4): **3.3×** gap.
- Stack=4 vs no-stack: **+31%** relative improvement.
- Stack=4 std is **4.6×** the no-stack std — frame stacking increases variance substantially.

### Bugs encountered

None. `FrameStackWrapper` and `make_env.py` `frame_stack` parameter operated as
expected. The PIL 12-channel error (from an earlier smoke test) had already been fixed.

### Fixes applied

None required for this run.

### Limitations

- 20k steps is far below convergence for Hopper-v4 in any condition.
  Results characterise early-training dynamics, not asymptotic capability.
- Three seeds give wide variance estimates; strong conclusions require 5–10 seeds.
- Identical PPO hyperparameters across all conditions. Pixel-based PPO,
  especially with frame stacking, may benefit from lower learning rates or
  larger rollout buffers.
- CPU-only run. No GPU acceleration; pixel-mode runs are 3–8× slower per step.
- The early spike and decline in stack=4 may be an artefact of the short horizon;
  longer runs could clarify whether improvement is sustained.

---

## Entry 004 — 1M-step Benchmark (3 conditions · 3 seeds · MPS)

**Date:** 2026-03-06
**Environment:** Hopper-v4 (Gymnasium + MuJoCo)
**Obs modes:** state, pixels (no stack), pixels (stack=4)
**Img size:** 64 × 64 (pixels modes only)
**Seed(s):** 0, 1, 2
**Timesteps:** 1 000 000 (`reproduce_hopper_v4_1m.sh`)
**Eval every:** 10 000 steps · 5 deterministic eval episodes
**n_steps:** 2048 | **batch_size:** 64 | **epochs:** 10
**Device:** mps (Apple Silicon)

### Purpose

First full 1M-step run across all three conditions using the `reproduce_hopper_v4_1m.sh`
script. Primary questions addressed:

1. Does the pixel–state gap narrow or widen at longer horizons?
2. Does frame stacking (stack=4) sustain its 20k advantage over no-stack pixels?
3. Does pixel-mode variance reduce with more training?
4. Does state-based PPO approach meaningful locomotion performance on Hopper-v4?

### Observations

- **Training stability (state):** Policy improves slowly until ~200k steps, then
  accelerates strongly, reaching peaks of ~3000–3500 return in favourable seeds.
  However, Hopper-v4 is unstable — once the hopper falls, episode returns collapse,
  causing large oscillations throughout the second half of training. The final eval
  return of 1915 ± 992 reflects seeds landing at very different positions on this
  oscillation (some may be near peaks, others in troughs at step 1M).

- **Training stability (pixels, no stack):** Flat and stable throughout. The curve
  rises very slowly, reaching ~121 by 1M steps. No dramatic instability or collapse
  visible. Std of 40.2 is wide relative to mean (33%) but curves are well-separated
  in absolute scale from state.

- **Training stability (pixels, stack=4):** Similarly flat. Final return of 97.7
  is below no-stack despite an early advantage at 20k steps. No catastrophic
  collapse, but the added input complexity (C=12) does not translate to better
  performance at this scale.

- **Sample efficiency differences:** State advantage compounds over time. The eval
  curve for state is essentially flat until ~200k steps (consistent with 20k
  observations), then rises sharply. Pixel conditions show only slow, steady
  improvement throughout — no phase transition.

- **Representation bottleneck:** Both pixel conditions finish well below 125.
  On the scale of the state learning curve, both pixel curves appear as near-flat
  lines hugging the bottom of the plot, illustrating how severe the bottleneck is
  at 1M steps with a randomly-initialised CNN trained by PPO alone.

- **Variance across seeds:** State: very high (σ=992, ≈52% of mean) due to
  Hopper instability. No-stack: σ=40.2 (≈33%). Stack=4: σ=36.0 (≈37%).
  Pixel variance did not substantially reduce relative to state as training lengthened.

- **Frame stacking reversal confirmed:** At 20k, stack=4 > no-stack (65.6 vs 50.1).
  At 1M, no-stack > stack=4 (121.8 vs 97.7). Stacking appears to hinder rather than
  help optimisation at longer horizons under vanilla PPO with fixed hyperparameters.

### Quantitative

| Mode              | Final Eval Return (mean ± std) | Seeds | Step      |
|:------------------|:-------------------------------|------:|----------:|
| state             | 1915.1 ± 992.0                 |     3 | 1 001 472 |
| pixels (no stack) |  121.8 ± 40.2                  |     3 | 1 001 472 |
| pixels (stack=4)  |   97.7 ± 36.0                  |     3 | 1 001 472 |

- State vs pixels (no stack): **15.7×** gap (was 4.4× at 20k).
- State vs pixels (stack=4): **19.6×** gap (was 3.3× at 20k).
- Stack=4 vs no-stack: **−20%** (stack=4 is worse at 1M; was +31% at 20k).
- State absolute improvement from 20k: 219 → 1915 (+775%).
- No-stack absolute improvement from 20k: 50.1 → 121.8 (+143%).
- Stack=4 absolute improvement from 20k: 65.6 → 97.7 (+49%).

### Bugs encountered

`DEVICE=cuda` failed on macOS (CPU-only PyTorch build, CUDA not available).

### Fixes applied

Added `mps` to `--device` choices in `train_ppo.py` and `run_compare.py`.
Added startup device-availability validation with actionable error messages.
Run completed successfully with `DEVICE=mps`.

### Limitations

- Identical PPO hyperparameters across all conditions; no per-condition tuning.
  Pixel modes would likely benefit from a lower learning rate and/or larger
  rollout buffer at this scale.
- Three seeds. State std of ~992 is very wide; seed-level values would reveal
  whether the high mean reflects all three seeds or one outlier.
- Hopper-v4 instability means 1M-step state performance is sensitive to the
  final-step snapshot. A return measured at peak vs. trough differs by >2×.
- MPS (Apple Silicon) device used; minor numerical differences from CUDA
  are possible, though not expected to materially affect results.

---

## Template (copy for new entries)

**Date:**
**Environment:**
**Obs modes:**
**Img size:**
**Seed(s):**
**Timesteps:**
**Eval every:**
**n_steps:** | **batch_size:** | **epochs:**
**Device:**

### Purpose

### Observations

- Training stability:
- Sample efficiency differences:
- Representation bottlenecks:
- Variance across seeds:

### Quantitative

- Initial eval return:
- Final eval return:
- Mean ± std across seeds:

### Bugs encountered

### Fixes applied

### Limitations
