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
