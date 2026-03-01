# Results — Hopper-v4: State vs. Pixel PPO

*This file is populated by running `bash scripts/reproduce_hopper_v4_20k.sh`
or `python scripts/aggregate_results.py --runs_dir <dir> --save_md reports/results_hopper_v4.md`.
Until then, values are marked as pending.*

---

## Benchmark configuration

| Parameter           | Value                           |
|---------------------|---------------------------------|
| Environment         | Hopper-v4 (Gymnasium + MuJoCo) |
| Total timesteps     | 20 000                          |
| Eval every          | 2 000 steps                     |
| Eval episodes       | 5 (deterministic, mean action)  |
| Seeds               | 0, 1, 2                         |
| PPO n\_steps        | 2048                            |
| PPO batch size      | 64                              |
| PPO epochs          | 10                              |
| Learning rate       | 3 × 10⁻⁴                       |
| Pixel obs shape     | (3, 64, 64) — CHW float32       |
| CNN encoder         | Conv(3→32,k8,s4) → Conv(32→64,k4,s2) → Conv(64→64,k3,s1) → Linear(→256) |

---

## Summary table (20k steps, 3 seeds, 5 eval episodes)

*Populated by `aggregate_results.py`. Run the reproduce script to fill these values.*

| Mode         | Final Eval Return (mean ± std) | Seeds | Final Step |
|:-------------|:-------------------------------|------:|----------:|
| state        | — pending —                    | 3     | 20 000    |
| pixels       | — pending —                    | 3     | 20 000    |

To regenerate:

```bash
bash scripts/reproduce_hopper_v4_20k.sh
```

---

## Observations

*(Filled after running the 20k benchmark.)*

- **Sample efficiency gap:** State-based PPO is expected to learn faster than
  pixel-based PPO at this budget. The CNN encoder adds representational
  overhead, and the visual input is higher-dimensional.

- **Variance across seeds:** Both modes are expected to show high variance at
  20 000 steps, as this is still early-stage training for Hopper-v4.

- **Training stability:** Value loss in pixel mode may be higher due to the
  more complex observation space requiring more capacity to model.

- **Baseline reference:** A random policy on Hopper-v4 yields approximately
  10–30 return per episode. Meaningful locomotion typically requires 200 000+
  steps with standard PPO hyperparameters.

---

## Interpretation

20 000 steps is insufficient to reach convergence on Hopper-v4 in either
mode. These results should be interpreted as early-training behaviour, not
final performance. The primary purpose of this benchmark is infrastructure
validation (multi-seed runs complete, metrics are logged correctly, plots
render) rather than performance comparison.

For meaningful comparison, use at least 500 000 steps.

---

## Limitations

- Short training budget (20k steps) — no convergence expected.
- 3 seeds only — variance estimates are wide.
- No hyperparameter tuning for either mode.
- Pixel mode uses no frame stacking (single frame per step).
- No pretrained visual encoder.
- Single environment (Hopper-v4); generality not assessed.

---

*Last updated: 2026-03-01. Regenerate with `bash scripts/reproduce_hopper_v4_20k.sh`.*
