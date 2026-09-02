# Thermal-UDE

Julia implementation of a Universal Differential Equation (UDE) that predicts
lithium-ion battery cell temperature by combining a lumped thermal ODE with a
neural network that estimates internal heat generation. Companion code for
"Physics-Informed Machine Learning for Battery Heat Generation Estimation and
Temperature Prediction with Minimal Experimentation" (Energy and AI,
https://doi.org/10.1016/j.egyai.2026.100808).

## Repository layout

- `Case_1/Temp_pred_case1.jl` — loads `Dataset.jld2` + `opt_para_case_1.jld2`,
  solves the UDE with the trained parameters, and plots predicted vs.
  measured temperature.
- `Case_2/Temp_pred_case2.jl` — same structure, using `opt_para_case_2.jld2`.
  Case 2's neural network is trained starting from Case 1's optimized
  parameters (warm-started), not from a fresh random initialization.

Both scripts are inference/plotting scripts only — they load pre-trained
parameters and simulate. The training loop lives in `train.jl` (repo root),
documented in full under "Optimization procedure" below.

- `train.jl` — trains Case 1 or Case 2 (`julia --project=. train.jl 1` or
  `2`), writing the optimized parameters, validation metrics (RMSE/MAE/R²),
  and a validation plot to `results/case1/` or `results/case2/`. Looks for
  `Case_N/unique_Datasets.jld2` (key `"Datasets"`) first, falling back to
  the shipped `Case_N/Dataset.jld2` (key `"Dataset"`) if that isn't present.
  Checkpoints to `checkpoints/` and can be safely re-run to resume training
  that didn't finish in one sitting — see "Wall-clock budget, checkpointing,
  and resuming across runs" below.

## Governing equation (UDE)

State vector `u = [SOC, T]`.

**SOC dynamics** (known physics, Coulomb counting):

```
dSOC/dt = -I(t) / C_bat,   C_bat = 5 * 3600 (A·s, 5 Ah cell)
```

**Temperature dynamics** (lumped thermal model with NN-estimated heat generation):

```
dT/dt = C1 * (T - T∞) + C2 * G
```

- `T∞`: ambient temperature (initial condition / first sample of each test).
- `C1 = -0.00153`, `C2 = 0.020306`: fixed thermal parameters (heat-loss
  coefficient and a lumped thermal-mass/heat-generation scaling term).
- `G = U([SOC, T, I], p)^2`: heat generation, estimated by the neural network
  `U` (universal approximator) and squared so `G >= 0`, since resistive/
  irreversible heat generation cannot be negative. `I = I(t)` is the applied
  current. For a genuinely time-varying profile (WLTP) this is a linear
  interpolation of the measured current; for a constant-current
  discharge (0.5C/1C/2C) `I` is just the fixed discharge current as a
  plain scalar, not an interpolant — no interpolation lookup is needed
  or performed in that case, which is why Case 1 (constant-current only)
  trains noticeably faster than Case 2 (which includes WLTP).

The ODE system is solved with `Tsit5()` (`DifferentialEquations.jl`), and the
NN parameters `p` are the only free/trainable parameters of the model — `C1`,
`C2`, and `C_bat` are fixed physical constants.

## Neural network layer

The **shipped, pre-trained** parameters (`Case_1/opt_para_case_1.jld2`,
`Case_2/opt_para_case_2.jld2`, loaded by `Temp_pred_case1.jl` /
`Temp_pred_case2.jl`) were trained with a single-hidden-layer network:

```julia
U = Lux.Chain(
    Lux.Dense(3, 20, tanh),
    Lux.Dense(20, 1)
)
```

**`train.jl` uses a different, larger architecture** — two 20-unit hidden
layers instead of one:

```julia
U = Lux.Chain(
    Lux.Dense(3, 20, tanh),
    Lux.Dense(20, 20, tanh),
    Lux.Dense(20, 1)
)
```

These two networks are **not parameter-compatible** — a `ComponentArray`
trained against one has the wrong shape for the other. Because of this,
`train.jl`'s Case 2 warm-start has **no fallback to the shipped
`opt_para_case_1.jld2`** (that would silently hit a shape mismatch); it only
accepts a completed Case 1 checkpoint or `results/case1/opt_para_case1_trained.jld2`.
**Case 1 must be trained (and fully converged) with `train.jl` first** —
`julia train.jl 2` errors out immediately, with a clear message, if it isn't.

- Input (3 features): `[SOC, T, I]` — state of charge, cell temperature, and
  instantaneous current.
- Hidden layer(s): 20 units each, `tanh` activation.
- Output (1 unit): linear, no activation — squared in the ODE RHS to yield
  the non-negative heat generation term `G`.
- Parameters initialized via `Lux.setup` with `StableRNG(1111)` for
  reproducibility; `p` (a `ComponentArray`) is the vector optimized during
  training, `st` is the (non-trainable) layer state.

## Training / validation conditions

Experimental conditions are indexed by C-rate/profile and ambient
temperature: C-rates `{0.5C, 1C, 2C, WLTP}` × ambient temperatures
`{0°C, 10°C, 25°C}` = 12 possible conditions per dataset.

### Case 1 — constant-current only

- **Training condition (mixed/shuffled across conditions during training):**
  - 0.5C @ 0°C
  - 0.5C @ 25°C
  - 1C @ 10°C
  - 1C @ 25°C
  - 2C @ 0°C
  - 2C @ 10°C
- **Validation condition:** the remaining constant-current and WLTP
  conditions not included in the Case 1 training mix (e.g. 0.5C @ 10°C,
  1C @ 0°C, 2C @ 25°C, and the WLTP drive-cycle profiles), used to test
  generalization of the trained heat-generation network beyond the
  conditions it was fitted on.
- NN parameters initialized from `StableRNG(1111)` (cold start).

### Case 2 — mixed constant-current + WLTP, warm-started from Case 1

- **Initialization:** NN parameters are warm-started from Case 1's optimized
  parameters (`opt_para_case_1.jld2`), not re-initialized from scratch.
- **Training condition (mixed/shuffled across conditions during training):**
  - 0.5C @ 0°C
  - WLTP @ 10°C
  - 2C @ 25°C
  - 0.5C @ 25°C
  - WLTP @ 0°C
  - 1C @ 10°C
- **Validation condition:** the remaining conditions not used in the Case 2
  training mix (e.g. 1C @ 0°C, 0.5C @ 10°C, WLTP @ 25°C, and other untrained
  C-rate/temperature combinations), used to test generalization once a
  realistic drive cycle (WLTP) is included in training.

The plotting scripts (`Temp_pred_case1.jl` / `Temp_pred_case2.jl`) currently
demonstrate predictions on an illustrative set of 6 conditions (2C and WLTP
at 0/10/25°C) using each case's trained parameters — these are for
visualization only and should not be read as the authoritative train/
validation split described above.

## Optimization procedure

Training is implemented in `train.jl` (repo root; `julia --project=. train.jl
1` or `2`). It reproduces `opt_para_case_*.jld2` via a `SciML`/`DiffEqFlux`-
style ADAM → BFGS pipeline, with the two cases following different concrete
recipes:

### Case 1

1. **Mini-batch ADAM.** 5500 iterations, cycling round-robin through the 6
   training conditions — each iteration's objective is a single condition's
   plain MSE against measured temperature (**not normalized**), not the
   combined loss. Learning rate `0.001`. This is a fast warm-up: no
   interpolation is involved (Case 1 is constant-current only), so each
   iteration is cheap.
2. **ADAM on the total normalized loss, until it plateaus.** Objective is
   the sum, across all 6 training conditions, of each condition's MSE
   divided by that condition's overall temperature rise (`T[end] - T[1]`).
   "Plateaued" means the relative improvement over a trailing window of
   iterations has fallen below a small tolerance. Learning rate `0.001`.
3. **BFGS refinement, also on the total normalized loss** (same metric as
   step 2 — not the raw/unnormalized sum), initialized from step 2's result.
4. **Stopping criterion:** training stops as soon as the total **normalized**
   loss drops below `0.1`, checked after both step 2 and step 3. If it's
   still not below `0.1` after BFGS, steps 2–3 repeat (another round of
   plateau-ADAM followed by BFGS) until it is.

NN parameters initialized from `StableRNG(1111)` (cold start) before mini-
batching begins.

### Case 2

Warm-started from Case 1's optimized parameters (see the network architecture
compatibility caveat above — Case 1 must have actually finished) — no
mini-batch stage, no BFGS. A single ADAM stage on the total **raw** (not
normalized) loss — the plain sum of per-condition MSE — learning rate
`0.001`, run until the loss drops below `3.5`. Because Case 2's training mix
includes WLTP conditions, those conditions' current is a genuine
interpolation of the measured profile (constant-current conditions in the
mix still use a plain scalar).

All iteration counts, learning rates, plateau tolerances, and loss targets
are overridable via environment variables read at the top of `train.jl`
(e.g. `UDE_MINIBATCH_ITERS`, `UDE_CASE1_TARGET`, `UDE_CASE2_TARGET`) so CI
can run a reduced smoke test without editing the script.

Loss is always computed against measured temperature (`T`) over each
training condition's time series.

### Wall-clock budget, checkpointing, and resuming across runs

Reaching these loss targets can take longer than a single run has time for
(e.g. one GitHub Actions job). `train.jl` supports checkpointing so training
can be resumed across as many separate runs as it takes:

- `UDE_MAX_WALLTIME_SECONDS` bounds how long a single `julia train.jl <case>`
  invocation trains before it stops itself, saves a checkpoint, and exits
  cleanly (default: effectively unlimited, for local/manual runs).
- Checkpoints live in `checkpoints/case1_checkpoint.jld2` /
  `checkpoints/case2_checkpoint.jld2` (parameters + which stage to resume
  at), plus a plain-text `checkpoints/case{1,2}_status.txt` (`"done"` or
  `"in_progress"`) for easy shell-script inspection. Unlike `results/`,
  `checkpoints/` is **not** git-ignored — it's meant to be committed so it
  survives across separate CI runs.
- Re-running `julia train.jl <case>` automatically resumes from its
  checkpoint (if one exists and isn't already marked done) instead of
  starting over. Case 1 resumes at the exact stage (mini-batch / a specific
  round's ADAM / that round's BFGS) it was interrupted at; Case 2 just
  continues ADAM from the checkpointed parameters.
- `.github/workflows/train.yml` sets `UDE_MAX_WALLTIME_SECONDS` to 5 hours
  (leaving headroom under its own `timeout-minutes: 340`), trains Case 1,
  only attempts Case 2 if Case 1's status is `"done"`, commits any
  checkpoint changes back to the branch, uploads whatever's in `results/` as
  an artifact, and — if either case isn't `"done"` yet — re-dispatches
  itself (`gh workflow run train.yml`) so a fresh job continues training
  from the just-committed checkpoint. This repeats automatically, run after
  run, until both cases converge.

## Acceptance criterion

A trained model (Case 1 or Case 2) is considered acceptable when the
coefficient of determination between predicted and measured temperature
satisfies:

```
R^2 > 0.80
```

evaluated on the validation condition(s) for that case (not just the
training conditions), confirming the model generalizes to unseen
C-rate/temperature/profile combinations.

## Running the existing scripts

```bash
cd Case_1   # or Case_2
julia Temp_pred_case1.jl   # or Temp_pred_case2.jl
```

Requires `DifferentialEquations`, `Lux`, `JLD2`, `Plots`, `StableRNGs`,
`Statistics`, `ComponentArrays`, `Interpolations`, `MAT` in the active Julia
environment. Output: a 3×2 panel plot of measured vs. predicted temperature
per condition, saved as `Thermal_UDE_Case1.png` (note: both scripts currently
save to this same filename — rename before comparing Case 1 and Case 2
outputs side by side).
