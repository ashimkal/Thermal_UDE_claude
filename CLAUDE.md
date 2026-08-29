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
parameters and simulate. The training loop (data assembly across conditions,
loss, ADAM/BFGS optimization) is not itself in these files; it is documented
below as background for anyone reproducing or extending the training code.

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
  current, obtained by linear interpolation of the measured current profile.

The ODE system is solved with `Tsit5()` (`DifferentialEquations.jl`), and the
NN parameters `p` are the only free/trainable parameters of the model — `C1`,
`C2`, and `C_bat` are fixed physical constants.

## Neural network layer

```julia
U = Lux.Chain(
    Lux.Dense(3, 20, tanh),
    Lux.Dense(20, 1)
)
```

- Input (3 features): `[SOC, T, I]` — state of charge, cell temperature, and
  instantaneous current.
- Hidden layer: 20 units, `tanh` activation.
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

Training (not included as a runnable script in this repo, but the intended
procedure for reproducing `opt_para_case_*.jld2`) follows the standard
`SciML`/`DiffEqFlux` two-stage UDE training pattern:

1. **Stage 1 — ADAM.** Optimize the NN parameters `p` with the ADAM
   optimizer (adaptive, gradient-based, robust to a poor initial guess) to
   quickly move into a good region of parameter space and avoid local minima
   near initialization.
2. **Stage 2 — BFGS.** Switch to (L-)BFGS, a quasi-Newton method, initialized
   from the ADAM solution, to refine convergence and achieve a tighter fit
   once near the optimum. BFGS uses second-order curvature information and
   typically converges faster/more precisely than ADAM in this final phase.

Loss is computed against measured temperature (`T`) over each training
condition's time series, combined (e.g. summed/averaged) across the mixed
set of training conditions for that case.

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
