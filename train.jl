# Trains the Thermal-UDE neural network heat-generation model, mirroring
# the validated Case 1 training template:
#
#   Case 1 (constant-current only): no interpolation of current is used
#   at all -- each training condition's current is a plain constant
#   Float64, not a callable, so the ODE RHS never performs an
#   interpolation lookup. Training is:
#     1. Mini-batch ADAM: 5500 iterations, cycling round-robin through
#        the 6 training conditions, one condition's MSE per iteration
#        -- NOT normalized.
#     2. Round loop: ADAM on the total NORMALIZED loss until it
#        plateaus, then BFGS refinement, also on the total NORMALIZED
#        loss. Stops as soon as either stage's normalized loss drops
#        below 0.1; if BFGS still isn't below 0.1, another round of
#        ADAM + BFGS runs.
#
#   Case 2 (mixed CC + WLTP, warm-started from Case 1): WLTP conditions
#   need interpolation (current genuinely varies over time), so the
#   ODE RHS calls an interpolant for those; CC conditions still use a
#   plain constant. Training is a single ADAM stage on the total RAW
#   (not normalized) loss, starting from Case 1's optimized
#   parameters, until the loss drops below 3.5.
#
# Usage:
#   julia --project=. train.jl 1     # train Case 1 (cold start, or resume)
#   julia --project=. train.jl 2     # train Case 2 (warm-started from Case 1, or resume)
#
# All iteration counts / targets can be overridden via environment
# variables (see the `const ... = parse(...)` block below) so CI can
# run a reduced smoke test while local/full runs use the real budget.
#
# A run may not have enough wall-clock time (e.g. one GitHub Actions job)
# to reach the loss targets above. Set UDE_MAX_WALLTIME_SECONDS to bound
# how long this run trains before it checkpoints its progress to
# checkpoints/ and exits cleanly (still not converged). Re-running
# `julia train.jl <case>` picks the checkpoint back up rather than
# starting over -- repeat as many times as needed until it converges.
# checkpoints/ is committed to the repo (unlike results/, which is
# git-ignored) specifically so it survives across separate runs/jobs.

using DifferentialEquations, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using Statistics
using StableRNGs, Lux, Zygote, Plots, ComponentArrays, JLD2
using Interpolations

const MINIBATCH_ITERS = parse(Int, get(ENV, "UDE_MINIBATCH_ITERS", "5500"))

const CASE1_ADAM_LR = parse(Float64, get(ENV, "UDE_CASE1_ADAM_LR", "0.001"))
const CASE1_TARGET = parse(Float64, get(ENV, "UDE_CASE1_TARGET", "0.1"))
const CASE1_PLATEAU_WINDOW = parse(Int, get(ENV, "UDE_CASE1_PLATEAU_WINDOW", "200"))
const CASE1_PLATEAU_RELTOL = parse(Float64, get(ENV, "UDE_CASE1_PLATEAU_RELTOL", "1e-4"))
const CASE1_ADAM_MAX_ITERS = parse(Int, get(ENV, "UDE_CASE1_ADAM_MAX_ITERS", "20000"))
const CASE1_BFGS_ITERS = parse(Int, get(ENV, "UDE_CASE1_BFGS_ITERS", "50"))
# Rounds are cheap to loop through (each round's own iteration counts are
# what actually bound the work); the real stop condition is CASE1_TARGET
# or the wall-clock budget below, so this is set high rather than tight.
const CASE1_MAX_ROUNDS = parse(Int, get(ENV, "UDE_CASE1_MAX_ROUNDS", "1000"))

const CASE2_ADAM_LR = parse(Float64, get(ENV, "UDE_CASE2_ADAM_LR", "0.001"))
const CASE2_TARGET = parse(Float64, get(ENV, "UDE_CASE2_TARGET", "3.5"))
const CASE2_ADAM_MAX_ITERS = parse(Int, get(ENV, "UDE_CASE2_ADAM_MAX_ITERS", "100000000"))

# ---------------------------------------------------------------------
# Wall-clock budget + checkpointing
#
# A single run (e.g. one GitHub Actions job) may not have enough wall-clock
# time to reach the loss targets above. UDE_MAX_WALLTIME_SECONDS bounds how
# long this run trains before saving a checkpoint and exiting cleanly
# (default: effectively unlimited, for local/manual runs). Re-running
# `julia train.jl <case>` afterwards resumes from checkpoints/ instead of
# starting over. checkpoints/ is committed to the repo (unlike results/,
# which is git-ignored) specifically so it survives across separate CI runs.
# ---------------------------------------------------------------------

const MAX_WALLTIME_SECONDS = parse(Float64, get(ENV, "UDE_MAX_WALLTIME_SECONDS", "1e9"))
const START_TIME = time()
const TIMED_OUT = Ref(false)

function time_budget_exceeded()
    if !TIMED_OUT[] && (time() - START_TIME) > MAX_WALLTIME_SECONDS
        TIMED_OUT[] = true
    end
    return TIMED_OUT[]
end

const CHECKPOINT_DIR = joinpath(@__DIR__, "checkpoints")
checkpoint_path(case) = joinpath(CHECKPOINT_DIR, "case$(case)_checkpoint.jld2")
status_path(case) = joinpath(CHECKPOINT_DIR, "case$(case)_status.txt")

function save_checkpoint(case, p, stage_idx, done)
    mkpath(CHECKPOINT_DIR)
    save(checkpoint_path(case), Dict("p" => p, "stage_idx" => stage_idx, "done" => done))
    write(status_path(case), done ? "done" : "in_progress")
end

# Returns (p, stage_idx, done); p is `nothing` and stage_idx is 1 if no
# checkpoint exists yet.
function load_checkpoint(case)
    path = checkpoint_path(case)
    if !isfile(path)
        return nothing, 1, false
    end
    d = load(path)
    return d["p"], d["stage_idx"], d["done"]
end

# function to find the end of the discharge in the constant current discharge tests.
function find_discharge_end(Current_data, start=5)
    for i in start:length(Current_data)
        if abs(Current_data[i]) == 0
            return i
        end
    end
    return -1
end

# function to get the current value for the constant current discharge tests
function current_val(Crate)
    if Crate == "0p5C"
        return 0.5 * 5.0
    elseif Crate == "1C"
        return 1.0 * 5.0
    elseif Crate == "2C"
        return 2.0 * 5.0
    elseif Crate == "1p5C"
        return 1.5 * 5.0
    end
end

# Prefers the deduplicated training dataset (unique_Datasets.jld2 / "Datasets"),
# falling back to the repo's shipped inference dataset (Dataset.jld2 / "Dataset").
function load_dataset(case)
    dataset_dir = joinpath(@__DIR__, "Case_$case")
    unique_path = joinpath(dataset_dir, "unique_Datasets.jld2")
    legacy_path = joinpath(dataset_dir, "Dataset.jld2")
    if isfile(unique_path)
        return load(unique_path)["Datasets"]
    elseif isfile(legacy_path)
        return load(legacy_path)["Dataset"]
    else
        error("No dataset file found in $dataset_dir (expected unique_Datasets.jld2 or Dataset.jld2)")
    end
end

function build_network()
    U = Lux.Chain(Lux.Dense(3, 20, tanh), Lux.Dense(20, 20, tanh), Lux.Dense(20, 1))
    _, st = Lux.setup(StableRNG(1111), U)
    return U, st
end

# The UDE model. I_src is either a plain Float64 (constant current -- no
# interpolation, used for every Case 1 condition and Case 2's CC conditions)
# or a callable interpolant I_src(t) (Case 2's WLTP conditions only).
function ude_rhs!(du, u, p, t, T∞, I_src, U, st)
    Cbat = 5 * 3600
    I = I_src isa Real ? I_src : I_src(t)
    du[1] = -I / Cbat

    C₁ = -0.00153
    C₂ = 0.020306

    G = (U([u[1], u[2], I], p, st)[1][1])^2
    du[2] = (C₁ * (u[2] - T∞)) + (C₂ * G)
end

rmse(y, ŷ) = sqrt(mean(abs2, y .- ŷ))
mae(y, ŷ) = mean(abs, y .- ŷ)
function r2_score(y, ŷ)
    ss_res = sum(abs2, y .- ŷ)
    ss_tot = sum(abs2, y .- mean(y))
    return 1 - ss_res / ss_tot
end

# Training / validation conditions per CLAUDE.md (Crate, ambient temperature °C).
const CASE_TRAIN_CONDITIONS = Dict(
    1 => [("0p5C", 0), ("0p5C", 25), ("1C", 10), ("1C", 25), ("2C", 0), ("2C", 10)],
    2 => [("0p5C", 0), ("WLTP", 10), ("2C", 25), ("0p5C", 25), ("WLTP", 0), ("1C", 10)],
)

const CASE_VAL_CONDITIONS = Dict(
    1 => [("0p5C", 10), ("1C", 0), ("2C", 25), ("WLTP", 0), ("WLTP", 10), ("WLTP", 25)],
    2 => [("0p5C", 10), ("1C", 0), ("1C", 25), ("2C", 0), ("2C", 10), ("WLTP", 25)],
)

struct Condition
    label::String
    prob::ODEProblem
    t::Vector{Float64}
    T::Vector{Float64}
end

function condition_label(crate, temp)
    c = replace(crate, "p" => ".")
    c = replace(c, "C" => " C")
    return "$(c), $(temp)°C"
end

# Builds one training/validation condition. WLTP conditions get a genuine
# interpolant (current varies over time); every other condition gets a
# plain constant current -- no interpolation, no per-step lookup cost.
function build_condition(crate, temp, data_file, U, st)
    data = data_file["$(crate)_T$(temp)"]
    if crate == "WLTP"
        t_raw = data["time"]
        T_raw = data["temperature"]
        current_raw = data["current"]
        first_indices = unique(idx -> t_raw[idx], eachindex(t_raw))
        t = collect(Float64, t_raw[first_indices])
        T = collect(Float64, T_raw[first_indices])
        current = current_raw[first_indices]
        I_src = LinearInterpolation(t, current, extrapolation_bc=Interpolations.Flat())
        T∞ = T[1]
        SOC0 = 0.9
    else
        n = find_discharge_end(data["current"])
        t_raw = data["time"][2:n]
        T_raw = data["temperature"][2:n]
        first_indices = unique(idx -> t_raw[idx], eachindex(t_raw))
        t = collect(Float64, t_raw[first_indices])
        T = collect(Float64, T_raw[first_indices])
        I_src = current_val(crate)
        T∞ = T[1]
        SOC0 = 1.0
    end

    model!(du, u, p, tt) = ude_rhs!(du, u, p, tt, T∞, I_src, U, st)
    prob = ODEProblem(model!, [SOC0, T∞], (t[1], t[end]), nothing)
    return Condition(condition_label(crate, temp), prob, t, T)
end

build_conditions(pairs, data_file, U, st) = [build_condition(crate, temp, data_file, U, st) for (crate, temp) in pairs]

# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------

function single_condition_mse(c::Condition, p)
    sol = solve(remake(c.prob, p=p), Tsit5(), saveat=c.t,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
    pred = Array(sol)[2, :]
    return mean(abs2, c.T .- pred)
end

# Normalized total loss (each condition's MSE divided by its overall
# temperature rise) -- Case 1's ADAM and BFGS objective, and the metric
# Case 1's 0.1 stopping criterion is checked against.
function totalloss_normalized(conditions, p)
    total = 0.0
    for c in conditions
        sol = solve(remake(c.prob, p=p), Tsit5(), saveat=c.t,
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
        pred = Array(sol)[2, :]
        err = mean(abs2, c.T .- pred)
        total += err / (c.T[end] - c.T[1])
    end
    return total
end

# Raw total loss (plain sum of per-condition MSE, no normalization) --
# Case 2's ADAM objective and 3.5 stopping criterion.
function totalloss_raw(conditions, p)
    total = 0.0
    for c in conditions
        sol = solve(remake(c.prob, p=p), Tsit5(), saveat=c.t,
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
        pred = Array(sol)[2, :]
        total += mean(abs2, c.T .- pred)
    end
    return total
end

# ---------------------------------------------------------------------
# Optimization stages
# ---------------------------------------------------------------------

# Stochastic mini-batch ADAM: each iteration optimizes a single
# condition's MSE, cycling round-robin through `conditions`.
function minibatch_adam(conditions, p0; lr, maxiters, log_every=100)
    n = length(conditions)
    counter = Ref(0)
    function loss_fn(p)
        counter[] += 1
        idx = ((counter[] - 1) % n) + 1
        return single_condition_mse(conditions[idx], p)
    end
    iter = Ref(0)
    function cb(state, l)
        if time_budget_exceeded()
            return true
        end
        iter[] += 1
        if iter[] % log_every == 0
            println("    mini-batch iter $(iter[])  loss=$(l)")
        end
        return false
    end
    optf = OptimizationFunction((x, _) -> loss_fn(x), Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(lr); maxiters=maxiters, callback=cb)
    return res.u
end

# ADAM on `loss_fn`, stopping early once the loss drops below `target`,
# or (if `window` is given) once it plateaus: the relative improvement
# over the last `window` iterations falls below `rel_tol`. `max_iters`
# is a hard safety cap in case neither condition is ever met.
function run_adam(loss_fn, p0; lr, target, max_iters, window=nothing, rel_tol=1e-4, log_every=100, label="ADAM")
    hist = Float64[]
    function cb(state, l)
        if time_budget_exceeded()
            return true
        end
        push!(hist, l)
        if length(hist) % log_every == 0
            println("    $label iter $(length(hist))  loss=$(l)")
        end
        if l < target
            return true
        end
        if window !== nothing && length(hist) >= window
            recent = @view hist[end-window+1:end]
            denom = max(abs(first(recent)), eps())
            improvement = (first(recent) - last(recent)) / denom
            if improvement < rel_tol
                return true
            end
        end
        return false
    end
    optf = OptimizationFunction((x, _) -> loss_fn(x), Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(lr); maxiters=max_iters, callback=cb)
    return res.u, res.objective
end

function bfgs_refine(loss_fn, p0; maxiters, log_every=10)
    iter = Ref(0)
    function cb(state, l)
        if time_budget_exceeded()
            return true
        end
        iter[] += 1
        if iter[] % log_every == 0
            println("    BFGS iter $(iter[])  loss=$(l)")
        end
        return false
    end
    optf = OptimizationFunction((x, _) -> loss_fn(x), Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0)
    res = Optimization.solve(optprob, BFGS(initial_stepnorm=0.01); maxiters=maxiters, callback=cb)
    return res.u, res.objective
end

# ---------------------------------------------------------------------
# Case-level training
# ---------------------------------------------------------------------

# Flat list of stages Case 1 works through: mini-batch once, then
# (ADAM, BFGS) pairs, one pair per round. A "stage index" into this list
# is what gets checkpointed, so a resumed run can jump straight back in.
function case1_stage_list()
    stages = Symbol[:minibatch]
    for _ in 1:CASE1_MAX_ROUNDS
        push!(stages, :adam)
        push!(stages, :bfgs)
    end
    return stages
end

# Runs Case 1's stage list starting at `start_stage_idx` (1 = from
# scratch). Returns (p, loss, converged). If the wall-clock budget runs
# out mid-stage, checkpoints p and returns converged=false with a
# checkpoint pointing back at the SAME stage (that stage type is simply
# re-entered from p on the next run -- ADAM/BFGS are both fine to resume
# from an arbitrary point). Checkpoints after every stage, not just at
# the end, so at most one stage's work is ever at risk from an unclean
# interruption (e.g. the runner being killed outright).
function train_case1(conditions, p0; start_stage_idx=1)
    stages = case1_stage_list()
    p = p0
    idx = clamp(start_stage_idx, 1, length(stages))
    loss_val = totalloss_normalized(conditions, p)

    while idx <= length(stages)
        stage = stages[idx]
        if stage == :minibatch
            println("Stage $(idx)/$(length(stages)): mini-batch ADAM ($(MINIBATCH_ITERS) iters cap, not normalized)")
            p = minibatch_adam(conditions, p; lr=CASE1_ADAM_LR, maxiters=MINIBATCH_ITERS)
        elseif stage == :adam
            println("Stage $(idx)/$(length(stages)): ADAM on total normalized loss until plateau (target=$(CASE1_TARGET))")
            p, _ = run_adam(pp -> totalloss_normalized(conditions, pp), p;
                             lr=CASE1_ADAM_LR, target=CASE1_TARGET, max_iters=CASE1_ADAM_MAX_ITERS,
                             window=CASE1_PLATEAU_WINDOW, rel_tol=CASE1_PLATEAU_RELTOL, label="ADAM")
        else # :bfgs
            println("Stage $(idx)/$(length(stages)): BFGS refine on total normalized loss")
            p, _ = bfgs_refine(pp -> totalloss_normalized(conditions, pp), p; maxiters=CASE1_BFGS_ITERS)
        end

        loss_val = totalloss_normalized(conditions, p)
        println("  -> after stage $(idx) ($(stage)): normalized total loss=$(loss_val)")
        converged = loss_val < CASE1_TARGET
        exhausted_time = time_budget_exceeded()

        next_idx = (exhausted_time && !converged) ? idx : idx + 1
        save_checkpoint(1, p, next_idx, converged)

        if converged
            return p, loss_val, true
        end
        if exhausted_time
            println("Wall-clock budget exhausted mid-stage $(idx); checkpoint saved to resume from the same stage.")
            return p, loss_val, false
        end
        idx = next_idx
    end

    println("Exhausted $(length(stages)) stages ($(CASE1_MAX_ROUNDS) rounds) without reaching target $(CASE1_TARGET).")
    return p, loss_val, false
end

# Single ADAM stage on the raw total loss. Resuming after a timeout or an
# unconverged max_iters cap just means calling this again starting from
# the checkpointed p -- there's no multi-stage state to track.
function train_case2(conditions, p0_warm)
    println("Case 2: ADAM on total (raw, not normalized) loss (warm-started from Case 1) until target=$(CASE2_TARGET)")
    p, loss_val = run_adam(pp -> totalloss_raw(conditions, pp), p0_warm;
                            lr=CASE2_ADAM_LR, target=CASE2_TARGET, max_iters=CASE2_ADAM_MAX_ITERS,
                            window=nothing, label="ADAM")
    converged = loss_val < CASE2_TARGET
    save_checkpoint(2, p, 1, converged)
    if !converged
        reason = time_budget_exceeded() ? "wall-clock budget exhausted" : "max_iters reached"
        println("Case 2 not yet converged ($(reason)); checkpoint saved to resume.")
    end
    return p, loss_val, converged
end

# Case 2 can only warm-start from a Case 1 that has actually finished
# (loss_val < CASE1_TARGET) -- there's no usable shipped fallback since
# the network architecture (2 hidden layers) doesn't match the shipped
# opt_para_case_1.jld2 (trained on the old 1-hidden-layer network).
function load_case1_params()
    case1_p, _, case1_done = load_checkpoint(1)
    if case1_done && case1_p !== nothing
        println("Warm-starting Case 2 from Case 1's completed checkpoint.")
        return case1_p
    end

    fresh_path = joinpath(@__DIR__, "results", "case1", "opt_para_case1_trained.jld2")
    if isfile(fresh_path)
        println("Warm-starting Case 2 from Case 1's saved results file.")
        return load(fresh_path)["opt_para_case1_trained"]
    end

    error("Case 1 has not finished training yet (no completed checkpoint or results file). " *
          "Run/resume `julia train.jl 1` until it converges before training Case 2.")
end

# ---------------------------------------------------------------------
# Evaluation / plotting
# ---------------------------------------------------------------------

function evaluate(conditions, p)
    metrics = NamedTuple[]
    for c in conditions
        sol = solve(remake(c.prob, p=p), Tsit5(), saveat=c.t)
        pred_C = Array(sol)[2, :] .- 273.15
        meas_C = c.T .- 273.15
        push!(metrics, (
            condition = c.label,
            rmse = rmse(meas_C, pred_C),
            mae = mae(meas_C, pred_C),
            r2 = r2_score(meas_C, pred_C),
        ))
    end
    return metrics
end

function plot_conditions(conditions, p, out_path, title_prefix)
    n = length(conditions)
    ncols = 2
    nrows = ceil(Int, n / ncols)
    P = plot(layout=(nrows, ncols), size=(900, 350 * nrows), dpi=300, framestyle=:box)
    for (i, c) in enumerate(conditions)
        sol = solve(remake(c.prob, p=p), Tsit5(), saveat=c.t)
        pred_C = Array(sol)[2, :] .- 273.15
        meas_C = c.T .- 273.15
        t_hours = (c.t .- c.t[1]) ./ 3600

        plot!(P[i], t_hours, meas_C, label="Measured", color=:black, linewidth=1.5)
        plot!(P[i], t_hours, pred_C, label="Predicted", color=:red, linewidth=1.5, linestyle=:dash)
        title!(P[i], "$(title_prefix): $(c.label)", titlefontsize=10)
        xlabel!(P[i], "Time (h)")
        ylabel!(P[i], "Temperature (°C)")
    end
    plot!(P, legend=:best, left_margin=5Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm)
    savefig(P, out_path)
end

# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

function train_case(case::Int)
    @assert case in (1, 2) "case must be 1 or 2"
    data_file = load_dataset(case)
    U, st = build_network()

    train_conditions = build_conditions(CASE_TRAIN_CONDITIONS[case], data_file, U, st)
    val_conditions = build_conditions(CASE_VAL_CONDITIONS[case], data_file, U, st)

    checkpoint_p, stage_idx, already_done = load_checkpoint(case)

    if already_done
        println("Case $case checkpoint is already marked done; re-evaluating/re-plotting from it.")
        p_opt = checkpoint_p
        converged = true
    elseif case == 1
        p0 = checkpoint_p !== nothing ? checkpoint_p : ComponentArray(Lux.setup(StableRNG(1111), U)[1])
        if checkpoint_p !== nothing
            println("Resuming Case 1 from checkpoint at stage $(stage_idx).")
        end
        p_opt, final_loss, converged = train_case1(train_conditions, p0; start_stage_idx=stage_idx)
        println("Case 1 loss after this run: $(final_loss)  (converged=$(converged))")
    else
        p0_warm = checkpoint_p !== nothing ? checkpoint_p : load_case1_params()
        if checkpoint_p !== nothing
            println("Resuming Case 2 from checkpoint.")
        end
        p_opt, final_loss, converged = train_case2(train_conditions, p0_warm)
        println("Case 2 loss after this run: $(final_loss)  (converged=$(converged))")
    end

    if !converged
        println("Case $case has not converged yet -- exiting so a follow-up run can resume from the checkpoint.")
        return nothing
    end

    out_dir = joinpath(@__DIR__, "results", "case$case")
    mkpath(out_dir)
    save(joinpath(out_dir, "opt_para_case$(case)_trained.jld2"), "opt_para_case$(case)_trained", p_opt)

    metrics = evaluate(val_conditions, p_opt)
    open(joinpath(out_dir, "metrics_case$case.csv"), "w") do io
        println(io, "condition,rmse_C,mae_C,r2")
        for m in metrics
            println(io, "$(m.condition),$(m.rmse),$(m.mae),$(m.r2)")
        end
    end

    println("\nValidation metrics for Case $case:")
    for m in metrics
        println("  $(m.condition): RMSE=$(round(m.rmse, digits=4))°C  MAE=$(round(m.mae, digits=4))°C  R²=$(round(m.r2, digits=4))")
    end
    mean_r2 = mean(m.r2 for m in metrics)
    println("  Mean validation R² = $(round(mean_r2, digits=4))  (acceptance criterion: R² > 0.80)")

    plot_conditions(val_conditions, p_opt, joinpath(out_dir, "validation_case$case.png"), "Case $case")

    return p_opt, metrics
end

function main()
    if length(ARGS) != 1 || !(ARGS[1] in ("1", "2"))
        println("Usage: julia train.jl <1|2>")
        exit(1)
    end
    train_case(parse(Int, ARGS[1]))
end

main()
