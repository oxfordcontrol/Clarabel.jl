# Warmstart comparisons

The four paper comparisons use three independent solver sequences on the same problem updates:

- Smoothing: `Clarabel.solve!(solver; warmstart=true)` (also `:smoothing`).
- Convex combination: `Clarabel.solve!(solver; warmstart=:convex_combination, warmstart_weight=0.01)`.
- Standard cold start: `Clarabel.solve!(solver)` (also `warmstart=:cold`).

For the convex baseline, `warmstart_weight` is the unit-start weight `t`:
`s⁰ = (1-t)s* + t s_unit`, and likewise for `z`. The combination is formed
in the solver's equilibrated coordinates. It uses `x⁰ = (1-t)x*`, `τ⁰ = 1`,
and `κ⁰ = t`, combining the previous optimum (`κ*=0`) with the cone unit
start (`x_unit=0`, `τ_unit=κ_unit=1`). Equality slacks remain zero and their
free dual variables are blended with zero. This is the fixed-weight
unit-start baseline, without additional adaptations of the cited algorithm.
If no previous solved solution is available, both warmstart methods use a
cold start. Weights must be finite and in `(0,1]`.

## Environment

Use a Julia environment that develops this local Clarabel checkout and has
`JuMP`, `MLDatasets`, `CSV`, `JLD2`, `Plots`, `StatsBase`, and `TimerOutputs`.
The perturbation script also requires the local `ClarabelBenchmarks` package.
For example, from the Clarabel.jl root, create a separate benchmark environment:

```bash
julia --project=warmstart_test -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.develop(path="../ClarabelBenchmarks"); Pkg.add(["JuMP", "MLDatasets", "CSV", "JLD2", "Plots", "StatsBase", "TimerOutputs", "Statistics"])'
```

To check the implementation and saved data on a tiny SOCP first:

```bash
julia --project=warmstart_test warmstart_test/smoke_comparison.jl
```

The smoke script writes its JLD2 data and LaTeX table to a temporary directory,
or to `CLARABEL_COMPARISON_OUTPUT` if set.

## Full experiments

Run from the repository root:

```bash
python3 warmstart_test/run_paper_benchmarks.py --cases svm_l1 svm_l2 rebalancing frontier
# Or run the experiment scripts individually:
julia --project=warmstart_test warmstart_test/svm_L1.jl
julia --project=warmstart_test warmstart_test/svm_L2_robust.jl
julia --project=warmstart_test warmstart_test/portfolioOpt/portfolioOpt.jl
julia --project=warmstart_test warmstart_test/portfolioOpt/efficientFrontier.jl
julia --project=warmstart_test warmstart_test/portfolioOpt/higherorderRiskMeansure.jl
CLARABEL_PERTURBATION=A julia --project=warmstart_test warmstart_test/varying_perturbation.jl
CLARABEL_PERTURBATION=b julia --project=warmstart_test warmstart_test/varying_perturbation.jl
CLARABEL_PERTURBATION=q julia --project=warmstart_test warmstart_test/varying_perturbation.jl
```

`CLARABEL_BENCHMARK_CLASS` selects the perturbation problem class (default `mpc`).
MNIST must be available to MLDatasets; the portfolio scripts use the included CSV.
The Python runner executes the selected cases serially, fixes Julia
and BLAS threading to one, and saves progress and logs in `results/run_logs`.
By default it runs all four paper comparisons. Use `--cases` to select a subset
of `svm_l1`, `svm_l2`, `rebalancing`, and `frontier`.
The covariance helpers normalize by `day-1`, matching the estimator in the paper.

The paper compares three methods only for the four SVM and second-order cone
portfolio experiments. It retains the original two-method HMCR results and
figure: the nonsymmetric power cones are outside the scope of the cited
convex-combination method. The standalone HMCR script compares smoothing with
cold starting only and
saves reruns as `portfolio_higherorderRisk_rerun.jld2`, preserving the original
`portfolio_higherorderRisk.jld2` data used in the draft.

Export the saved results and render publication
artifacts (the renderer requires Python with NumPy and Matplotlib):

```bash
julia --project=warmstart_test warmstart_test/export_paper_results.jl
python3 warmstart_test/render_paper_results.py --paper /path/to/warmstart-paper
```

The exporter reports solve counts and the largest objective difference between
methods. The renderer writes the three revision-marked LaTeX tables used by the draft
under `tables/`, the vector PDF for portfolio rebalancing under `figures/`,
and `tables/comparison_summary.md`. It generates only the comparison artifacts
used in the final draft.

## Outputs

The four current benchmark datasets in `warmstart_test/results` have a
`_comparison.jld2` suffix. The original two-method HMCR dataset and the three
MPC perturbation datasets are retained with their original filenames.
Superseded SVM and mean-variance datasets and unused generated tables and
figures have been removed. Tables contain Smooth / Convex / Cold
columns for both iterations and solve time. Times come from the full `solve!`
timer, including initialization. The solver copies are created before timing.
Both warmstart paths are warmed up on disposable solver copies before the
measured updates, leaving the three problem sequences unchanged.

JLD2 files retain `iter_w`, `iter_c`, `solve_time_w`, and `solve_time_c` and add
`iter_cc`, `solve_time_cc`, `convex_weight`, and per-method status arrays and
success counts. Failed solves are stored as time `Inf` and iteration count `-1`.
Geometric means use pairs with finite, positive values; inspect the status arrays
and success counts when comparing failure rates. Perturbation files contain
`geomean_iter_cc`, `geomean_time_cc`, `noise_levels`, and a `comparison_runs`
entry for each noise level containing all per-problem measurements.

Existing numerical tables in the paper are not updated automatically. Rerun
the experiments before reporting results from the changed SOC scaling or the
new convex-combination baseline.
