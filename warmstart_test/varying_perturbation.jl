using Random, StatsBase, LinearAlgebra, SparseArrays
using Clarabel, JuMP, ClarabelBenchmarks, JLD2, TimerOutputs
include("save_table.jl")

class = get(ENV,"CLARABEL_BENCHMARK_CLASS","mpc")
perturbation = Symbol(get(ENV,"CLARABEL_PERTURBATION","A"))
perturbation in (:A,:b,:q) || error("CLARABEL_PERTURBATION must be A, b, or q")
problems = collect(keys(ClarabelBenchmarks.PROBLEMS[class]))
nsample = length(problems)
noise_levels = 10.0.^(range(-3,0,length=31))
geomean_iter = fill(NaN,length(noise_levels))
geomean_time = fill(NaN,length(noise_levels))
geomean_iter_cc = fill(NaN,length(noise_levels))
geomean_time_cc = fill(NaN,length(noise_levels))
comparison_runs = Vector{Dict}(undef,length(noise_levels))

for (k,noise) in enumerate(noise_levels)
    tables = Dict{Symbol,Any}(:problems=>problems,:varVal=>collect(1:nsample),
        :warm_time=>fill(Inf,nsample),:cold_time=>fill(Inf,nsample),
        :warm_iterations=>fill(-1,nsample),:cold_iterations=>fill(-1,nsample))
    init_convex_comparison!(tables,nsample; weight=0.01)
    for (i,example) in enumerate(problems)
        println("Current problem ",i," is ",example,"; noise = ",noise)
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model,"equilibrate_enable",false)
        set_optimizer_attribute(model,"presolve_enable",false)
        set_optimizer_attribute(model,"chordal_decomposition_enable",false)
        ClarabelBenchmarks.PROBLEMS[class][example](model)
        solver = model.moi_backend.optimizer.model.optimizer.solver
        solver_c,solver_cc = deepcopy(solver),deepcopy(solver)
        warmup_comparison!(solver; weight=0.01)

        # All methods receive the same perturbation and original solved problem.
        seed = Random.MersenneTwister(1000+i)
        updated = deepcopy(getproperty(solver.data,perturbation))
        values = perturbation == :A ? updated.nzval : updated
        num_noise = Int(min(ceil(0.1*length(values)),20))
        indices = StatsBase.sample(seed,eachindex(values),num_noise; replace=false)
        signs = 1 .- 2*rand(seed,num_noise)
        for (j,index) in enumerate(indices)
            values[index] = abs(values[index]) > 1e-6 ?
                (1+noise*signs[j])*values[index] : noise*signs[j]
        end
        update! = perturbation == :A ? Clarabel.update_A! :
            perturbation == :b ? Clarabel.update_b! : Clarabel.update_q!

        update!(solver,updated)
        Clarabel.solve!(solver; warmstart=true)
        record_comparison_result!(tables,:warm,solver,i)
        sleep(0.01)

        update!(solver_cc,updated)
        solve_convex_comparison!(solver_cc,tables,i)
        sleep(0.01)

        update!(solver_c,updated)
        Clarabel.solve!(solver_c)
        record_comparison_result!(tables,:cold,solver_c,i)
        sleep(0.01)
    end
    geomean_iter[k] = compute_geomean(tables[:warm_iterations],tables[:cold_iterations])
    geomean_time[k] = compute_geomean(tables[:warm_time],tables[:cold_time])
    geomean_iter_cc[k] = compute_geomean(tables[:convex_iterations],tables[:cold_iterations])
    geomean_time_cc[k] = compute_geomean(tables[:convex_time],tables[:cold_time])
    comparison_runs[k] = comparison_data(tables)
end

# Separate files preserve the historical datasets and distinguish A/b/q runs.
output = joinpath(@__DIR__,"results",class*"_varying_disturbance_"*string(perturbation)*"_comparison.jld2")
mkpath(dirname(output))
JLD2.jldsave(output; geomean_iter,geomean_time,geomean_iter_cc,geomean_time_cc,
            comparison_runs,convex_weight=0.01,noise_levels,
            methods=["smoothing","convex_combination","cold"])
