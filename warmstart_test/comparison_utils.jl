using Statistics, TimerOutputs, LinearAlgebra

# Compile both warmstart paths on disposable copies before measuring updates.
function warmup_comparison!(solver; weight=0.01)
    solver.solution.status in (Clarabel.SOLVED,Clarabel.ALMOST_SOLVED) || return nothing
    for method in (true,:convex_combination)
        scratch = deepcopy(solver)
        scratch.settings.verbose = false
        Clarabel.solve!(scratch; warmstart=method,warmstart_weight=weight)
    end
    return nothing
end

# Keep the existing smoothing/cold data keys and add an independent baseline.
function init_convex_comparison!(tables,nsample; weight=0.01)
    tables[:convex_weight] = weight
    tables[:convex_time] = fill(Inf,nsample)
    tables[:convex_iterations] = fill(-1,nsample)
    for method in (:warm,:convex,:cold)
        tables[Symbol(method,:_status)] = fill("NOT_RUN",nsample)
        for diagnostic in (:objective,:primal_residual,:dual_residual,:gap_absolute,:gap_relative)
            tables[Symbol(method,:_,diagnostic)] = fill(NaN,nsample)
        end
    end
    return tables
end

function record_comparison_result!(tables,method,solver,i)
    solved = solver.solution.status == Clarabel.SOLVED
    tables[Symbol(method,:_time)][i] = solved ?
        TimerOutputs.tottime(solver.timers["solve!"])/1e9 : Inf
    tables[Symbol(method,:_iterations)][i] = solved ? solver.info.iterations : -1
    tables[Symbol(method,:_status)][i] = string(solver.solution.status)
    tables[Symbol(method,:_objective)][i] = solver.solution.obj_val
    tables[Symbol(method,:_primal_residual)][i] = solver.info.res_primal
    tables[Symbol(method,:_dual_residual)][i] = solver.info.res_dual
    tables[Symbol(method,:_gap_absolute)][i] = solver.info.gap_abs
    tables[Symbol(method,:_gap_relative)][i] = solver.info.gap_rel
    println("Comparison ",i," ",method,": ",solver.solution.status,
            ", iterations=",solver.info.iterations,", time=",
            tables[Symbol(method,:_time)][i])
    flush(stdout)
    if method == :cold && haskey(tables,:output_prefix)
        save_comparison(tables,tables[:output_prefix])
    end
    return nothing
end

function solve_convex_comparison!(solver,tables,i)
    Clarabel.solve!(solver; warmstart=:convex_combination,
                   warmstart_weight=tables[:convex_weight])
    record_comparison_result!(tables,:convex,solver,i)
    return nothing
end

function compute_geomean(warm,cold)
    valid = isfinite.(warm) .& isfinite.(cold) .& (warm .> 0) .& (cold .> 0)
    any(valid) || return NaN
    return exp(mean(log.(warm[valid]./cold[valid])))
end

function comparison_data(tables)
    data = Dict{String,Any}(string(key)=>value for (key,value) in tables)
    data["julia_version"] = string(VERSION)
    data["cpu"] = Sys.CPU_NAME
    data["blas_threads"] = BLAS.get_num_threads()
    # Preserve the variable names used by the existing result readers.
    for (prefix,method) in (("w",:warm),("cc",:convex),("c",:cold))
        data["iter_"*prefix] = tables[Symbol(method,:_iterations)]
        data["solve_time_"*prefix] = tables[Symbol(method,:_time)]
        data[string(method)*"_success_count"] = count(isfinite,tables[Symbol(method,:_time)])
    end
    for method in (:warm,:convex)
        data[string(method)*"_geomean_iter"] =
            compute_geomean(tables[Symbol(method,:_iterations)],tables[:cold_iterations])
        data[string(method)*"_geomean_time"] =
            compute_geomean(tables[Symbol(method,:_time)],tables[:cold_time])
    end
    data["methods"] = ["smoothing","convex_combination","cold"]
    return data
end

# Callers import JLD2. Use separate files so historical results remain intact.
function save_comparison(tables,filename)
    path = filename*"_comparison.jld2"
    mkpath(dirname(path))
    JLD2.jldopen(path,"w") do file
        for (key,value) in comparison_data(tables)
            file[key] = value
        end
    end
    return path
end
