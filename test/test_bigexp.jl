using ECOS, Mosek, MosekTools
using JuMP, MathOptInterface
# const MOI = MathOptInterface
using LinearAlgebra
using ConicBenchmarkUtilities
using Profile
using TimerOutputs
using Printf, StatsBase

#include("../src\\Clarabel.jl")
using Clarabel
# using Hypatia

coneMap = Dict(:Zero => MOI.Zeros, :Free => :Free,
                     :NonPos => MOI.Nonpositives, :NonNeg => MOI.Nonnegatives,
                     :SOC => MOI.SecondOrderCone, :SOCRotated => MOI.RotatedSecondOrderCone,
                     :ExpPrimal => MOI.ExponentialCone, :ExpDual => MOI.DualExponentialCone)

function exp_model(exInd::Int; optimizer = Clarabel.Optimizer)

    cbfpath  = joinpath(@__DIR__,"primal_exp_cbf")
    filelist = readdir(cbfpath)

    datadir = filelist[exInd]   #"gp_dave_1.cbf.gz"
    dat = readcbfdata(joinpath(cbfpath,datadir)) # .cbf.gz extension also accepted

    # In MathProgBase format:
    c, A, b, con_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat)
    # Note: The sense in MathProgBase form is always minimization, and the objective offset is zero.
    # If sense == :Max, you should flip the sign of c before handing off to a solver.
    if sense == :Max
        c .*= -1
    end

    #println("con_cones = ", con_cones)
    #println("var_cones = ", var_cones)

    num_con = size(A,1)
    num_var = size(A,2)

    model = Model(optimizer)
    set_optimizer_attribute(model, "direct_solve_method", :qdldl)

    # model = Model(ECOS.Optimizer)
    @variable(model, x[1:num_var])

    #Tackling constraint
    for i in eachindex(con_cones)
        cur_cone = con_cones[i]
        # println(coneMap[cur_cone[1]])

        if coneMap[cur_cone[1]] == :Free
            continue
        elseif coneMap[cur_cone[1]] == MOI.ExponentialCone
            @constraint(model, b[cur_cone[2]] - A[cur_cone[2],:]*x in MOI.ExponentialCone())
        # elseif coneMap[cur_cone[1]] == MOI.DualExponentialCone
        #     @constraint(model, b[cur_cone[2]] - A[cur_cone[2],:]*x in MOI.DualExponentialCone())
        else
            @constraint(model, b[cur_cone[2]] - A[cur_cone[2],:]*x in coneMap[cur_cone[1]](length(cur_cone[2])))
        end
    end

    for i in eachindex(var_cones)
        cur_var = var_cones[i]
        # println(coneMap[cur_var[1]])

        if coneMap[cur_var[1]] == :Free
            continue
        elseif coneMap[cur_var[1]] == MOI.ExponentialCone
            @constraint(model, x[cur_var[2]] in MOI.ExponentialCone())
        # elseif coneMap[cur_var[1]] == MOI.DualExponentialCone
        #     @constraint(model, x[cur_var[2]] in MOI.DualExponentialCone())
        else
            @constraint(model, x[cur_var[2]] in coneMap[cur_var[1]](length(cur_var[2])))
        end
    end

    @objective(model, Min, sum(c.*x))

    return model
end

# the input number i corresponds to the i-th example in CBLIB. Example 7,8,32
# 4 is very small, also 12,13

function run(index)

    verbosity = true        
    maxiter     = 100

    model_clarabel = exp_model(index; optimizer = Clarabel.Optimizer) 
    set_optimizer_attribute(model_clarabel, "verbose", verbosity)
    set_optimizer_attribute(model_clarabel, "max_iter", maxiter)
    set_optimizer_attribute(model_clarabel, "equilibrate_enable", true)
    set_optimizer_attribute(model_clarabel, "static_regularization_constant",1e-7)
    set_optimizer_attribute(model_clarabel, "static_regularization_proportional",eps()^(2))  #disables it?
    set_optimizer_attribute(model_clarabel, "linesearch_backtrack_step",0.80)  #matches ECOS
    set_optimizer_attribute(model_clarabel, "max_step_fraction",0.99);  #default 0.99
    set_optimizer_attribute(model_clarabel, "min_primaldual_step_length", 0.2)
    set_optimizer_attribute(model_clarabel, "static_regularization_enable",true)
    set_optimizer_attribute(model_clarabel, "direct_solve_method",:qdldl)
    set_optimizer_attribute(model_clarabel, "iterative_refinement_reltol",1e-8 )   #default 1e-8
    set_optimizer_attribute(model_clarabel, "iterative_refinement_abstol",1e-10)  #default 1e-10
    optimize!(model_clarabel) 

    #model_ecos = exp_model(index; optimizer = ECOS.Optimizer) 
    #set_optimizer_attribute(model_ecos, "verbose", verbosity)
    #set_optimizer_attribute(model_ecos, "maxit", maxiter)
    #optimize!(model_ecos) 

    #println(solution_summary(model_clarabel))
    #println(solution_summary(model_ecos))

    solver = model_clarabel.moi_backend.optimizer.model.optimizer.inner
    return model_clarabel  #model_ecos
end

function run_all()

    status_c = []
    status_e = [] 
    for i = 1:32
        model_c = run(i)
        push!(status_c,solution_summary(model_c))
        #push!(status_e,solution_summary(model_e))
        @printf("%i ",i)
    end
    println()

    for i = 1:length(status_c)
        @printf("%i:  Clarabel: status %s.\t Iterations: %i. \t time: %e\n", 
        i, status_c[i].termination_status,status_c[i].barrier_iterations,status_c[i].solve_time)
       #@printf("%i:  ECOS    : status %s.\t Iterations: %i. \t time: %e\n", 
       #i, status_e[i].termination_status,status_e[i].barrier_iterations,status_e[i].solve_time)
       #println()
    end
    println("Clarabel iterations : ", sum(i->status_c[i].barrier_iterations,1:32))
    println("Clarabel time       : ", StatsBase.geomean(map(i->status_c[i].solve_time,1:32)))
    #println("ECOS iterations     : ", sum(i->status_e[i].barrier_iterations,1:32))
    #println("ECOS time           : ", StatsBase.geomean(map(i->status_e[i].solve_time,1:32)))

end

#bad problems 
# 21 : infeasible.   Hits iteration limit?  Works with smaller regularization.
# 19,32 fails.   Bad pivots, but nearly(?) solved.  ECOS solves both.
# maybe need to switch to dual scaling sooner?


# @enter Clarabel.solve!(solver)

# pprof()



# Q  = (solver.kktsystem.kktsolver.ldlsolver.factors.workspace.triuA) + Diagonal(solver.kktsystem.kktsolver.ldlsolver.factors.workspace.Dsigns).*7e-8
# (rows,cols) = findnz(Q)
# [rows cols Q.nzval];

nothing