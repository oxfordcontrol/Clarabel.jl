using ECOS, Mosek, MosekTools
using JuMP, MathOptInterface
# const MOI = MathOptInterface
using LinearAlgebra
using ConicBenchmarkUtilities
using Profile
using TimerOutputs

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

index = 7

model_clarabel = exp_model(index; optimizer = Clarabel.Optimizer) 
model_ecos = exp_model(index; optimizer = ECOS.Optimizer) 
set_optimizer_attribute(model_clarabel, "verbose", true)
set_optimizer_attribute(model_clarabel, "static_regularization_constant",1e-7)
set_optimizer_attribute(model_clarabel, "static_regularization_enable",true)
set_optimizer_attribute(model_clarabel, "direct_solve_method",:qdldl)
set_optimizer_attribute(model_ecos, "verbose", true)
optimize!(model_clarabel) 
optimize!(model_ecos) 

println(solution_summary(model_clarabel))
println(solution_summary(model_ecos))

solver = model_clarabel.moi_backend.optimizer.model.optimizer.inner

#@profile Clarabel.solve!(solver)

#pprof()