using ECOS, Mosek, MosekTools
using JuMP, MathOptInterface
# const MOI = MathOptInterface
using LinearAlgebra
using ConicBenchmarkUtilities

using Profile,StatProfilerHTML, TimerOutputs

include("../src\\Clarabel.jl")
# using Clarabel
# using Hypatia

coneMap = Dict(:Zero => MOI.Zeros, :Free => :Free,
                     :NonPos => MOI.Nonpositives, :NonNeg => MOI.Nonnegatives,
                     :SOC => MOI.SecondOrderCone, :SOCRotated => MOI.RotatedSecondOrderCone,
                     :ExpPrimal => MOI.ExponentialCone, :ExpDual => MOI.DualExponentialCone)

filelist = readdir(pwd()*"./primal_exp_cbf")

# dat = readcbfdata("./exp_cbf/car.cbf.gz") # .cbf.gz extension also accepted

for j = 1:32    #length(filelist)
    println("Current file is ", j)
    datadir = filelist[j]   #"gp_dave_1.cbf.gz"
    dat = readcbfdata("./primal_exp_cbf/"*datadir) # .cbf.gz extension also accepted

    println("Current file is: ", datadir)

    # In MathProgBase format:
    c, A, b, con_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat)
    # Note: The sense in MathProgBase form is always minimization, and the objective offset is zero.
    # If sense == :Max, you should flip the sign of c before handing off to a solver.
    if sense == :Max
        c .*= -1
    end

    num_con = size(A,1)
    num_var = size(A,2)

    model = Model(Clarabel.Optimizer)
    set_optimizer_attribute(model, "direct_solve_method", :qdldl)
    # set_optimizer_attribute(model, "static_regularization_eps", 1e-7)
    set_optimizer_attribute(model, "tol_gap_abs", 1e-8)
    set_optimizer_attribute(model, "tol_gap_rel", 1e-8)
    set_optimizer_attribute(model, "tol_infeas_abs", 1e-8)
    set_optimizer_attribute(model, "tol_infeas_rel", 1e-8)
    # set_optimizer_attribute(model, "proportional_eps", Float64(1e-16))
    
    # model = Model(Hypatia.Optimizer)
    # model = Model(ECOS.Optimizer)
    
    # model = Model(Mosek.Optimizer)
    # set_optimizer_attribute(model, "MSK_IPAR_PRESOLVE_USE", MSK_PRESOLVE_MODE_OFF)

    # TimerOutputs.enable_debug_timings(Clarabel)

    @variable(model, x[1:num_var])

    #Tackling constraint
    for i = 1:length(con_cones)
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

    for i = 1:length(var_cones)
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
    
    optimize!(model)
end
