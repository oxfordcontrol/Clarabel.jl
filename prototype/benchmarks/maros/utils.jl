
using MAT, JuMP, Gurobi
using LinearAlgebra
using .IPSolver
using Printf
using DelimitedFiles

Base.@kwdef mutable struct TestResult
    time::Float64
    status::MOI.TerminationStatusCode
    cost::Float64
end

function dropinfs(A,b)

    b = b[:]
    finidx = findall(<(5e19), abs.(b))
    b = b[finidx]
    A = A[finidx,:]
    return A,b

end

function data_osqp_form(vars)

    n = Int(vars["n"])
    m = Int(vars["m"])
    A   = vars["A"]
    P   = vars["P"]
    c   = vars["q"][:]
    c0  = vars["r"]
    l   = vars["l"][:]
    u   = vars["u"][:]

    #force a true double transpose
    #to ensure data is sorted within columns
    A = (A'.*1)'.*1
    P = (P'.*1)'.*1

    return P,c,A,l,u
end

function data_ecos_form(vars)

    P,c,A,l,u = data_osqp_form(vars)
    #make into single constraint
    A = [A; -A]
    b = [u;-l]
    A,b = dropinfs(A,b)

    return P,c,A,b

end

function data_ecos_form_eq(vars)

    P,c,A,l,u = data_osqp_form(vars)

    #separate into equalities and inequalities
    eqidx = l .== u
    Aeq = A[eqidx,:]
    beq = l[eqidx]

    #make into single constraint
    Aineq = A[.!eqidx,:]
    lineq = l[.!eqidx,:]
    uineq = u[.!eqidx,:]
    Aineq = [Aineq; -Aineq]
    bineq = [uineq;-lineq]
    Aineq,bineq = dropinfs(Aineq,bineq)

    return P,c,Aineq,bineq,Aeq,beq

end

function data_clarabel_form_eq(vars)

    P,c,Aineq,bineq,Aeq,beq = data_ecos_form_eq(vars)

    cone_types = Vector{IPSolver.SupportedCones}()
    cone_dims  = Vector{Int64}()
    if(length(beq) > 0)
        push!(cone_types,IPSolver.ZeroConeT)
        push!(cone_dims,length(beq))
    end
    if(length(bineq) > 0)
        push!(cone_types,IPSolver.NonnegativeConeT)
        push!(cone_dims,length(bineq))
    end

    A = [Aeq;Aineq]
    b = [beq;bineq]

    return P,c,A,b,cone_types,cone_dims
end



function data_clarabel_form(vars)

    P,c,A,b = data_ecos_form(vars)
    m = length(b)
    n = length(c)

    cone_types = [IPSolver.NonnegativeConeT]
    cone_dims  = [length(b)]

    return P,c,A,b,cone_types,cone_dims
end


function solve_ecos_eq(vars)

    P,c,Aineq,bineq,Aeq,beq = data_ecos_form_eq(vars)


    #see if cholesky fails.  If so, try to fix it
    try
        cholesky(P)
    catch
        P = P+1e-12*I
    end


    model = Model(ECOS.Optimizer)
    @variable(model, x[1:length(c)])
    @constraint(model, c1, Aineq*x .<= bineq)
    @constraint(model, c2, Aeq*x .== beq)
    @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

    #Run the opimization
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "maxit", 100)
    try
        optimize!(model)
        time = JuMP.solve_time(model)
        cost = JuMP.objective_value(model)
        status = JuMP.termination_status(model)
        return TestResult(time,status,cost)
    catch
        println("ECOS FAIL")
        time = NaN
        cost = NaN
        status = MOI.OTHER_ERROR
        return TestResult(time,status,cost)
    end

end

function solve_hypatia_eq(vars)

    P,c,Aineq,bineq,Aeq,beq = data_ecos_form_eq(vars)

    model = Model(Hypatia.Optimizer)
    @variable(model, x[1:length(c)])
    @constraint(model, c1, Aineq*x .<= bineq)
    @constraint(model, c2, Aeq*x .== beq)
    @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

    #Run the opimization
    set_optimizer_attribute(model, "verbose", 1)

    try
        optimize!(model)
        time = JuMP.solve_time(model)
        cost = JuMP.objective_value(model)
        status = JuMP.termination_status(model)
        return TestResult(time,status,cost)
    catch
        println("Hypatia FAIL")
        time = NaN
        cost = NaN
        status = MOI.OTHER_ERROR
        return TestResult(time,status,cost)
    end

end

function solve_gurobi_eq(vars)

    P,c,Aineq,bineq,Aeq,beq = data_ecos_form_eq(vars)

    model = Model(Gurobi.Optimizer)
    @variable(model, x[1:length(c)])
    @constraint(model, c1, Aineq*x .<= bineq)
    @constraint(model, c2, Aeq*x .== beq)
    @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

    #Run the opimization
    set_optimizer_attribute(model, "Presolve", 0)
    #set_optimizer_attribute(model, "Threads", 1)

    try
        optimize!(model)
        time = JuMP.solve_time(model)
        cost = JuMP.objective_value(model)
        status = JuMP.termination_status(model)
        return TestResult(time,status,cost)
    catch
        println("GUROBI FAIL")
        time = NaN
        cost = NaN
        status = MOI.OTHER_ERROR
        return TestResult(time,status,cost)
    end

end



function solve_ecos(vars)

    P,c,A,b = data_ecos_form(vars)


    #see if cholesky fails.  If so, try to fix it
    try
        cholesky(P)
    catch
        P = P+1e-12*I
    end

    model = Model(ECOS.Optimizer)
    @variable(model, x[1:length(c)])
    @constraint(model, c1, A*x .<= b)
    @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

    #Run the opimization
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "maxit", 100)



    try
        optimize!(model)
        time = JuMP.solve_time(model)
        cost = JuMP.objective_value(model)
        status = JuMP.termination_status(model)
        return TestResult(time,status,cost)
    catch
        println("ECOS HARD FAIL")
        time = NaN
        cost = NaN
        status = MOI.OTHER_ERROR
        return TestResult(time,status,cost)
    end

end

function solve_osqp(vars)

    #use ECOS style constraints.   Probably works
    #out to be the same since everything is just
    #converted to one sided and then back
    P,c,A,b = data_ecos_form(vars)

    model = Model(OSQP.Optimizer)
    @variable(model, x[1:length(c)])
    @constraint(model, c1, A*x .<= b)
    @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

    #Run the opimization
    set_optimizer_attribute(model, "verbose", 0)
    optimize!(model)


    time = JuMP.solve_time(model)
    cost = JuMP.objective_value(model)
    status = JuMP.termination_status(model)

    return TestResult(time,status,cost)

end

function solve_clarabel_eq(vars)

    P,c,A,b,cone_types,cone_dims = data_clarabel_form_eq(vars)
    settings = IPSolver.Settings(
            max_iter=100,
            direct_kkt_solver=true,
            #static_regularization_eps = 1e-7,
            dynamic_regularization_delta = 2e-7,
            dynamic_regularization_eps = 1e-13,
            verbose = true,
            equilibrate_enable = true,
    )
    solver   = IPSolver.Solver(P,c,A,b,cone_types,cone_dims,settings)
    IPSolver.solve!(solver)

    time = solver.info.solve_time
    cost = solver.info.cost_primal

    if(any(isnan.(solver.variables.x)))
        status = MOI.OTHER_ERROR
    elseif(solver.info.status == IPSolver.SOLVED)
        status = MOI.OPTIMAL
    elseif(solver.info.status == IPSolver.PRIMAL_INFEASIBLE)
        status = MOI.INFEASIBLE
    elseif(solver.info.status == IPSolver.DUAL_INFEASIBLE)
        status = MOI.DUAL_INFEASIBLE
    elseif(solver.info.status == IPSolver.MAX_ITERATIONS)
        status = MOI.ITERATION_LIMIT
    else
        status = MOI.OTHER_ERROR
    end

    return TestResult(time,status,cost), solver

end


function solve_clarabel(vars)

    P,c,A,b,cone_types,cone_dims = data_clarabel_form(vars)
    settings = IPSolver.Settings(
            max_iter=100,
            direct_kkt_solver=true,
            #static_regularization_eps = 1e-7,
            dynamic_regularization_delta = 2e-7,
            dynamic_regularization_eps = 1e-13,
            verbose = false,
            equilibrate_enable = true,
    )
    solver   = IPSolver.Solver(P,c,A,b,cone_types,cone_dims,settings)
    IPSolver.solve!(solver)

    time = solver.info.solve_time
    cost = solver.info.cost_primal

    if(any(isnan.(solver.variables.x)))
        status = MOI.OTHER_ERROR
    elseif(solver.info.status == IPSolver.SOLVED)
        status = MOI.OPTIMAL
    elseif(solver.info.status == IPSolver.PRIMAL_INFEASIBLE)
        status = MOI.INFEASIBLE
    elseif(solver.info.status == IPSolver.DUAL_INFEASIBLE)
        status = MOI.DUAL_INFEASIBLE
    elseif(solver.info.status == IPSolver.MAX_ITERATIONS)
        status = MOI.ITERATION_LIMIT
    else
        status = MOI.OTHER_ERROR
    end

    return TestResult(time,status,cost), solver

end

function get_ref_solutions(filename)

    data = readdlm(filename;header=true)[1]
    #drops second header element.  Last column is objective
    out = Dict{String,Float64}()

    for i = 1:size(data,1)
        key = uppercase(data[i,1])
        obj = data[i,end]
        out[key] = obj
    end

    return out

end
