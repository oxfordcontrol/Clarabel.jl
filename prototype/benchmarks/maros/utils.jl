using Revise
using MAT, JuMP
using LinearAlgebra
using Printf

Base.@kwdef struct TestResult
    time::Float64
    status::MOI.TerminationStatusCode
    cost::Float64
end

function dropinfs(A,b)

    b = b[:]
    finidx = findall(<(1e20), abs.(b))
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

function data_clarabel_form(vars)

    P,c,A,b = data_ecos_form(vars)
    cone_types = [IPSolver.NonnegativeConeT]
    cone_dims  = [length(b)]

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


function solve_ecos(vars)

    P,c,A,b = data_ecos_form(vars)
    model = Model(ECOS.Optimizer)
    @variable(model, x[1:length(c)])
    @constraint(model, c1, A*x .<= b)
    @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

    #Run the opimization
    set_optimizer_attribute(model, "verbose", 1)
    set_optimizer_attribute(model, "maxit", 50)
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
    set_optimizer_attribute(model, "verbose", 1)
    optimize!(model)


    time = JuMP.solve_time(model)
    cost = JuMP.objective_value(model)
    status = JuMP.termination_status(model)

    return TestResult(time,status,cost)

end


function solve_clarabel(vars)

    P,c,A,b,cone_types,cone_dims = data_clarabel_form(vars)
    settings = IPSolver.Settings(max_iter=100,direct_kkt_solver=true,verbose = true)
    solver   = IPSolver.Solver()
    IPSolver.setup!(solver,P,c,A,b,cone_types,cone_dims,settings)
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
