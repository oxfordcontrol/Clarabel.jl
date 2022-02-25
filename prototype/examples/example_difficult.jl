using Revise
include("../IPSolver.jl")
using .IPSolver
using Random
using LinearAlgebra
using Printf
using StatProfilerHTML

function lasso_data(Type::Type{T} = Float64) where {T <: AbstractFloat}

    # generate problem data
    rng = Random.MersenneTwister(12345)
    n = 8
    m = 50 * n
    F = rand(rng, T, m, n)

    vtrue = sprand(rng, T, n, 1, 0.1)
    noise = T(0.1) * rand(rng, T, m, 1)
    b = F * vtrue + noise
    μMax = norm(F' * b, Inf)
    μ = T(0.1) * μMax

    # define lasso problem as SOCP
    A1 = -sparse([1 zeros(T, 1, 2 * n + 1) 1 zeros(T, 1, m);
    -1 zeros(T, 1, 2 * n) 1 zeros(T, 1, m + 1);
    zeros(T, m, 1) -2 * F zeros(T, m, n + 2) diagm( 0 => ones(T, m))])

    A2 = -sparse([zeros(T, n, 1) diagm(0 => ones(T, n)) -diagm(0 => ones(T, n)) zeros(T, n, m + 2);
    zeros(T, n, 1) -diagm(0 => ones(T, n)) -diagm(0 => ones(T, n)) zeros(T, n, m + 2)])
    A3 = -sparse([zeros(T, 1, 2 * n + 1) -one(T) zeros(T, 1, m + 1);
    zeros(T, 1, 2 * n + 2) -one(T) zeros(T, 1, m);
    zeros(T, m, 2 * n + 3) -diagm( 0 => ones(T, m))])
    b1 = T[1; 1; -2 * b[:]]
    b2 = zeros(T, 2 * n)
    b3 = zeros(T, m + 2)

    c = [one(T); zeros(T, n); μ * ones(T, n); zeros(T, m + 2)]
    P = spzeros(T, length(c), length(c))
    P = sparse(I(length(c)).*1e-6)

    A = [A1;A2;A3]
    b = [b1;b2;b3]

    cone_types = [IPSolver.NonnegativeConeT,
    IPSolver.NonnegativeConeT,
    IPSolver.SecondOrderConeT]
    cone_dims  = [length(b1),
    length(b2),
    length(b3)]

    return (P,c,A,b,cone_types,cone_dims,A1,A2,A3,b1,b2,b3)

end

P,c,A,b,cone_types,cone_dims,A1,A2,A3,b1,b2,b3 = lasso_data()
n = length(c)

#solve in JuMP
using JuMP
using MosekTools, OSQP, ECOS

@printf("\n\nJuMP\n-------------------------\n\n")
model = Model(ECOS.Optimizer)
@variable(model, x[1:n])
@constraint(model, c1, A1*x .<= b1)
@constraint(model, c2, A2*x .<= b2)
@constraint(model, c3, b3-A3*x in SecondOrderCone())
@objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

#Run the opimization
optimize!(model)

settings = IPSolver.Settings(max_iter=50,direct_kkt_solver=true)
solver   = IPSolver.Solver(P,c,A,b,cone_types,cone_dims,settings)
IPSolver.solve!(solver)

s = solver

data = s.data
vars = s.variables
res  = s.residuals
x = vars.x
z = vars.z.vec
s = vars.s.vec
τ = vars.τ
κ = vars.κ

nothing