using Revise
using Clarabel
using LinearAlgebra
using Printf
using StatProfilerHTML
import Random
Random.seed!(242713)

function basic_QP_data_dualinf(Type::Type{T}) where {T <: AbstractFloat}

    #x = [1;-1] is in ker(P) and always feasible
    P = sparse(T[1. 1.;1. 1.])
    c = T[1; -1.]
    A = sparse(T[1. 1;1. 0;])
    b = [1.;1]
    cone_types = [Clarabel.NonnegativeConeT]
    cone_dims  = [2,]

    return (P,c,A,b,cone_types,cone_dims)
end

#solve in JuMP
using JuMP
using MosekTools, OSQP, ECOS

P,c,A,b,cone_types,cone_dims = basic_QP_data_dualinf(Float64)
P = P.*0
A = A[1:1,:]
b = b[1:1]

@printf("\n\nJuMP\n-------------------------\n\n")
model = Model(ECOS.Optimizer)
@variable(model, x[1:2])
@constraint(model, c1, A*x .<= b)
@objective(model, Min, sum(c.*x) + 1/2*x'*P*x)
#Run the opimization
optimize!(model)


@printf("\n\n-------------------------\n\n")
@printf("\nClarabel\n-------------------------\n\n")

settings = Clarabel.Settings(max_iter=10,direct_kkt_solver=true)
cone_types = [Clarabel.NonnegativeConeT]
cone_dims  = [1,]

solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims,settings)
Clarabel.solve!(solver)


data = solver.data
vars = solver.variables
res  = solver.residuals
x = vars.x
z = vars.z.vec
s = vars.s.vec
τ = vars.τ
κ = vars.κ
