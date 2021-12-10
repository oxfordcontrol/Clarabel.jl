using Revise
include("../IPSolver.jl")
using .IPSolver
using LinearAlgebra
using Printf
using ClearStacktrace
using StatProfilerHTML

A = SparseMatrixCSC(I(3)*1.)
A = [A;-A].*2
c = [3.;-2.;1.]
b = ones(Float64,6)
cone_types = [IPSolver.NonnegativeConeT, IPSolver.NonnegativeConeT]
cone_dims  = [3,3]

# #add an equality constraint
# a = [1 1 -2]
# A = [a;A]
# b = [pi/2;b]
# cone_types = [IPSolver.ZeroConeT, IPSolver.NonnegativeConeT, IPSolver.NonnegativeConeT]
# cone_dims  = [1,3,3]

# #primal infeasible variation
# b[1] = -1
# b[4] = -1

# #dual infeasible variation
# A = SparseMatrixCSC(I(3)*1.)
# A[1,1] = eps(Float64)
# A = [A;-A]
# c = [1.;0;0]
# b = ones(Float64,6)
# cone_types = [IPSolver.NonnegativeConeT, IPSolver.NonnegativeConeT]
# cone_dims  = [3,3]

#solve in JuMP
using JuMP
using MosekTools, OSQP, ECOS

@printf("\n\nJuMP\n-------------------------\n\n")
model = Model(ECOS.Optimizer)
@variable(model, x[1:3])
@constraint(model, c1, A*x .<= b)
@objective(model, Min, sum(c.*x))

#Run the opimization
optimize!(model)
print(JuMP.value.(x))
#
# @printf("\n\n-------------------------\n\n")
@printf("\n\n-------------------------\n\n")
@printf("\nClarabel (Direct)\n-------------------------\n\n")

settings = IPSolver.Settings(max_iter=20,verbose=true,direct_kkt_solver=true)
solver   = IPSolver.Solver()
IPSolver.setup!(solver,c,A,b,cone_types,cone_dims,settings)
IPSolver.solve!(solver)

@printf("\nClarabel (Indirect)\n-------------------------\n\n")

settings = IPSolver.Settings(max_iter=20,verbose=true,direct_kkt_solver=false)
solver   = IPSolver.Solver()
IPSolver.setup!(solver,c,A,b,cone_types,cone_dims,settings)
IPSolver.solve!(solver)
