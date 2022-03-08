using Revise
include("../Clarabel.jl")
using .Clarabel
using LinearAlgebra
using Printf
using StatProfilerHTML

A = SparseMatrixCSC(I(3)*1.)
P = SparseMatrixCSC(I(3)*1.)
#P[1,1] = 25
P[2,2] = 2.; P[3,3] = 3.
A = [A;-A].*2
A[1,1] = 100
A[end,end] = 100
c = [3.;-2.;1.]*10
b = ones(Float64,6)
cone_types = [Clarabel.NonnegativeConeT, Clarabel.NonnegativeConeT]
cone_dims  = [3,3]

# #add an equality constraint
# a = [1 1 -2]
# A = [a;A]
# b = [pi/2;b]
# cone_types = [Clarabel.ZeroConeT, Clarabel.NonnegativeConeT, Clarabel.NonnegativeConeT]
# cone_dims  = [1,3,3]
#
# #primal infeasible variation
# b[1] = -1
# b[4] = -1

# # dual infeasible variation
# P = SparseMatrixCSC(I(3)*1.)
# P[1,1] = 0
# A = SparseMatrixCSC(I(3)*1.)
# A = [A;-A]
# A[4,1] = 1      #swap lower bound on first variable to redundant upper
# c = [1.;0;0]
# b = ones(Float64,6)
# cone_types = [Clarabel.NonnegativeConeT, Clarabel.NonnegativeConeT]
# cone_dims  = [3,3]

#solve in JuMP
using JuMP
using MosekTools, OSQP, ECOS

@printf("\n\nJuMP\n-------------------------\n\n")
model = Model(ECOS.Optimizer)
@variable(model, Jx[1:3])
@constraint(model, c1, A*Jx .<= b)
@objective(model, Min, sum(c.*Jx) + 1/2*Jx'*P*Jx)

#Run the opimization
optimize!(model)
Jx = JuMP.value.(Jx)
Js = b - A*Jx
Jz = -JuMP.dual.(c1)

#
# @printf("\n\n-------------------------\n\n")
@printf("\n\n-------------------------\n\n")
@printf("\nClarabel (Direct)\n-------------------------\n\n")

settings = Clarabel.Settings(max_iter=20,verbose=false,direct_kkt_solver=true,equilibrate_enable = true)
solver   = Clarabel.Solver()
Clarabel.setup!(solver,P,c,A,b,cone_types,cone_dims,settings)
Clarabel.solve!(solver)
