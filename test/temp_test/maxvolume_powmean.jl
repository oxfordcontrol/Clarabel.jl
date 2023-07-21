using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using JLD,JLD2
# include(".\\..\\src\\Clarabel.jl")
using Clarabel
using Hypatia
using BenchmarkTools

"""
Maximum volume hypercube} from Hypatia.jl,

https://github.com/chriscoey/Hypatia.jl/tree/master/examples/maxvolume,
"""

n = 500
# ensure there will be a feasible solution
x = randn(n)
# A = sparse(Symmetric(sprand(n,n,1.0/n)) + 10I)
A = sparse(1.0*I(n))
gamma = norm(A * x) / sqrt(n)
freq = ones(n)
freq ./= n

tol = 1e-4

#######################################################################
# YC: Benchmarking should be implemented separately if you want to 
#     obtain the plot.
#######################################################################


#Result from Clarabel's power mean cone
println("power mean cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
@constraint(model, vcat(x,t) in Clarabel.PowerMeanConeT(freq,n))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
MOI.set(model, MOI.Silent(), true)      #Diable printing information
optimize!(model)
clarabel_val = objective_value(model)


#Result from Hypatia
println("power mean cones via Hypatia")
model = Model(Hypatia.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
@constraint(model, vcat(t,x) in Hypatia.HypoPowerMeanCone(freq,false))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
MOI.set(model, MOI.Silent(), true)
optimize!(model)
# println(solution_summary(model))
hypatia_val = objective_value(model)
@assert isapprox(clarabel_val,hypatia_val,atol = tol)