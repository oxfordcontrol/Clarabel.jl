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

#Result from Mosek
println("Three-dimensional cones via Mosek")
model = Model(Mosek.Optimizer)
@variable(model, p[1:n])
@variable(model,q[1:n-1])
@variable(model,r[1:n-2])
@objective(model, Max, q[end])
# trnasform a general power cone into a product of 3x3 power cones
power = freq[1] + freq[2]
@constraint(model, vcat(p[2],p[1],q[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:n-2
    global power += freq[i+2]
    @constraint(model, r[i] == q[i])
    @constraint(model, vcat(p[i+2],r[i],q[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
@constraint(model, vcat(gamma, A * p) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * p) in MOI.NormOneCone(n + 1))
MOI.set(model, MOI.Silent(), true)
optimize!(model)
opt_val = objective_value(model)

#Result from Clarabel's generalized power cone
println("Three-dimensional cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@variable(model,z[1:n-1])
@objective(model, Max, z[end])
# trnasform a general power cone into a product of three-dimensional power cones
power = freq[1] + freq[2]
@constraint(model, vcat(x[2],x[1],z[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:n-2
    global power += freq[i+2]
    @constraint(model, vcat(x[i+2],z[i],z[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
# @constraint(model, vcat(gamma, A * x) in MOI.SecondOrderCone(n + 1))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
MOI.set(model, MOI.Silent(), true)      #Diable printing information
optimize!(model)
@assert isapprox(opt_val,objective_value(model),atol = tol)

#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
@constraint(model, vcat(x,t) in Clarabel.GenPowerConeT(freq,n,1))
# @constraint(model, vcat(gamma, A * x) in MOI.SecondOrderCone(n + 1))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
MOI.set(model, MOI.Silent(), true)      #Diable printing information
optimize!(model)
@assert isapprox(opt_val,objective_value(model),atol = tol)

#Result from Hypatia
println("generalized power cones via Hypatia")
model = Model(Hypatia.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
@constraint(model, vcat(x,t) in Hypatia.GeneralizedPowerCone(freq,1,false))
# @constraint(model, vcat(gamma, A * x) in MOI.SecondOrderCone(n + 1))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
MOI.set(model, MOI.Silent(), true)
optimize!(model)
@assert isapprox(opt_val,objective_value(model),atol = tol)