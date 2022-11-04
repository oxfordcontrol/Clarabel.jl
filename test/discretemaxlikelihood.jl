using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using JLD,JLD2
# include(".\\..\\src\\Clarabel.jl")
using Clarabel

"""
Discrete maximum likelihood from Hypatia.jl,

https://github.com/chriscoey/Hypatia.jl/tree/master/examples/discretemaxlikelihood,
"""
d = 10000
freq = Float64.(rand(1:(2 * d), d))
freq ./= sum(freq)      # normalize the sum to be 1

#Result from Clarabel's 3x3 power cones
println("3x3 cones")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model,q[1:d-1])
@objective(model, Max, q[end])
@constraint(model, sum(p) == 1)
# trnasform a general power cone into a product of 3x3 power cones
power = freq[1] + freq[2]
@constraint(model, vcat(p[2],p[1],q[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:d-2
    global power += freq[i+2]
    @constraint(model, vcat(p[i+2],q[i],q[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
optimize!(model)

#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Max, t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(p,t) in Clarabel.GenPowerConeT(freq,d,1))
optimize!(model)

#Result from Hypatia
println("generalized power cones via Hypatia")
using Hypatia
model = Model(Hypatia.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Max, t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(p,t) in Hypatia.GeneralizedPowerCone(freq,1,false))
optimize!(model)