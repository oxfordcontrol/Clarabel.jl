using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using JLD,JLD2
# include(".\\..\\src\\Clarabel.jl")
using Clarabel
using Hypatia

"""
Discrete maximum likelihood from Hypatia.jl,

https://github.com/chriscoey/Hypatia.jl/tree/master/examples/discretemaxlikelihood,
"""
d = 500
freq = Float64.(rand(1:(2 * d), d))
freq ./= sum(freq)      # normalize the sum to be 1

#Result from Clarabel's 3x3 power cones
println("3x3 cones")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model,q[1:d-1])
@objective(model, Min, -q[end])
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
@objective(model, Min, -t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(p,t) in Clarabel.GenPowerConeT(freq,d,1))
optimize!(model)

#Result from Mosek
println("3x3 cones by Mosek")
model = Model(Mosek.Optimizer)
@variable(model, p[1:d])
@variable(model,q[1:d-1])
@variable(model,r[1:d-2])
@objective(model, Min, -q[end])
@constraint(model, sum(p) == 1)
# trnasform a general power cone into a product of 3x3 power cones
power = freq[1] + freq[2]
@constraint(model, vcat(p[2],p[1],q[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:d-2
    global power += freq[i+2]
    @constraint(model, r[i] == q[i])
    @constraint(model, vcat(p[i+2],r[i],q[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
optimize!(model)

#Result from Hypatia
println("generalized power cones via Hypatia")
model = Model(Hypatia.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Min, -t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(p,t) in Hypatia.GeneralizedPowerCone(freq,1,false))
optimize!(model)