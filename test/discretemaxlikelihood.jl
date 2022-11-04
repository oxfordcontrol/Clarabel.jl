using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using JLD,JLD2
# include(".\\..\\src\\Clarabel.jl")
using Clarabel

d = 10000
freq = Float64.(rand(1:(2 * d), d))
freq ./= sum(freq)

freq = ones(Float64,d)/d    #corresponds to GeometricMeanCone

println("\n\nJuMP\n-------------------------\n\n")
println("3x3 cones")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Max, t)
@constraint(model, sum(p) == 1)
# @constraint(model, vcat(p,t) in Clarabel.GenPowerConeT(freq,d,1))
@constraint(model, vcat(t,p) in MOI.GeometricMeanCone(d+1))
optimize!(model)

# println("generalized power cones via Hypatia")
# using Hypatia
# model = Model(Hypatia.Optimizer)
# @variable(model, p[1:d])
# @variable(model, t)
# @objective(model, Max, t)
# @constraint(model, sum(p) == 1)
# # @constraint(model, vcat(p,t) in Clarabel.GenPowerConeT(freq,d,1))
# @constraint(model, vcat(p,t) in Hypatia.GeneralizedPowerCone(freq,1,false))
# optimize!(model)

println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Max, t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(p,t) in Clarabel.GenPowerConeT(freq,d,1))
optimize!(model)