using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using JLD,JLD2
# include(".\\..\\..\\src\\Clarabel.jl")
using Clarabel
using Hypatia
using BenchmarkTools

"""
Discrete maximum likelihood from Hypatia.jl,

https://github.com/chriscoey/Hypatia.jl/tree/master/examples/discretemaxlikelihood,
"""
d = 500
freq = Float64.(rand(1:(2 * d), d))
freq ./= sum(freq)      # normalize the sum to be 1


#######################################################################
# YC: Benchmarking should be implemented separately if you want to 
#     obtain the plot.
#######################################################################


#Result from Clarabel's power mean cone
println("power mean cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Min, -t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(p,t) in Clarabel.PowerMeanConeT(freq,d))
# MOI.set(model, MOI.Silent(), true)      #Diable printing information
optimize!(model)


#Result from Hypatia
println("power mean cones via Hypatia")
model = Model(Hypatia.Optimizer)
@variable(model, p[1:d])
@variable(model, t)
@objective(model, Min, -t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(t,p) in Hypatia.HypoPowerMeanCone(freq,false))
MOI.set(model, MOI.Silent(), true)      #Diable printing information
optimize!(model)
println(solution_summary(model))
