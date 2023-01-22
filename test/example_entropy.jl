using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using JLD,JLD2
include(".\\..\\src\\Clarabel.jl")
# using Clarabel

"""
Discrete maximum likelihood from Hypatia.jl,

https://github.com/chriscoey/Hypatia.jl/tree/master/examples/discretemaxlikelihood,
"""
d = 3
dim = 7

#Result from Clarabel
println("entropy cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, q[1:d])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, sum(p) == 1)
@constraint(model, vcat(t,p,q) in Clarabel.EntropyConeT(dim))
optimize!(model)
