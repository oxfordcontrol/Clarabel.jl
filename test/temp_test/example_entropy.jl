using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools,ECOS
# using JLD,JLD2
# include(".\\..\\src\\Clarabel.jl")
using Clarabel
# using Hypatia
using BenchmarkTools

d = 5000
dim = 2*d+1
p_init = rand(d)
# p_init = ones(d)
p_init = p_init./sum(p_init)
p_init[1] += 1-sum(p_init)

#Result from three-dimensional exponential cones Clarabel
println("three-dimensional exponential cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, q[1:d])
@variable(model, r[1:d])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, sum(q) == 1)
@constraint(model, p .== p_init)
@constraint(model, -sum(r) <= t)
for i = 1:d
    @constraint(model, vcat(r[i],q[i],p[i]) in MOI.ExponentialCone())
end
optimize!(model)

#Result from Clarabel
println("entropy cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, p[1:d])
@variable(model, q[1:d])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, sum(q) == 1)
@constraint(model, p .== p_init)
@constraint(model, vcat(t,p,q) in Clarabel.EntropyConeT(dim))
optimize!(model)


#Result from three-dimensional exponential cones ECOS
println("three-dimensional exponential cones via ECOS")
model = Model(ECOS.Optimizer)
@variable(model, p[1:d])
@variable(model, q[1:d])
@variable(model, r[1:d])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, sum(q) == 1)
@constraint(model, p .== p_init)
@constraint(model, -sum(r) <= t)
for i = 1:d
    @constraint(model, vcat(r[i],q[i],p[i]) in MOI.ExponentialCone())
end
# @constraint(model, vcat(t,p,q) in MOI.RelativeEntropyCone(dim))
optimize!(model)


#Result from Mosek
println("entropy cones via Mosek")
model = Model(Mosek.Optimizer)
@variable(model, p[1:d])
@variable(model, q[1:d])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, sum(q) == 1)
@constraint(model, p .== p_init)
@constraint(model, vcat(t,p,q) in MOI.RelativeEntropyCone(dim))
optimize!(model)

# #Result from Hypatia
# using Hypatia
# println("entropy cones via Hypatia")
# model = Model(Hypatia.Optimizer)
# @variable(model, p[1:d])
# @variable(model, q[1:d])
# @variable(model, t)
# @objective(model, Min, t)
# @constraint(model, sum(q) == 1)
# @constraint(model, p .== p_init)
# @constraint(model, vcat(t,p,q) in Hypatia.EpiRelEntropyCone{Float64}(dim))
# optimize!(model)

# ############################################################
# # Another example from signomial and polynomial Optimization, 
# # from https://github.com/chriscoey/Hypatia.jl/tree/master/examples/signomialmin
# ############################################################

# instance = SignomialMinJuMP{Float64}(5,10)

# # model = Model(Clarabel.Optimizer)
# # model = build(instance,model)
# model = clarabel_build(instance)

# set_optimizer_attribute(model, "max_iter", 500)
# optimize!(model)