using LinearAlgebra, SparseArrays, JuMP
using Mosek, MosekTools
# using ECOS
using Clarabel
# include("../src\\Clarabel.jl")

# load example data
f = open(joinpath(@__DIR__, "chip_data.txt"))
lines = readlines(f)
close(f)
n_data = length(lines)
n_half = div(n_data, 2)
x1 = zeros(n_data)
x2 = zeros(n_data)
y = zeros(Float64, n_data)
for (i, l) in enumerate(lines)
    s = split(l, ",")
    x1[i] = parse(Float64, s[1])
    x2[i] = parse(Float64, s[2])
    y[i] = parse(Float64, s[3])
end

function map_feature(x1, x2)
    deg = 6
    x_new = ones(length(x1))
    for i = 1:deg, j = 0:i
        x_new = hcat(x_new, x1.^(i-j) .* x2.^j)
    end
    return x_new
  end

X = map_feature(x1, x2);
size(X)

n_theta = size(X, 2)
n = n_data
μ  = 1.

# m = JuMP.Model(ECOS.Optimizer)
# @variable(m, v)
# @variable(m, θ[1:n_theta])
# @variable(m, e[1:n])
# @variable(m, t1[1:n])
# @variable(m, t2[1:n])
# @variable(m, s1[1:n])
# @variable(m, s2[1:n])

# @objective(m, Min, μ * v + sum(e))
# @constraint(m, [v; θ] in MOI.SecondOrderCone(n_theta + 1))

# # create the constraints for each sample points
# for i = 1:n
#   yi = y[i]
#   x = X[i, :]
#   yi == 1. ? (a = -1) : (a = 1)
#   @constraint(m, [a * dot(θ, x) - e[i]; s1[i]; t1[i] ] in MOI.ExponentialCone())
#   @constraint(m, [-e[i]; s2[i]; t2[i]] in MOI.ExponentialCone())
#   @constraint(m, t1[i] + t2[i] <= 1)
#   @constraint(m, s1[i] == 1)
#   @constraint(m, s2[i] == 1)
# end
# JuMP.optimize!(m)



#####################################################
# m = JuMP.Model(ECOS.Optimizer)

m = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(m, "direct_solve_method", :qdldl)
set_optimizer_attribute(m, "static_regularization_constant", 1e-8)
set_optimizer_attribute(m, "tol_gap_abs", 1e-8)
set_optimizer_attribute(m, "tol_gap_rel", 1e-8)
# set_optimizer_attribute(m, "max_iter", 10)

@variable(m, v)
@variable(m, θ[1:n_theta])
@variable(m, e[1:n])
@variable(m, t1[1:n])
@variable(m, t2[1:n])
@variable(m, s1[1:n])
@variable(m, s2[1:n])

@objective(m, Min, μ * v + sum(e))
@constraint(m, [v; θ] in MOI.SecondOrderCone(n_theta + 1))

# create the constraints for each sample points
for i = 1:n
  yi = y[i]
  x = X[i, :]
  yi == 1. ? (a = -1) : (a = 1)
  @constraint(m, [a * dot(θ, x) - e[i]; s1[i]; t1[i] ] in MOI.ExponentialCone())
  @constraint(m, [-e[i]; s2[i]; t2[i]] in MOI.ExponentialCone())
  @constraint(m, t1[i] + t2[i] <= 1)
  @constraint(m, s1[i] == 1)
  @constraint(m, s2[i] == 1)
end
JuMP.optimize!(m)
