using LinearAlgebra, SparseArrays
# using Clarabel
using JuMP,Mosek,MosekTools
using JLD,JLD2
# include("..\\..\\src\\Clarabel.jl")
using Clarabel
using Hypatia

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_powm_data(Type::Type{T}) where {T <: AbstractFloat}
 
    #x is of dimension 6
    # x = (u1, u2, w, x1, y1, z1)
    n = 6
    # P = spzeros(T, n, n)
    P = T.(spdiagm(0 => rand(n)))
    q = zeros(T, n)
    q[3] = q[6] = -one(T)
    cones = Clarabel.SupportedCone[]

    # (u1,u2,w) in K_powm([0.3; 0.7]),
    # and (x1, y1, z1) in K_powm([0.6,0.4]) 
    A1 = spdiagm(0 => ones(T, n))
    b1 = zeros(T, n)
    α11 = T.([0.3; 0.7])
    α12 = T.([0.6;0.4])
    dim1 = 2
    dim2 = 1
    push!(cones,Clarabel.PowerMeanConeT(α11,dim1))    
    push!(cones,Clarabel.PowerMeanConeT(α12,dim1))    
    
    # u1 + 3x1 == 3
    A2 = T[-1.0 0 0 -3.0 0 0]
    b2 = T[3.]
    push!(cones,Clarabel.ZeroConeT(1))

    # u1 >= 1
    # x2 >= -1
    A3 = zeros(T,2,n)
    A3[1,1] = 1.0
    A3[2,5] = 1.0
    b3 = T[-1.; 1.]
    push!(cones,Clarabel.NonnegativeConeT(2))

    # [u1, x2, x3] ∈ SOC 
    A4 = zeros(T,3,n)
    A4[1,1] = 1.0
    A4[2,5] = 1.0
    A4[3,6] = 1.0
    b4 = zeros(T,3)
    push!(cones,Clarabel.SecondOrderConeT(3))

    A = -sparse([A1;A2;A3;A4])  #take minus sign here 
    b = [b1;b2;b3;b4]    

    return (P,q,A,b,A1,A2,A3,A4,b1,b2,b3,b4,cones)
end

T = Float64
P,q,A,b,A1,A2,A3,A4,b1,b2,b3,b4,cones = basic_powm_data(T)

# P,q,A,b,cones,T,A1,A2,A3,A4,b1,b2,b3,b4 = load("test\\genpow_data.jld", "P", "q", "A", "b", "cones", "T", "A1", "A2", "A3", "A4", "b1", "b2", "b3", "b4")

n = size(A,2)
println("\n\nClarabel\n-------------------------\n\n")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
# @constraint(model, c1, b1[1:3]-A1[1:3,:]*x in MOI.PowerCone(0.3))
# @constraint(model, c2, b1[4:6]-A1[4:6,:]*x in MOI.PowerCone(0.6))
@constraint(model, c11, x[3] >= 0)
@constraint(model, c12, x[6] >= 0)
@constraint(model, c1, x[1:3] in Clarabel.PowerMeanConeT([0.3,0.7],2))
@constraint(model, c2, x[4:6] in Clarabel.PowerMeanConeT([0.6,0.4],2))
# @constraint(model, c3, b2-A2*x .== 0.)
# @constraint(model, c4, b3-A3*x .>= 0.)
# @constraint(model, c5, b4-A4*x in MOI.SecondOrderCone(3))

# set_optimizer_attribute(model,"equilibrate_enable", false)
@objective(model, Min, sum(q.*x) + 1/2*x'*P*x)
optimize!(model)

# settings = Clarabel.Settings{T}(
#                             iterative_refinement_reltol = 1e-16, iterative_refinement_abstol = 1e-16,
# #                             # tol_feas = 1e-6, tol_gap_abs = 1e-6, tol_gap_rel = 1e-6,
#                             )
# solver = nothing
# solver = Clarabel.Solver{T}()
# Clarabel.setup!(solver,P,q,A,b,cones)
# Clarabel.solve!(solver)


println("\n\nMosek\n-------------------------\n\n")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@constraint(model, c1, x[1:3] in MOI.PowerCone(0.3))
@constraint(model, c2, x[4:6] in MOI.PowerCone(0.6))
@constraint(model, c11, x[3] >= 0)
@constraint(model, c12, x[6] >= 0)
# @constraint(model, c1, b1[1:3]-A1[1:3,:]*x in Clarabel.PowerMeanConeT([0.3,0.7],2))
# @constraint(model, c2, b1[4:6]-A1[4:6,:]*x in Clarabel.PowerMeanConeT([0.6,0.4],2))
# @constraint(model, c3, b2-A2*x .== 0.)
# @constraint(model, c4, b3-A3*x .>= 0.)
# @constraint(model, c5, b4-A4*x in MOI.SecondOrderCone(3))

# set_optimizer_attribute(model,"equilibrate_enable", false)
@objective(model, Min, sum(q.*x) + 1/2*x'*P*x)
optimize!(model)

println("\n\nHypatia\n-------------------------\n\n")
# model = Model(Clarabel.Optimizer)
model = Model(Hypatia.Optimizer)
@variable(model, x[1:n])
# @constraint(model, c1, b1[1:3]-A1[1:3,:]*x in MOI.PowerCone(0.3))
# @constraint(model, c2, b1[4:6]-A1[4:6,:]*x in MOI.PowerCone(0.6))
@constraint(model, c11, x[3] >= 0)
@constraint(model, c12, x[6] >= 0)
@constraint(model, c1, vcat(x[3], x[1:2]) in Hypatia.HypoPowerMeanCone([0.3,0.7],false))
@constraint(model, c2, vcat(x[6], x[4:5]) in Hypatia.HypoPowerMeanCone([0.6,0.4],false))
# @constraint(model, c3, b2-A2*x .== 0.)
# @constraint(model, c4, b3-A3*x .>= 0.)
# @constraint(model, c5, b4-A4*x in MOI.SecondOrderCone(3))

@objective(model, Min, sum(q.*x) + 1/2*x'*P*x)
optimize!(model)