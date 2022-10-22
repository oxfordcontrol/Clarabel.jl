using LinearAlgebra, SparseArrays
# using Clarabel
using JuMP,Mosek,MosekTools
include(".\\..\\src\\Clarabel.jl")

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_genpow_data(Type::Type{T}) where {T <: AbstractFloat}
 
    #x is of dimension 10
    # x = (u, w, x1, y1, z1)
    n = 6
    # P = spzeros(T, n, n)
    P = spdiagm(0 => rand(n))
    q = zeros(T, n)
    q[3] = q[6] = -one(T)
    cones = Clarabel.SupportedCone[]

    # (u,w) in K_genpow([0.1; 0.3; 0.6]), w ∈ R^{4}
    # and (x1, y1, z1) in K_pow(0.6) 
    A1 = spdiagm(0 => ones(T, n))
    b1 = zeros(T, n)
    α = [0.3; 0.7]
    dim1 = 2
    dim2 = 1
    # push!(cones,Clarabel.PowerConeT(0.3))
    push!(cones,Clarabel.GenPowerConeT(α,dim1,dim2))    
    # push!(cones,Clarabel.PowerConeT(0.6))
    push!(cones,Clarabel.GenPowerConeT([0.6;0.4],dim1,dim2))    
    
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

    A = -sparse([A1;A2;A3;A4])
    b = [b1;b2;b3;b4]    

    return (P,q,A,b,A1,A2,A3,A4,b1,b2,b3,b4,cones)
end

function basic_genpow_data_2(Type::Type{T}) where {T <: AbstractFloat}
 
    #x is of dimension 10
    # x = (u, w, x1, y1, z1)
    n = 10
    # P = spzeros(T, n, n)
    P = spdiagm(0 => rand(n))
    q = zeros(T, n)
    q[3] = q[6] = -one(T)
    cones = Clarabel.SupportedCone[]

    # (u,w) in K_genpow([0.1; 0.3; 0.6]), w ∈ R^{4}
    # and (x1, y1, z1) in K_pow(0.6) 
    A1 = spdiagm(0 => ones(T, n))
    b1 = zeros(T, n)
    α = [0.3; 0.2; 0.5]
    dim1 = 3
    dim2 = 4
    # push!(cones,Clarabel.PowerConeT(0.3))
    push!(cones,Clarabel.GenPowerConeT(α,dim1,dim2))    
    push!(cones,Clarabel.PowerConeT(0.6))
    # push!(cones,Clarabel.GenPowerConeT([0.6;0.4],dim1,dim2))    
    
    # u1 + w2 + 3x1 == 3
    A2 = T[-1.0 0 0 0 1.0 0 0 -3.0 0 0]
    b2 = T[3.]
    push!(cones,Clarabel.ZeroConeT(1))

    # u1 >= 1
    # w2 >= -1
    A3 = zeros(T,2,n)
    A3[1,1] = 1.0
    A3[2,5] = 1.0
    b3 = T[-1.; 1.]
    push!(cones,Clarabel.NonnegativeConeT(2))

    # # [u1, x2, x3] ∈ SOC 
    # A4 = zeros(T,3,n)
    # A4[1,1] = 1.0
    # A4[2,5] = 1.0
    # A4[3,6] = 1.0
    # b4 = zeros(T,3)
    # push!(cones,Clarabel.SecondOrderConeT(3))

    A = -sparse([A1;A2;A3])
    b = [b1;b2;b3]    

    return (P,q,A,b,A1,A2,A3,b1,b2,b3,cones)
end

P,q,A,b,A1,A2,A3,A4,b1,b2,b3,b4,cones = basic_genpow_data(Float64)
# P,q,A,b,A1,A2,A3,b1,b2,b3,cones = basic_genpow_data_2(Float64)


# n = size(A,2)
# println("\n\nJuMP\n-------------------------\n\n")
# opt = Mosek.Optimizer()
# model = Model(() -> opt)
# @variable(model, x[1:n])
# @constraint(model, c1, b1[1:3]-A1[1:3,:]*x in MOI.PowerCone(0.3))
# @constraint(model, c2, b1[4:6]-A1[4:6,:]*x in MOI.PowerCone(0.6))
# @constraint(model, c3, b2-A2*x .== 0.)
# @constraint(model, c4, b3-A3*x .>= 0.)
# @constraint(model, c5, b4-A4*x in MOI.SecondOrderCone(3))

# @objective(model, Min, sum(q.*x) + 1/2*x'*P*x)
# optimize!(model)

setting = Clarabel.Settings(
                            # iterative_refinement_reltol = 1e-14, iterative_refinement_abstol = 1e-14,
                            # tol_feas = 1e-6, tol_gap_abs = 1e-6, tol_gap_rel = 1e-6,
                            )
solver = Clarabel.Solver(P,q,A,b,cones,setting)

Clarabel.solve!(solver)

