using LinearAlgebra, SparseArrays
using JuMP,Mosek,MosekTools
using Clarabel

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_genpow_data(Type::Type{T}) where {T <: AbstractFloat}
 
    n = 6
    # P = spzeros(T, n, n)
    P = T.(spdiagm(0 => rand(n)))
    q = zeros(T, n)
    q[3] = q[6] = -one(T)
    cones = Clarabel.SupportedCone[]

    # Two power cone constraints but can be represented as GeneralizedPowerCone
    A1 = spdiagm(0 => ones(T, n))
    b1 = zeros(T, n)
    α11 = T.([0.3; 0.7])
    α12 = T.([0.6;0.4])
    dim1 = 2
    dim2 = 1
    # push!(cones,Clarabel.PowerConeT(T(0.3)))
    push!(cones,Clarabel.GenPowerConeT(α11,dim1,dim2))    
    # push!(cones,Clarabel.PowerConeT(T(0.6)))
    push!(cones,Clarabel.GenPowerConeT(α12,dim1,dim2))    
    
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

T = Float64
P,q,A,b,A1,A2,A3,A4,b1,b2,b3,b4,cones = basic_genpow_data(T)

n = size(A,2)
println("\n\nJuMP\n-------------------------\n\n")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@constraint(model, c1, b1[1:3]-A1[1:3,:]*x in Clarabel.GenPowerConeT([0.3,0.7],2,1))
@constraint(model, c2, b1[4:6]-A1[4:6,:]*x in MOI.PowerCone(0.6))
@constraint(model, c3, b2-A2*x .== 0.)
@constraint(model, c4, b3-A3*x .>= 0.)
@constraint(model, c5, b4-A4*x in MOI.SecondOrderCone(3))

@objective(model, Min, sum(q.*x) + 1/2*x'*P*x)
optimize!(model)
