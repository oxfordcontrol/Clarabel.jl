using LinearAlgebra, SparseArrays
# using Clarabel
include(".\\..\\src\\Clarabel.jl")

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_genpow_data(Type::Type{T}) where {T <: AbstractFloat}
 
    #x is of dimension 10
    # x = (u, w, x1, y1, z1)
    n = 10
    P = spzeros(T, n, n)
    q = zeros(T, n)
    q[3] = q[6] = -one(T)
    q[10] = one(T)
    cones = Clarabel.SupportedCone[]

    # (u,w) in K_genpow([0.1; 0.3; 0.6]), w ∈ R^{4}
    # and (x1, y1, z1) in K_pow(0.6) 
    A1 = -spdiagm(0 => ones(T, n))
    b1 = zeros(T, n)
    α = [0.1; 0.3; 0.6]
    dim1 = 3
    dim2 = 4
    push!(cones,Clarabel.GenPowerConeT(α,dim1,dim2))    
    push!(cones,Clarabel.PowerConeT(0.6))
    
    # u1 + 2w3 + 3x1 == 3
    A2 = T[1.0 0 0 0 0 2.0 0 1.0 0 0]
    b2 = T[3.]
    push!(cones,Clarabel.ZeroConeT(1))

    # u1 >= 1
    # w2 >= -1
    A3 = zeros(T,2,n)
    A3[1,1] = -1.0
    A3[2,5] = -1.0
    b3 = T[-1.; 1.]
    push!(cones,Clarabel.NonnegativeConeT(2))

    A = -sparse([A1;A2;A3])
    b = [b1;b2;b3]    

    return (P,q,A,b,cones)
end

P,q,A,b,cones = basic_genpow_data(Float64)
solver = Clarabel.Solver(P,q,A,b,cones)
Clarabel.solve!(solver)