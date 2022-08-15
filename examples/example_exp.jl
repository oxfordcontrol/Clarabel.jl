# include("../src\\Clarabel.jl")
using Clarabel
using LinearAlgebra, SparseArrays
using JuMP, Mosek, MosekTools, ECOS
import MathOptInterface
# const MOI = MathOptInterface

using Profile,StatProfilerHTML, TimerOutputs
# using Debugger
function expconeData(Type::Type{T}) where {T<: AbstractFloat}

    #x is of dimension 7
    A1 = hcat(ones(T,1,3), zeros(T,1,4))        #ZeroCone
    b1 = T(10.)
    A2 = hcat(zeros(T,3,2), - T.(Matrix(1.0I, 3, 3)), zeros(T,3,2))       #NNCone
    b2 = zeros(T,3)
    A3 = hcat(zeros(T,3,4), - T.(Matrix(1.0I, 3, 3)))       #socone
    b3 = zeros(T,3)
    A4 = zeros(T,3,7)             #psdcone, problematic
    A4[1,2] = T(1.)
    A4[2,4] = T(1.0)
    A4[3,6] = T(1.)
    b4 = T.([10.;0;10.])
    A5 = zeros(T,3,7)               #expcone
    A5[1,1] = T(-1.0)
    A5[2,3] = T(-1.)
    A5[3,5] = T(-1.)
    b5 = zeros(T,3)
    A6 = zeros(T,3,7)               #powcone
    A6[1,1] = T(-1.0)
    A6[2,6] = T(-1.)
    A6[3,7] = T(-1.)
    b6 = zeros(T,3)

    c = T.([1.0; 0.5; -2.; -0.1; 1.0; 3.; 0.])
    P = spzeros(T,length(c), length(c))
    P = sparse(I(length(c)).*T(1e-1))

    # A = sparse([A1;A2;A3;A4])
    # b = [b1;b2;b3;b4]
    A = sparse([A1;A2;A3;A5;A6])
    b = [b1;b2;b3;b5;b6]
    # A = sparse([A1;A2;A5])
    # b = [b1;b2;b5]
    # A = sparse([A1;A2;A3])
    # b = [b1;b2;b3]
    # A = sparse([A5;A6])
    # b = [b5;b6]

    cones = [
    Clarabel.ZeroConeT(length(b1)),
    Clarabel.NonnegativeConeT(length(b2)),
    Clarabel.SecondOrderConeT(length(b3)),
    # Clarabel.PSDTriangleConeT,
    Clarabel.ExponentialConeT(),
    # Clarabel.PowerConeT(3.0/4),
    Clarabel.PowerConeT(1.0/3),
    ]



    α = Vector{Union{T,Nothing}}([
        nothing;
        nothing;
        nothing;
        # nothing;
        # nothing;
        3.0/4;
        1.0/3;
        ])

    return (P,c,A,b,cones,A1,A2,A3,A4,A5,A6,b1,b2,b3,b4,b5,b6,α)

end

# set data type first
T = Float64
# T = BigFloat
P,c,A,b,cones,A1,A2,A3,A4,A5,A6,b1,b2,b3,b4,b5,b6,α = expconeData(T)
n = 7

# using Hypatia

# println("\n\nJuMP\n-------------------------\n\n")
# opt = Mosek.Optimizer()
# model = Model(() -> opt)
# @variable(model, x[1:n])
# @constraint(model, c1, A1*x .== b1)
# @constraint(model, c2, A2*x .<= b2)
# # @constraint(model, c3, b3-A3*x in MOI.SecondOrderCone(cone_dims[3]))
# # @constraint(model, c4, b4-A4*x in MOI.PositiveSemidefiniteConeTriangle(cone_dims[4]))
# @constraint(model, c5, b5-A5*x in MOI.ExponentialCone())
# # @constraint(model, c6, b6-A6*x in MOI.PowerCone(Float64(α[end])))
# @objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

# #Run the opimization
# optimize!(model)

# settings = Clarabel.Settings{BigFloat}(max_iter=50,direct_kkt_solver=true, equilibrate_enable = true)
# solver   = Clarabel.Solver{BigFloat}()
# α =  Vector{Union{BigFloat,Nothing}}([
#         nothing;
#         nothing;
#         nothing;
#         # nothing;
#         nothing;
#         # BigFloat(1)/3
#         ])
# Clarabel.setup!(solver,BigFloat.(P),BigFloat.(c),BigFloat.(A),BigFloat.(b),cones,cone_dims,α,settings)
# Clarabel.solve!(solver)

settings = Clarabel.Settings{Float64}(max_iter=50,direct_solve_method = :qdldl)
solver   = Clarabel.Solver{Float64}()
Clarabel.setup!(solver,P,c,A,b,cones,settings)
Clarabel.solve!(solver)
