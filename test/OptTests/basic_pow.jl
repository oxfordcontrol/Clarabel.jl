using Test, LinearAlgebra, SparseArrays, Random

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_pow_data(Type::Type{T}) where {T <: AbstractFloat}
 
    #x is of dimension 7
    # x = (x1, y, z1, x2, y2, z2)
    n = 6
    P = spzeros(T, n, n)
    q = zeros(T, 6)
    q[3] = q[6] = -one(T)
    cones = Clarabel.SupportedCone[]

    # (x1, y, z1) in K_pow(0.6) \times K_pow(0.1)
    A1 = spdiagm(0 => ones(T, 6))
    b1 = zeros(T, 6)
    push!(cones,Clarabel.PowerConeT(0.6))
    push!(cones,Clarabel.PowerConeT(0.1))
    
    # x1 + 2y + 3x2 == 3
    A2 = T[1.0 2.0 0 3.0 0 0]
    b2 = T[-3.]
    push!(cones,Clarabel.ZeroConeT(1))

    # y2 == 1
    A3 = T[0 0 0 0 1.0 0]
    b3 = T[-1.]
    push!(cones,Clarabel.ZeroConeT(1))

    A = -sparse([A1;A2;A3])
    b = [b1;b2;b3]    

    return (P,q,A,b,cones)
end


@testset "Basic PowerCone Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-3)

        @testset "Basic PowerCone Tests (T = $(FloatT))" begin

            @testset "feasible" begin

                P,c,A,b,cones = basic_pow_data(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.info.status == Clarabel.SOLVED
                @test isapprox(solver.info.cost_primal, FloatT(-1.8458), atol=tol)

            end

        end      #end "Basic PowerCone Tests (FloatT)"

    end # UnitTestFloats

end #"Basic PowerCone Tests"

nothing
