using Test, LinearAlgebra, SparseArrays, Clarabel


#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_genpow_data(Type::Type{T}) where {T <: AbstractFloat}

    #x is of dimension 6
    # x = (x1, y, z1, x2, y2, z2)
    n = 6
    P = spzeros(T, n, n)
    q = zeros(T, 6)
    q[3] = q[6] = -one(T)
    A = sparse(T[-1. 0. 0. 0. 0. 0.;
    0. -1. 0. 0. 0. 0.;
    0. 0. -1. 0. 0. 0.;
    0. 0. 0. -1. 0. 0.;
    0. 0. 0. 0. -1. 0.;
    0. 0. 0. 0. 0. -1.;
    1. 2. 0. 3. 0. 0.;
    0. 0. 0. 0. 1. 0.]
    )
    b = T[0., 0., 0., 0., 0., 0., 3., 1.]
    cones = Clarabel.SupportedCone[]

    push!(cones,Clarabel.GenPowerConeT([0.6,0.4],1))
    push!(cones,Clarabel.GenPowerConeT([0.1,0.9],1))
    push!(cones,Clarabel.ZeroConeT(2))

    return (P,q,A,b,cones)
end


@testset "Basic GenPowerCone Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-3)

        @testset "Basic GenPowerCone Tests (T = $(FloatT))" begin

            @testset "feasible" begin

                P,c,A,b,cones = basic_genpow_data(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.info.status == Clarabel.SOLVED
                @test isapprox(solver.info.cost_primal, FloatT(-1.8458), atol=tol)

            end

        end      #end "Basic GenPowerCone Tests (FloatT)"

    end # UnitTestFloats

end #"Basic GenPowerCone Tests"

nothing
