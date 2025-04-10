using Test, LinearAlgebra, SparseArrays, StableRNGs

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_SOCP_data(Type::Type{T}) where {T <: AbstractFloat}

    rng = StableRNGs.StableRNG(242713)
    n = 3
    P = randn(rng,n,n)*1
    P = SparseMatrixCSC{T}(convert(Matrix{T},(P'*P)))
    A = SparseMatrixCSC{T}(I(n)*one(T))
    A1 = [A;-A]*2
    c = T[0.1;-2.;1.]
    b1 = ones(T,6)
    cones = Clarabel.SupportedCone[Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(3)]


    #add a SOC constraint
    A2 = SparseMatrixCSC{T}(I(n)*one(T))
    b2 = [0;0;0]
    A = [A1; A2]
    b = [b1; b2]
    push!(cones,Clarabel.SecondOrderConeT(3))

    return (P,c,A,b,cones)
end


@testset "Basic SOCP Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-3)

        @testset "Basic SOCP Tests (T = $(FloatT))" begin

            @testset "feasible" begin

                P,c,A,b,cones = basic_SOCP_data(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.SOLVED
                @test isapprox(
                    norm(solver.solution.x - FloatT[-0.127715, 0.108242, -0.067784]),
                    zero(FloatT);
                    atol=tol,
                )
                @test isapprox(solver.solution.obj_val, FloatT(-0.148520), atol=tol)
                @test isapprox(solver.solution.obj_val_dual, FloatT(-0.148520), atol=tol)


            end

            @testset "feasible_sparse" begin

                # same data, but with one SOC cone so that we get the
                # sparse representation for code coverage
                P,c,A,b,cones = basic_SOCP_data(FloatT)
                cones = Clarabel.SupportedCone[Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(6)]
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.SOLVED

            end

            @testset "infeasible" begin

                P,c,A,b,cones = basic_SOCP_data(FloatT)
                b[7] = -10.;

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.PRIMAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)

            end

        end      #end "Basic SOCP Tests (FloatT)"

    end # UnitTestFloats

end #"Basic SOCP Tests"

nothing
