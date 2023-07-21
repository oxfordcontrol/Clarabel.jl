using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

@testset "Basic Equality Constrained Tests" begin

    for FloatT in UnitTestFloats

        @testset "Equality constraint tests (T = $(FloatT))" begin

            tol = FloatT(1e-3)

            @testset "equality constrained" begin

                P = sparse(I(3).*one(FloatT))
                c = zeros(FloatT,3)
                A = sparse(FloatT[0. 1. 1.;0. 1. -1.]) #two constraints only
                b = FloatT[2.,0.]
                cones = [Clarabel.ZeroConeT(2)]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test isapprox(norm(solver.solution.x - FloatT[0., 1., 1.]), zero(FloatT), atol=tol)
                @test solver.solution.status == Clarabel.SOLVED

            end

            @testset "equality constrained (2)" begin

                P = sparse(I(3).*one(FloatT))
                c = FloatT[1., 2., 3.]
                A = sparse(FloatT[1. 1. 1.;0. 1. -1.]) #two constraints only
                b = FloatT[2.,0.]
                cones = [Clarabel.ZeroConeT(2)]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test isapprox(norm(solver.solution.x - FloatT[10., 1., 1.]./6), zero(FloatT), atol=tol)
                @test solver.solution.status == Clarabel.SOLVED

            end

            @testset "equality constrained (redundant rows)" begin

                P = sparse(I(3).*one(FloatT))
                c = zeros(FloatT,3)
                A = sparse(FloatT[0. 1. 1.;0. 1. -1.]) #two constraints only
                b = FloatT[2.,0.]
                cones = [Clarabel.ZeroConeT(2)]
                A = sparse([A;A])
                b = [b;b]
                cones = [cones;cones]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test isapprox(norm(solver.solution.x - FloatT[0., 1., 1.]), zero(FloatT), atol=tol)
                @test solver.solution.status == Clarabel.SOLVED

            end

            @testset "equality constrained primal infeasible" begin

                P = sparse(I(3).*one(FloatT))
                c = zeros(FloatT,3)
                A = sparse(FloatT[
                    0.  1. 1.;
                    0.  1. -1.;
                    1.  2. -1.;
                    2. -1.  3.]) #4 constraints, 3 vars
                b = ones(FloatT,4)
                cones = [Clarabel.ZeroConeT(4)]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.PRIMAL_INFEASIBLE

            end

            @testset "equality constrained dual infeasible" begin

                P = sparse(I(3).*one(FloatT))
                P[1,1] = 0;
                c = ones(FloatT,3)
                A = sparse(FloatT[0. 1. 1.;0. 1. -1.]) #two constraints only
                b = FloatT[2.,0.]
                cones = [Clarabel.ZeroConeT(2)]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE

            end


        end      #end "tests (FloatT)"
    end
end # UnitTestFloats
