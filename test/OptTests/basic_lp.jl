using Test, LinearAlgebra, SparseArrays
FloatT = Float64
tol = FloatT(1e-3)

@testset "Basic LP Tests" begin

    function basic_LP_data(Type::Type{T}) where {T <: AbstractFloat}

        P = spzeros(T,3,3)
        A = sparse(I(3)*T(1.))
        A = [A;-A].*2
        c = [3.;-2.;1.]
        b = ones(T,6)
        cone_types = [Clarabel.NonnegativeConeT, Clarabel.NonnegativeConeT]
        cone_dims  = [3,3]

        return (P,c,A,b,cone_types,cone_dims)
    end

    @testset "Basic LP Tests (T = $(FloatT))" begin

        @testset "feasible" begin

            P,c,A,b,cone_types,cone_dims = basic_LP_data(FloatT)
            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.SOLVED
            @test isapprox(norm(solver.variables.x - FloatT[-0.5; 0.5; -0.5]), zero(FloatT), atol=tol)
            @test isapprox(solver.info.cost_primal, FloatT(-3.), atol=tol)

        end

        @testset "primal infeasible" begin

            P,c,A,b,cone_types,cone_dims = basic_LP_data(FloatT)
            b[1] = -1
            b[4] = -1

            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.PRIMAL_INFEASIBLE

        end

        @testset "dual infeasible" begin

            P,c,A,b,cone_types,cone_dims = basic_LP_data(FloatT)
            A[4,1] = 1   #swap lower bound on first variable to redundant upper
            c .= [1.;0;0]

            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.DUAL_INFEASIBLE

        end

        @testset "dual infeasible (ill conditioned)" begin

            P,c,A,b,cone_types,cone_dims = basic_LP_data(FloatT)
            A[1,1] = eps(FloatT)
            A[4,1] = -eps(FloatT)
            c .= [1.;0;0]

            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.DUAL_INFEASIBLE

        end

        @testset "dual infeasible (rank deficient KKT)" begin

            P,c,A,b,cone_types,cone_dims = basic_LP_data(FloatT)
            A[1,1] = eps(FloatT)
            A[4,1] = -eps(FloatT)
            c .= [1.;0;0]

            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.DUAL_INFEASIBLE

        end

    end # UnitTestFloats
end
nothing
