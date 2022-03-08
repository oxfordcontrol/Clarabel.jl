using Test, LinearAlgebra, SparseArrays
FloatT = Float64
tol = FloatT(1e-3)

@testset "Basic QP Tests" begin

    function basic_QP_data(Type::Type{T}) where {T <: AbstractFloat}
        P = sparse(T[4. 1;1 2])
        c = T[1; 1.]
        A = sparse(T[1. 1;1 0; 0 1])
        l = T[1.;0;0]
        u = T[1.;0.7;0.7]

        cone_types = [Clarabel.NonnegativeConeT, Clarabel.NonnegativeConeT]
        cone_dims  = [3,3]

        A = [-A;A]
        b = [-l;u]

        return (P,c,A,b,cone_types,cone_dims)
    end

    function basic_QP_data_dualinf(Type::Type{T}) where {T <: AbstractFloat}

        #x = [1;-1] is in ker(P) and always feasible
        P = sparse(T[1. 1.;1. 1.])
        c = T[1; -1.]
        A = sparse(T[1. 1;1. 0;])
        b = [1.;1]
        cone_types = [Clarabel.NonnegativeConeT]
        cone_dims  = [2,]

        return (P,c,A,b,cone_types,cone_dims)
    end

    @testset "Basic QP Tests (T = $(FloatT))" begin

        @testset "feasible" begin

            P,c,A,b,cone_types,cone_dims = basic_QP_data(FloatT)
            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.SOLVED
            @test isapprox(norm(solver.variables.x - FloatT[0.3; 0.7]), zero(FloatT), atol=tol)
            @test isapprox(solver.info.cost_primal, FloatT(1.8800000298331538), atol=tol)

        end

        @testset "primal infeasible" begin

            P,c,A,b,cone_types,cone_dims = basic_QP_data(FloatT)
            b[1] = -1
            b[4] = -1

            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.PRIMAL_INFEASIBLE

        end

        @testset "dual infeasible" begin

            P,c,A,b,cone_types,cone_dims = basic_QP_data_dualinf(FloatT)
            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.DUAL_INFEASIBLE
        end

        @testset "dual infeasible (non-QSD KKT)" begin

            P,c,A,b,cone_types,cone_dims = basic_QP_data_dualinf(FloatT)
            A = A[1:1,:]
            b = b[1:1]
            cone_types = [Clarabel.NonnegativeConeT]
            cone_dims  = [1,]

            solver   = Clarabel.Solver(P,c,A,b,cone_types,cone_dims)
            Clarabel.solve!(solver)

            @test solver.info.status == Clarabel.DUAL_INFEASIBLE
        end

    end # UnitTestFloats
end
nothing
