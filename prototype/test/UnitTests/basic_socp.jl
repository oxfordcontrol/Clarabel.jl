using Test, LinearAlgebra, Statistics, Random
FloatT = Float64
tol = FloatT(1e-3)

@testset "Basic SOCP Tests" begin

    function basic_SOCP_data(Type::Type{T}) where {T <: AbstractFloat}

        n = 3
        P = randn(T,n,n)*1
        P = SparseMatrixCSC{T}(P'*P)
        A = SparseMatrixCSC{T}(I(n)*1.)
        A = [A;-A]*2
        c = T[0.1;-2.;1.]
        b = ones(T,6)
        cone_types = [IPSolver.NonnegativeConeT, IPSolver.NonnegativeConeT]
        cone_dims  = [3,3]

        #add a SOC constraint
        Asoc = SparseMatrixCSC{T}(I(n)*1.)
        A = [A;-Asoc]
        push!(b,0.,0.,0.)
        push!(cone_dims,3)
        push!(cone_types,IPSolver.SecondOrderConeT)

        return (P,c,A,b,cone_types,cone_dims)
    end

    @testset "Basic SOCP Tests (T = $(FloatT))" begin

        @testset "feasible" begin

            P,c,A,b,cone_types,cone_dims = basic_SOCP_data(FloatT)
            solver   = IPSolver.Solver(P,c,A,b,cone_types,cone_dims)
            IPSolver.solve!(solver)

            @test solver.info.status == IPSolver.SOLVED
            @test isapprox(
                norm(solver.variables.x -
                FloatT[0.3874336317389936, 0.3874318938275443, 0.0011605535861680225]),
                zero(FloatT), atol=tol)
            @test isapprox(solver.info.cost_primal, FloatT(-3.6748e-01), atol=tol)

        end

    end # UnitTestFloats
end
nothing
